"""Async client for the OpenAI-compatible vLLM APIs."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from core.config import settings
from core.errors import UpstreamServiceError


class VLLMClientError(UpstreamServiceError):
    """Raised when a vLLM response cannot satisfy the client contract."""


class EmbeddingResponseError(VLLMClientError):
    """Raised when vLLM returns invalid embedding data."""


class VLLMClient:
    """Provide LLM and embedding operations through separate vLLM servers.

    vLLM exposes an OpenAI-compatible HTTP API.  The chat model and embedding
    model commonly run in different vLLM processes, so this client keeps a
    distinct base URL for each endpoint while preserving Braind's existing
    inference-client interface.
    """

    def __init__(
        self,
        llm_base_url: str | None = None,
        embed_base_url: str | None = None,
        *,
        embedding_dimensions: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a vLLM client.

        Args:
            llm_base_url: Base URL of the chat/completion vLLM server.
            embed_base_url: Base URL of the embedding vLLM server.
            embedding_dimensions: Expected length for every embedding vector.
            client: Optional injected HTTP client for deterministic testing.
        """
        self.llm_base_url = (llm_base_url or settings.vllm_llm_base_url).rstrip("/")
        self.embed_base_url = (embed_base_url or settings.vllm_embed_base_url).rstrip("/")
        self.embedding_dimensions = embedding_dimensions or settings.embedding_dimensions
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        self._owns_client = client is None

    async def __aenter__(self) -> VLLMClient:
        """Enter the client context."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Close the underlying HTTP client on context exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP connection pool."""
        if not self._client.is_closed:
            await self._client.aclose()

    async def embed(self, text: str, model: str | None = None) -> list[float]:
        """Return one validated embedding vector for ``text``."""
        return (await self.embed_batch([text], model=model))[0]

    async def embed_batch(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Return ordered, dimension-validated vectors for all input strings."""
        if not texts:
            return []

        response = await self._client.post(
            f"{self.embed_base_url}/v1/embeddings",
            json={"model": model or settings.embed_model, "input": texts},
        )
        response.raise_for_status()
        payload = response.json()
        raw_items = payload.get("data")
        if not isinstance(raw_items, list):
            raise EmbeddingResponseError("vLLM embedding response has no data list")

        ordered_items = sorted(raw_items, key=lambda item: item.get("index", -1))
        if len(ordered_items) != len(texts):
            raise EmbeddingResponseError(
                f"vLLM returned {len(ordered_items)} embeddings for {len(texts)} inputs"
            )

        embeddings: list[list[float]] = []
        for expected_index, item in enumerate(ordered_items):
            if not isinstance(item, dict) or item.get("index") != expected_index:
                raise EmbeddingResponseError("vLLM embedding indices are missing or non-contiguous")
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or len(embedding) != self.embedding_dimensions:
                actual = len(embedding) if isinstance(embedding, list) else "invalid"
                raise EmbeddingResponseError(
                    "vLLM embedding dimension mismatch: "
                    f"expected {self.embedding_dimensions}, got {actual}"
                )
            try:
                embeddings.append([float(value) for value in embedding])
            except (TypeError, ValueError) as exc:
                raise EmbeddingResponseError("vLLM embedding contains a non-numeric value") from exc
        return embeddings

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """Generate a text completion through vLLM's completion endpoint."""
        body: dict[str, Any] = {
            "model": model or settings.llm_model,
            "prompt": prompt,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens if max_tokens is not None else settings.llm_max_tokens,
            "stream": stream,
        }
        if system:
            body["prompt"] = f"{system}\n\n{prompt}"
        if stream:
            return self._stream_completion(body)

        response = await self._client.post(f"{self.llm_base_url}/v1/completions", json=body)
        response.raise_for_status()
        return self._completion_text(response.json())

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        stream: bool = False,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | AsyncIterator[str]:
        """Return an OpenAI-compatible chat completion or streaming iterator."""
        body = self._chat_body(
            messages,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream and not tools,
        )
        if tools:
            body["tools"] = tools
        if stream and not tools:
            return self._stream_chat(body)

        response = await self._client.post(f"{self.llm_base_url}/v1/chat/completions", json=body)
        self._raise_for_status(response)
        payload = response.json()
        message = self._first_message(payload)
        if tools:
            return json.dumps(
                {
                    "content": message.get("content") or "",
                    "tool_calls": message.get("tool_calls") or [],
                },
                ensure_ascii=False,
            )
        return str(message.get("content") or "")

    async def chat_json(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Request JSON-mode chat output and safely salvage valid JSON text."""
        body = self._chat_body(
            messages,
            system=system,
            model=model,
            temperature=temperature if temperature is not None else 0.2,
            max_tokens=max_tokens,
            stream=False,
        )
        body["response_format"] = {"type": "json_object"}
        response = await self._client.post(f"{self.llm_base_url}/v1/chat/completions", json=body)
        self._raise_for_status(response)
        content = str(self._first_message(response.json()).get("content") or "")
        return self._parse_json_content(content)

    async def is_available(self) -> bool:
        """Return whether both the chat and embedding vLLM servers are healthy."""
        llm_ok = await self.is_llm_available()
        embed_ok = await self.is_embedding_available()
        return llm_ok and embed_ok

    async def is_llm_available(self) -> bool:
        """Return whether the chat vLLM endpoint lists its loaded model."""
        return await self._server_is_available(self.llm_base_url)

    async def is_embedding_available(self) -> bool:
        """Return whether the embedding vLLM endpoint lists its loaded model."""
        return await self._server_is_available(self.embed_base_url)

    async def _server_is_available(self, base_url: str) -> bool:
        try:
            response = await self._client.get(f"{base_url}/v1/models")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def _chat_body(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible chat completion request body."""
        request_messages = list(messages)
        if system:
            request_messages.insert(0, {"role": "system", "content": system})
        return {
            "model": model or settings.llm_model,
            "messages": request_messages,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens if max_tokens is not None else settings.llm_max_tokens,
            "stream": stream,
        }

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Raise a typed error that keeps the upstream body on non-2xx.

        ``httpx.Response.raise_for_status`` discards the response body, which
        makes transient upstream failures (e.g. a vLLM 400) opaque in logs.
        Reading the body here surfaces the server's actual error detail.
        """
        if response.status_code < 400:
            return
        try:
            detail = response.text[:500]
        except Exception:
            detail = "<body unavailable>"
        raise VLLMClientError(
            f"vLLM returned {response.status_code} {response.reason_phrase}: {detail}"
        )

    @staticmethod
    def _first_message(payload: Any) -> dict[str, Any]:
        """Extract the first message from an OpenAI chat response."""
        try:
            message = payload["choices"][0]["message"]
        except (IndexError, KeyError, TypeError) as exc:
            raise VLLMClientError("vLLM chat response has no first message") from exc
        if not isinstance(message, dict):
            raise VLLMClientError("vLLM chat response message is invalid")
        return message

    @staticmethod
    def _completion_text(payload: Any) -> str:
        """Extract text from the first OpenAI completion choice."""
        try:
            return str(payload["choices"][0]["text"] or "")
        except (IndexError, KeyError, TypeError) as exc:
            raise VLLMClientError("vLLM completion response has no first choice") from exc

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        """Parse JSON content, falling back to a JSON object embedded in text."""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return {"raw": content}
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"raw": content}
        return parsed if isinstance(parsed, dict) else {"raw": content}

    async def _stream_completion(self, body: dict[str, Any]) -> AsyncIterator[str]:
        """Yield text chunks from a server-sent completion stream."""
        async with self._client.stream(
            "POST", f"{self.llm_base_url}/v1/completions", json=body
        ) as response:
            response.raise_for_status()
            async for payload in self._stream_payloads(response):
                choices = payload.get("choices") or []
                if choices:
                    yield str(choices[0].get("text") or "")

    async def _stream_chat(self, body: dict[str, Any]) -> AsyncIterator[str]:
        """Yield text chunks from a server-sent chat stream."""
        async with self._client.stream(
            "POST", f"{self.llm_base_url}/v1/chat/completions", json=body
        ) as response:
            response.raise_for_status()
            async for payload in self._stream_payloads(response):
                choices = payload.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    yield str(delta.get("content") or "")

    async def _stream_payloads(self, response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        """Parse OpenAI server-sent-event data frames from a response stream."""
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload