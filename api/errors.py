"""Central FastAPI exception handlers and safe public error envelopes."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.errors import BraindError
from core.logging import get_logger

logger = get_logger(__name__)

_INTERNAL_ERROR = {
    "code": "internal_error",
    "message": "An internal error occurred.",
}
_VALIDATION_ERROR = {
    "code": "validation_error",
    "message": "The request could not be validated.",
}


def error_payload(code: str, message: str) -> dict[str, dict[str, str]]:
    """Build the stable JSON error envelope returned by the API."""
    return {"error": {"code": code, "message": message}}


def safe_error_payload(exc: Exception) -> dict[str, dict[str, str]]:
    """Return the public-safe error payload for a caught exception."""
    if isinstance(exc, BraindError):
        return error_payload(exc.code, exc.public_message)
    return error_payload(**_INTERNAL_ERROR)


def _request_context(request: Request) -> dict[str, Any]:
    """Return non-sensitive request fields suitable for structured logs."""
    return {"method": request.method, "path": request.url.path}


def register_exception_handlers(app: FastAPI) -> None:
    """Register API-wide exception handlers with safe, stable responses."""

    @app.exception_handler(BraindError)
    async def braind_error_handler(request: Request, exc: BraindError) -> JSONResponse:
        """Translate known application exceptions into public API errors."""
        logger.warning(
            "api.braind_error",
            **_request_context(request),
            status_code=exc.status_code,
            error_code=exc.code,
            exception_type=type(exc).__name__,
        )
        return JSONResponse(safe_error_payload(exc), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Normalize schema-validation errors without echoing rejected values."""
        logger.info(
            "api.request_validation_error",
            **_request_context(request),
            status_code=422,
            error_code=_VALIDATION_ERROR["code"],
            error_count=len(exc.errors()),
        )
        return JSONResponse(error_payload(**_VALIDATION_ERROR), status_code=422)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Normalize intentional HTTP errors without exposing exception details."""
        if 400 <= exc.status_code < 500:
            status_code = exc.status_code
            payload = error_payload("http_error", "The request could not be completed.")
        else:
            status_code = 500
            payload = error_payload(**_INTERNAL_ERROR)
        logger.warning(
            "api.http_error",
            **_request_context(request),
            status_code=status_code,
            error_code=payload["error"]["code"],
        )
        return JSONResponse(payload, status_code=status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Log unexpected failures internally and never expose their details."""
        logger.error(
            "api.unhandled_error",
            **_request_context(request),
            status_code=500,
            error_code=_INTERNAL_ERROR["code"],
            exception_type=type(exc).__name__,
            exc_info=True,
        )
        return JSONResponse(error_payload(**_INTERNAL_ERROR), status_code=500)
