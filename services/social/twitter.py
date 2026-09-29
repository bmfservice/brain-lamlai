"""Twitter/X integration via API v2."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from core.logging import get_logger
from services.social.base import (
    BaseSocialPlatform,
    SocialAccount,
    SocialComment,
    SocialMetrics,
    SocialPost,
)

logger = get_logger(__name__)


class TwitterPlatform(BaseSocialPlatform):
    """Twitter/X API v2 integration.
    
    Requires:
    - Bearer Token (for read operations)
    - OAuth 2.0 Access Token (for write operations)
    
    Permissions: tweet.read, tweet.write, users.read
    """

    platform_name = "twitter"

    def __init__(self, credentials: dict[str, Any]):
        super().__init__(credentials)
        self.bearer_token = credentials.get("bearer_token")
        self.access_token = credentials.get("access_token")
        self.base_url = "https://api.twitter.com/2"
        self._client = httpx.AsyncClient(timeout=30.0)

    async def authenticate(self) -> bool:
        try:
            url = f"{self.base_url}/users/me"
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()
            
            self.authenticated = True
            logger.info("twitter.auth_success")
            return True
        except Exception as exc:
            logger.error("twitter.auth_failed", error=str(exc))
            return False

    async def post(self, content: SocialPost) -> dict[str, Any]:
        try:
            url = f"{self.base_url}/tweets"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }
            
            payload = {"text": content.content[:280]}  # Twitter char limit
            
            response = await self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json().get("data", {})
            tweet_id = result.get("id", "")
            
            return {
                "post_id": tweet_id,
                "url": f"https://twitter.com/i/web/status/{tweet_id}",
                "status": "published",
                "platform": "twitter",
            }
        except Exception as exc:
            logger.error("twitter.post_failed", error=str(exc))
            return {"error": str(exc), "status": "failed"}

    async def schedule_post(self, content: SocialPost, scheduled_time: datetime) -> str:
        raise NotImplementedError("Twitter API v2 doesn't support native scheduling")

    async def delete_post(self, post_id: str) -> bool:
        try:
            url = f"{self.base_url}/tweets/{post_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = await self._client.delete(url, headers=headers)
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.error("twitter.delete_failed", error=str(exc))
            return False

    async def get_posts(self, limit: int = 20, since: datetime | None = None) -> list[dict]:
        try:
            url = f"{self.base_url}/users/me/tweets"
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            params = {"max_results": min(limit, 100)}
            
            response = await self._client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            return response.json().get("data", [])
        except Exception as exc:
            logger.error("twitter.fetch_failed", error=str(exc))
            return []

    async def get_post_metrics(self, post_id: str) -> SocialMetrics:
        try:
            url = f"{self.base_url}/tweets/{post_id}"
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            params = {"tweet.fields": "public_metrics"}
            
            response = await self._client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            metrics = response.json().get("data", {}).get("public_metrics", {})
            
            return SocialMetrics(
                post_id=post_id,
                platform="twitter",
                likes=metrics.get("like_count", 0),
                comments=metrics.get("reply_count", 0),
                shares=metrics.get("retweet_count", 0),
                impressions=metrics.get("impression_count", 0),
            )
        except Exception as exc:
            logger.error("twitter.metrics_failed", error=str(exc))
            return SocialMetrics(post_id=post_id, platform="twitter")

    async def get_comments(self, post_id: str, limit: int = 50) -> list[SocialComment]:
        # Twitter API v2 requires search for replies
        return []

    async def reply_to_comment(self, comment_id: str, message: str) -> bool:
        try:
            url = f"{self.base_url}/tweets"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }
            payload = {"text": message[:280], "reply": {"in_reply_to_tweet_id": comment_id}}
            
            response = await self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.error("twitter.reply_failed", error=str(exc))
            return False

    async def get_account_info(self) -> SocialAccount:
        try:
            url = f"{self.base_url}/users/me"
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            params = {"user.fields": "public_metrics,description,verified"}
            
            response = await self._client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json().get("data", {})
            metrics = data.get("public_metrics", {})
            
            return SocialAccount(
                platform="twitter",
                account_id=data.get("id", ""),
                username=data.get("username", ""),
                display_name=data.get("name", ""),
                followers=metrics.get("followers_count", 0),
                following=metrics.get("following_count", 0),
                bio=data.get("description", ""),
                is_verified=data.get("verified", False),
            )
        except Exception as exc:
            logger.error("twitter.account_info_failed", error=str(exc))
            return SocialAccount(platform="twitter", account_id="", username="", display_name="")

    async def get_audience_insights(self) -> dict[str, Any]:
        return {}  # Requires Premium/Enterprise API

    async def search_mentions(self, query: str | None = None, limit: int = 20) -> list[dict]:
        return []  # Requires search endpoint

    async def close(self) -> None:
        await self._client.aclose()
