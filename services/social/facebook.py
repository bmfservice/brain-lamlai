"""Facebook/Meta integration via Graph API."""

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


class FacebookPlatform(BaseSocialPlatform):
    """Facebook Graph API integration.
    
    Requires:
    - Facebook App ID & Secret
    - Page Access Token (long-lived)
    - Page ID
    
    Permissions needed:
    - pages_manage_posts
    - pages_read_engagement
    - pages_read_user_content
    - pages_manage_metadata
    """

    platform_name = "facebook"
    API_VERSION = "v18.0"

    def __init__(self, credentials: dict[str, Any]):
        """Initialize Facebook platform.
        
        Args:
            credentials: {
                "page_id": "...",
                "access_token": "...",  # Page access token
                "app_id": "..." (optional),
                "app_secret": "..." (optional)
            }
        """
        super().__init__(credentials)
        self.page_id = credentials.get("page_id")
        self.access_token = credentials.get("access_token")
        self.base_url = f"https://graph.facebook.com/{self.API_VERSION}"
        self._client = httpx.AsyncClient(timeout=30.0)

    async def authenticate(self) -> bool:
        """Verify access token and permissions."""
        try:
            url = f"{self.base_url}/me"
            params = {"access_token": self.access_token}
            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            if data.get("id"):
                self.authenticated = True
                logger.info(
                    "facebook.auth_success",
                    page_id=self.page_id,
                    account=data.get("name"),
                )
                return True

            return False
        except Exception as exc:
            logger.error("facebook.auth_failed", error=str(exc))
            return False

    async def post(self, content: SocialPost) -> dict[str, Any]:
        """Publish a post to Facebook page."""
        try:
            url = f"{self.base_url}/{self.page_id}/feed"

            data = {
                "message": content.content,
                "access_token": self.access_token,
            }

            # Add link if provided
            if content.link:
                data["link"] = content.link

            # Add call to action
            if content.call_to_action:
                data["call_to_action"] = {"type": content.call_to_action}

            response = await self._client.post(url, data=data)
            response.raise_for_status()

            result = response.json()
            post_id = result.get("id", "")

            logger.info("facebook.post_published", post_id=post_id)

            return {
                "post_id": post_id,
                "url": f"https://facebook.com/{post_id}",
                "status": "published",
                "platform": "facebook",
            }
        except Exception as exc:
            logger.error("facebook.post_failed", error=str(exc))
            return {"error": str(exc), "status": "failed"}

    async def schedule_post(
        self, content: SocialPost, scheduled_time: datetime
    ) -> str:
        """Schedule post for future (Facebook uses Unix timestamp)."""
        try:
            url = f"{self.base_url}/{self.page_id}/feed"

            unix_time = int(scheduled_time.timestamp())

            data = {
                "message": content.content,
                "published": False,  # Unpublished initially
                "scheduled_publish_time": unix_time,
                "access_token": self.access_token,
            }

            if content.link:
                data["link"] = content.link

            response = await self._client.post(url, data=data)
            response.raise_for_status()

            result = response.json()
            post_id = result.get("id", "")

            logger.info(
                "facebook.post_scheduled",
                post_id=post_id,
                time=scheduled_time.isoformat(),
            )

            return post_id
        except Exception as exc:
            logger.error("facebook.schedule_failed", error=str(exc))
            raise

    async def delete_post(self, post_id: str) -> bool:
        """Delete a Facebook post."""
        try:
            url = f"{self.base_url}/{post_id}"
            params = {"access_token": self.access_token}
            response = await self._client.delete(url, params=params)
            response.raise_for_status()

            logger.info("facebook.post_deleted", post_id=post_id)
            return True
        except Exception as exc:
            logger.error("facebook.delete_failed", post_id=post_id, error=str(exc))
            return False

    async def get_posts(
        self, limit: int = 20, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Fetch recent posts from Facebook page."""
        try:
            url = f"{self.base_url}/{self.page_id}/posts"
            params = {
                "access_token": self.access_token,
                "limit": limit,
                "fields": (
                    "id,message,created_time,permalink_url,"
                    "likes.summary(true),comments.summary(true),shares"
                ),
            }

            if since:
                params["since"] = int(since.timestamp())

            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            posts = data.get("data", [])

            logger.info("facebook.posts_fetched", count=len(posts))

            return [
                {
                    "post_id": p.get("id"),
                    "content": p.get("message", ""),
                    "url": p.get("permalink_url"),
                    "created_at": p.get("created_time"),
                    "likes": p.get("likes", {}).get("summary", {}).get("total_count", 0),
                    "comments": p.get("comments", {}).get("summary", {}).get("total_count", 0),
                    "shares": p.get("shares", {}).get("count", 0),
                }
                for p in posts
            ]
        except Exception as exc:
            logger.error("facebook.fetch_posts_failed", error=str(exc))
            return []

    async def get_post_metrics(self, post_id: str) -> SocialMetrics:
        """Get detailed metrics for a post."""
        try:
            url = f"{self.base_url}/{post_id}"
            params = {
                "access_token": self.access_token,
                "fields": (
                    "id,likes.summary(true),comments.summary(true),"
                    "shares,reactions.summary(true)"
                ),
            }

            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            likes = data.get("likes", {}).get("summary", {}).get("total_count", 0)
            comments = data.get("comments", {}).get("summary", {}).get("total_count", 0)
            shares = data.get("shares", {}).get("count", 0)

            # Fetch insights for reach/impressions (requires different endpoint)
            insights = await self._get_post_insights(post_id)

            reach = insights.get("reach", 0)
            impressions = insights.get("impressions", 0)

            engagement_rate = self._calculate_engagement_rate(
                likes, comments, shares, reach
            )

            return SocialMetrics(
                post_id=post_id,
                platform="facebook",
                likes=likes,
                comments=comments,
                shares=shares,
                reach=reach,
                impressions=impressions,
                engagement_rate=engagement_rate,
            )
        except Exception as exc:
            logger.error("facebook.metrics_failed", post_id=post_id, error=str(exc))
            return SocialMetrics(post_id=post_id, platform="facebook")

    async def _get_post_insights(self, post_id: str) -> dict[str, int]:
        """Get post insights (reach, impressions)."""
        try:
            url = f"{self.base_url}/{post_id}/insights"
            params = {
                "access_token": self.access_token,
                "metric": "post_impressions,post_reach",
            }

            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json().get("data", [])

            insights = {}
            for metric in data:
                name = metric.get("name", "")
                values = metric.get("values", [])
                if values:
                    if name == "post_impressions":
                        insights["impressions"] = values[0].get("value", 0)
                    elif name == "post_reach":
                        insights["reach"] = values[0].get("value", 0)

            return insights
        except Exception as exc:
            logger.warning("facebook.insights_failed", error=str(exc))
            return {"reach": 0, "impressions": 0}

    async def get_comments(
        self, post_id: str, limit: int = 50
    ) -> list[SocialComment]:
        """Get comments on a post."""
        try:
            url = f"{self.base_url}/{post_id}/comments"
            params = {
                "access_token": self.access_token,
                "limit": limit,
                "fields": "id,from,message,created_time,like_count,comment_count",
            }

            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json().get("data", [])

            return [
                SocialComment(
                    comment_id=c.get("id", ""),
                    post_id=post_id,
                    author=c.get("from", {}).get("name", ""),
                    author_id=c.get("from", {}).get("id", ""),
                    content=c.get("message", ""),
                    timestamp=datetime.fromisoformat(
                        c.get("created_time", "").replace("Z", "+00:00")
                    ),
                    likes=c.get("like_count", 0),
                    replies=c.get("comment_count", 0),
                )
                for c in data
            ]
        except Exception as exc:
            logger.error("facebook.comments_failed", error=str(exc))
            return []

    async def reply_to_comment(self, comment_id: str, message: str) -> bool:
        """Reply to a comment."""
        try:
            url = f"{self.base_url}/{comment_id}/comments"
            data = {"message": message, "access_token": self.access_token}

            response = await self._client.post(url, data=data)
            response.raise_for_status()

            logger.info("facebook.reply_posted", comment_id=comment_id)
            return True
        except Exception as exc:
            logger.error("facebook.reply_failed", error=str(exc))
            return False

    async def get_account_info(self) -> SocialAccount:
        """Get Facebook page information."""
        try:
            url = f"{self.base_url}/{self.page_id}"
            params = {
                "access_token": self.access_token,
                "fields": "id,name,username,followers_count,about,link,verification_status",
            }

            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            account = SocialAccount(
                platform="facebook",
                account_id=data.get("id", ""),
                username=data.get("username", ""),
                display_name=data.get("name", ""),
                followers=data.get("followers_count", 0),
                bio=data.get("about", ""),
                profile_url=data.get("link", ""),
                is_verified=data.get("verification_status") == "blue_verified",
                is_business=True,  # Page is always business
            )

            self._cache_account(account)
            return account
        except Exception as exc:
            logger.error("facebook.account_info_failed", error=str(exc))
            return SocialAccount(
                platform="facebook",
                account_id=self.page_id or "",
                username="",
                display_name="",
            )

    async def get_audience_insights(self) -> dict[str, Any]:
        """Get audience demographics."""
        try:
            url = f"{self.base_url}/{self.page_id}/insights"
            params = {
                "access_token": self.access_token,
                "metric": "page_fans_gender_age,page_fans_locale,page_fans_city",
            }

            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json().get("data", [])

            insights = {}
            for metric in data:
                name = metric.get("name", "")
                values = metric.get("values", [])
                if values:
                    insights[name] = values[0].get("value", {})

            return {
                "demographics": insights.get("page_fans_gender_age", {}),
                "locations": insights.get("page_fans_city", {}),
                "locales": insights.get("page_fans_locale", {}),
            }
        except Exception as exc:
            logger.error("facebook.insights_failed", error=str(exc))
            return {}

    async def search_mentions(
        self, query: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search for mentions (requires additional permissions)."""
        # Note: This requires page_read_user_content permission
        # and may not be available for all apps
        logger.warning("facebook.search_mentions_not_implemented")
        return []

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()
