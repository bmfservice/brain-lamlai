"""LinkedIn integration via Marketing API."""

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


class LinkedInPlatform(BaseSocialPlatform):
    """LinkedIn Marketing API integration.
    
    Requires:
    - LinkedIn App Client ID & Secret
    - OAuth 2.0 Access Token
    - Organization/Person URN
    
    Permissions needed:
    - w_member_social (post on behalf)
    - r_basicprofile
    - r_organization_social (if posting as company)
    - w_organization_social
    """

    platform_name = "linkedin"

    def __init__(self, credentials: dict[str, Any]):
        """Initialize LinkedIn platform.
        
        Args:
            credentials: {
                "access_token": "...",
                "person_urn": "urn:li:person:...",  # For personal posts
                "organization_urn": "urn:li:organization:...",  # For company posts
                "account_type": "person"|"organization"
            }
        """
        super().__init__(credentials)
        self.access_token = credentials.get("access_token")
        self.person_urn = credentials.get("person_urn")
        self.organization_urn = credentials.get("organization_urn")
        self.account_type = credentials.get("account_type", "person")
        self.base_url = "https://api.linkedin.com/v2"
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def author_urn(self) -> str:
        """Get appropriate author URN based on account type."""
        if self.account_type == "organization":
            return self.organization_urn or ""
        return self.person_urn or ""

    async def authenticate(self) -> bool:
        """Verify access token."""
        try:
            url = f"{self.base_url}/me"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if data.get("id"):
                self.authenticated = True
                logger.info(
                    "linkedin.auth_success",
                    account_type=self.account_type,
                    account_id=data.get("id"),
                )
                return True

            return False
        except Exception as exc:
            logger.error("linkedin.auth_failed", error=str(exc))
            return False

    async def post(self, content: SocialPost) -> dict[str, Any]:
        """Publish a post to LinkedIn."""
        try:
            url = f"{self.base_url}/ugcPosts"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
            }

            payload = {
                "author": self.author_urn,
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": content.content},
                        "shareMediaCategory": "NONE",
                    }
                },
                "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
            }

            # Add link if provided
            if content.link:
                payload["specificContent"]["com.linkedin.ugc.ShareContent"][
                    "shareMediaCategory"
                ] = "ARTICLE"
                payload["specificContent"]["com.linkedin.ugc.ShareContent"]["media"] = [
                    {
                        "status": "READY",
                        "originalUrl": content.link,
                    }
                ]

            response = await self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            post_id = result.get("id", "")

            logger.info("linkedin.post_published", post_id=post_id)

            return {
                "post_id": post_id,
                "url": f"https://www.linkedin.com/feed/update/{post_id}",
                "status": "published",
                "platform": "linkedin",
            }
        except Exception as exc:
            logger.error("linkedin.post_failed", error=str(exc))
            return {"error": str(exc), "status": "failed"}

    async def schedule_post(
        self, content: SocialPost, scheduled_time: datetime
    ) -> str:
        """Schedule post (LinkedIn doesn't support native scheduling via API).
        
        Note: LinkedIn Marketing API doesn't support scheduled posts directly.
        This would need to be handled by storing the post locally and
        publishing at the scheduled time.
        """
        logger.warning("linkedin.scheduling_not_supported")
        raise NotImplementedError(
            "LinkedIn API doesn't support native post scheduling. "
            "Use a local scheduler to publish at the specified time."
        )

    async def delete_post(self, post_id: str) -> bool:
        """Delete a LinkedIn post."""
        try:
            url = f"{self.base_url}/ugcPosts/{post_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = await self._client.delete(url, headers=headers)
            response.raise_for_status()

            logger.info("linkedin.post_deleted", post_id=post_id)
            return True
        except Exception as exc:
            logger.error("linkedin.delete_failed", post_id=post_id, error=str(exc))
            return False

    async def get_posts(
        self, limit: int = 20, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Fetch recent posts."""
        try:
            url = f"{self.base_url}/ugcPosts"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            params = {
                "q": "authors",
                "authors": self.author_urn,
                "count": limit,
            }

            response = await self._client.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            elements = data.get("elements", [])

            logger.info("linkedin.posts_fetched", count=len(elements))

            posts = []
            for p in elements:
                share_content = (
                    p.get("specificContent", {})
                    .get("com.linkedin.ugc.ShareContent", {})
                    .get("shareCommentary", {})
                )

                posts.append(
                    {
                        "post_id": p.get("id"),
                        "content": share_content.get("text", ""),
                        "created_at": p.get("created", {}).get("time"),
                        "lifecycle_state": p.get("lifecycleState"),
                    }
                )

            return posts
        except Exception as exc:
            logger.error("linkedin.fetch_posts_failed", error=str(exc))
            return []

    async def get_post_metrics(self, post_id: str) -> SocialMetrics:
        """Get metrics for a post (requires Analytics API access)."""
        try:
            # LinkedIn analytics require separate API access
            # This is a simplified version
            url = f"{self.base_url}/socialActions/{post_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"}

            response = await self._client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()

            likes = data.get("likesSummary", {}).get("totalLikes", 0)
            comments = data.get("commentsSummary", {}).get("totalComments", 0)

            return SocialMetrics(
                post_id=post_id,
                platform="linkedin",
                likes=likes,
                comments=comments,
                shares=0,  # Requires separate analytics call
                reach=0,
                impressions=0,
            )
        except Exception as exc:
            logger.error("linkedin.metrics_failed", post_id=post_id, error=str(exc))
            return SocialMetrics(post_id=post_id, platform="linkedin")

    async def get_comments(
        self, post_id: str, limit: int = 50
    ) -> list[SocialComment]:
        """Get comments on a post."""
        try:
            url = f"{self.base_url}/socialActions/{post_id}/comments"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            params = {"count": limit}

            response = await self._client.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json().get("elements", [])

            return [
                SocialComment(
                    comment_id=c.get("$URN", ""),
                    post_id=post_id,
                    author=c.get("actor", ""),
                    author_id=c.get("actor", ""),
                    content=c.get("message", {}).get("text", ""),
                    timestamp=datetime.fromtimestamp(c.get("created", {}).get("time", 0) / 1000),
                    likes=c.get("likesSummary", {}).get("totalLikes", 0),
                )
                for c in data
            ]
        except Exception as exc:
            logger.error("linkedin.comments_failed", error=str(exc))
            return []

    async def reply_to_comment(self, comment_id: str, message: str) -> bool:
        """Reply to a comment (LinkedIn doesn't support comment replies via API)."""
        logger.warning("linkedin.reply_not_supported")
        return False

    async def get_account_info(self) -> SocialAccount:
        """Get LinkedIn account information."""
        try:
            if self.account_type == "organization":
                url = f"{self.base_url}/organizations/{self.organization_urn.split(':')[-1]}"
            else:
                url = f"{self.base_url}/me"

            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = await self._client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()

            if self.account_type == "organization":
                account = SocialAccount(
                    platform="linkedin",
                    account_id=str(data.get("id", "")),
                    username=data.get("vanityName", ""),
                    display_name=data.get("localizedName", ""),
                    followers=data.get("followersCount", 0),
                    bio=data.get("localizedDescription", ""),
                    is_business=True,
                )
            else:
                account = SocialAccount(
                    platform="linkedin",
                    account_id=data.get("id", ""),
                    username="",
                    display_name=(
                        f"{data.get('localizedFirstName', '')} "
                        f"{data.get('localizedLastName', '')}"
                    ),
                    bio="",
                )

            self._cache_account(account)
            return account
        except Exception as exc:
            logger.error("linkedin.account_info_failed", error=str(exc))
            return SocialAccount(
                platform="linkedin",
                account_id="",
                username="",
                display_name="",
            )

    async def get_audience_insights(self) -> dict[str, Any]:
        """Get audience insights (requires Analytics API)."""
        # LinkedIn Analytics API requires separate access
        logger.warning("linkedin.analytics_requires_separate_access")
        return {}

    async def search_mentions(
        self, query: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search mentions (not available via standard API)."""
        logger.warning("linkedin.mentions_not_available")
        return []

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()
