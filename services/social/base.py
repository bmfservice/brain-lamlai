"""Base interface for all social media platforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SocialPost:
    """Social media post data structure."""

    content: str
    media: list[str] = field(default_factory=list)  # URLs or file paths
    scheduled_time: datetime | None = None
    target_audience: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    link: str | None = None
    call_to_action: str | None = None


@dataclass
class SocialMetrics:
    """Social media post metrics."""

    post_id: str
    platform: str
    likes: int = 0
    comments: int = 0
    shares: int = 0
    reach: int = 0
    impressions: int = 0
    engagement_rate: float = 0.0
    clicks: int = 0
    saves: int = 0
    video_views: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SocialAccount:
    """Social media account information."""

    platform: str
    account_id: str
    username: str
    display_name: str
    followers: int = 0
    following: int = 0
    bio: str = ""
    profile_url: str = ""
    is_verified: bool = False
    is_business: bool = False


@dataclass
class SocialComment:
    """Social media comment data."""

    comment_id: str
    post_id: str
    author: str
    author_id: str
    content: str
    timestamp: datetime
    likes: int = 0
    replies: int = 0
    sentiment: str = "neutral"  # positive|neutral|negative


class BaseSocialPlatform(ABC):
    """Abstract base class for all social media platforms.
    
    Each platform (Facebook, LinkedIn, etc.) implements this interface.
    """

    platform_name: str = "unknown"

    def __init__(self, credentials: dict[str, Any]):
        """Initialize platform with credentials.
        
        Args:
            credentials: Platform-specific authentication credentials
        """
        self.credentials = credentials
        self.authenticated = False
        self._account: SocialAccount | None = None

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with platform API.
        
        Returns:
            True if authentication successful
        """
        pass

    @abstractmethod
    async def post(self, content: SocialPost) -> dict[str, Any]:
        """Publish a post to the platform.
        
        Args:
            content: Post content and metadata
            
        Returns:
            {"post_id": "...", "url": "...", "status": "published"}
        """
        pass

    @abstractmethod
    async def schedule_post(
        self, content: SocialPost, scheduled_time: datetime
    ) -> str:
        """Schedule a post for future publication.
        
        Args:
            content: Post content
            scheduled_time: When to publish
            
        Returns:
            Scheduled post ID
        """
        pass

    @abstractmethod
    async def delete_post(self, post_id: str) -> bool:
        """Delete a post.
        
        Args:
            post_id: Post to delete
            
        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    async def get_posts(
        self, limit: int = 20, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Fetch recent posts from account.
        
        Args:
            limit: Maximum posts to fetch
            since: Only posts after this date
            
        Returns:
            List of post objects
        """
        pass

    @abstractmethod
    async def get_post_metrics(self, post_id: str) -> SocialMetrics:
        """Get metrics for a specific post.
        
        Args:
            post_id: Post to analyze
            
        Returns:
            Metrics including likes, comments, reach, etc.
        """
        pass

    @abstractmethod
    async def get_comments(
        self, post_id: str, limit: int = 50
    ) -> list[SocialComment]:
        """Get comments on a post.
        
        Args:
            post_id: Post to fetch comments from
            limit: Maximum comments
            
        Returns:
            List of comments
        """
        pass

    @abstractmethod
    async def reply_to_comment(
        self, comment_id: str, message: str
    ) -> bool:
        """Reply to a comment.
        
        Args:
            comment_id: Comment to reply to
            message: Reply message
            
        Returns:
            True if reply posted
        """
        pass

    @abstractmethod
    async def get_account_info(self) -> SocialAccount:
        """Get current account information.
        
        Returns:
            Account details
        """
        pass

    @abstractmethod
    async def get_audience_insights(self) -> dict[str, Any]:
        """Get audience demographics and insights.
        
        Returns:
            {
                "demographics": {...},
                "interests": [...],
                "locations": [...],
                "active_times": {...}
            }
        """
        pass

    @abstractmethod
    async def search_mentions(
        self, query: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search for mentions of account or keywords.
        
        Args:
            query: Search keywords (None = all mentions)
            limit: Maximum results
            
        Returns:
            List of mentions
        """
        pass

    # Helper methods (implemented in base class)

    async def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return self.authenticated

    def get_platform_name(self) -> str:
        """Get platform name."""
        return self.platform_name

    async def get_cached_account(self) -> SocialAccount | None:
        """Get cached account info without API call."""
        return self._account

    def _cache_account(self, account: SocialAccount) -> None:
        """Cache account information."""
        self._account = account

    @staticmethod
    def _calculate_engagement_rate(
        likes: int, comments: int, shares: int, reach: int
    ) -> float:
        """Calculate engagement rate percentage."""
        if reach == 0:
            return 0.0
        total_engagement = likes + comments + shares
        return round((total_engagement / reach) * 100, 2)
