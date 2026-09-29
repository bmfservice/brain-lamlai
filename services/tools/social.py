"""Braind Tools - Social Media Operations."""

from __future__ import annotations

from typing import Any

from core.agent.tool import BaseTool
from core.logging import get_logger
from services.social.manager import SocialMediaManager

logger = get_logger(__name__)


class SocialPostTool(BaseTool):
    """Post content to social media platforms."""

    name = "social_post"
    description = (
        "Đăng nội dung lên mạng xã hội (Facebook, LinkedIn, Twitter/X). "
        "Có thể đăng 1 hoặc nhiều nền tảng cùng lúc. "
        "AI tự động điều chỉnh nội dung cho từng platform."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Nội dung cần đăng",
            },
            "platforms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Danh sách platforms: ['facebook', 'linkedin', 'twitter']",
            },
            "adapt_content": {
                "type": "boolean",
                "description": "True = AI tự động điều chỉnh cho từng platform (recommended)",
            },
            "link": {
                "type": "string",
                "description": "Link đính kèm (optional)",
            },
        },
        "required": ["content", "platforms"],
    }

    def __init__(self, social: SocialMediaManager):
        self.social = social

    async def execute(
        self,
        content: str,
        platforms: list[str],
        adapt_content: bool = True,
        link: str | None = None,
    ) -> Any:
        results = await self.social.cross_post(
            content, platforms, adapt_per_platform=adapt_content, link=link
        )

        # Format response
        success_count = len([r for r in results.values() if r.get("status") == "published"])
        failed_count = len(results) - success_count

        return {
            "status": "completed",
            "posted": success_count,
            "failed": failed_count,
            "platforms": results,
            "message": f"Đã đăng thành công lên {success_count}/{len(platforms)} platforms",
        }


class SocialAnalyticsTool(BaseTool):
    """Get social media analytics."""

    name = "social_analytics"
    description = (
        "Xem analytics từ social media: likes, comments, shares, reach, engagement rate. "
        "Tổng hợp từ tất cả platforms đã kết nối."
    )
    parameters = {
        "type": "object",
        "properties": {
            "time_period": {
                "type": "string",
                "enum": ["24h", "7d", "30d"],
                "description": "Khoảng thời gian phân tích (default: 7d)",
            },
        },
    }

    def __init__(self, social: SocialMediaManager):
        self.social = social

    async def execute(self, time_period: str = "7d") -> Any:
        analytics = await self.social.get_unified_analytics(time_period)

        # Format summary
        platforms = analytics.get("platforms", {})
        summary = {
            "period": time_period,
            "total_engagement": analytics.get("total_engagement", 0),
            "platforms_active": analytics.get("platform_count", 0),
            "breakdown": {},
        }

        for platform_name, metrics in platforms.items():
            if "error" in metrics:
                summary["breakdown"][platform_name] = {"status": "error"}
            else:
                summary["breakdown"][platform_name] = {
                    "posts": metrics.get("posts_count", 0),
                    "likes": metrics.get("total_likes", 0),
                    "comments": metrics.get("total_comments", 0),
                    "shares": metrics.get("total_shares", 0),
                    "reach": metrics.get("total_reach", 0),
                    "engagement_rate": f"{metrics.get('avg_engagement_rate', 0)}%",
                }

        return summary


class SocialMonitorTool(BaseTool):
    """Monitor social media engagement."""

    name = "social_monitor"
    description = (
        "Theo dõi comments và mentions trên social media. "
        "Phát hiện comments cần trả lời, mentions của brand, engagement opportunities."
    )
    parameters = {
        "type": "object",
        "properties": {
            "platforms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Platforms cần monitor (bỏ trống = tất cả)",
            },
            "auto_respond": {
                "type": "boolean",
                "description": "True = AI tự động trả lời comments (cẩn thận!)",
            },
        },
    }

    def __init__(self, social: SocialMediaManager):
        self.social = social

    async def execute(
        self, platforms: list[str] | None = None, auto_respond: bool = False
    ) -> Any:
        engagements = await self.social.monitor_engagement(platforms)

        # Count total pending
        total_comments = sum(
            e.get("total_comments", 0)
            for e in engagements.values()
            if "error" not in e
        )

        # Auto-respond if requested
        responses_posted = 0
        if auto_respond:
            for platform_name, data in engagements.items():
                if "error" in data:
                    continue
                comments = data.get("recent_comments", [])
                if comments:
                    responses = await self.social.respond_to_comments(
                        platform_name, comments, auto_respond=True
                    )
                    responses_posted += len(
                        [r for r in responses if r.get("status") == "posted"]
                    )

        return {
            "total_comments": total_comments,
            "platforms": engagements,
            "auto_responded": responses_posted if auto_respond else 0,
            "message": f"Tìm thấy {total_comments} comments. "
            + (f"Đã tự động trả lời {responses_posted}." if auto_respond else ""),
        }


class SocialTimingTool(BaseTool):
    """Analyze optimal posting times."""

    name = "social_timing"
    description = (
        "Phân tích thời gian đăng bài tối ưu cho từng platform dựa trên engagement data. "
        "AI phân tích patterns và suggest best times to post."
    )
    parameters = {"type": "object", "properties": {}}

    def __init__(self, social: SocialMediaManager):
        self.social = social

    async def execute(self) -> Any:
        timing = await self.social.find_optimal_posting_times()

        if "error" in timing:
            return timing

        # Format recommendations
        recommendations = {}
        for platform, data in timing.items():
            if isinstance(data, dict) and "best_times" in data:
                recommendations[platform] = {
                    "best_days": data.get("best_days", []),
                    "best_times": data.get("best_times", []),
                    "reasoning": data.get("reasoning", ""),
                }

        return {
            "recommendations": recommendations,
            "message": "Phân tích hoàn tất. Đây là thời gian tối ưu để đăng bài cho từng platform.",
        }
