"""Social Media Manager - Multi-platform orchestration."""

from __future__ import annotations

from typing import Any

from core.logging import get_logger
from core.vllm import VLLMClient
from services.social.base import BaseSocialPlatform, SocialPost

logger = get_logger(__name__)


class SocialMediaManager:
    """Orchestrates multi-platform social media operations.
    
    Capabilities:
    - Cross-platform posting
    - Content adaptation per platform
    - Unified analytics
    - Engagement monitoring
    - Optimal timing analysis
    """

    def __init__(
        self,
        platforms: dict[str, BaseSocialPlatform],
        vllm: VLLMClient | None = None,
    ):
        """Initialize manager with platforms.
        
        Args:
            platforms: {"facebook": fb_platform, "linkedin": li_platform, ...}
            vllm: AI client for content adaptation
        """
        self.platforms = platforms
        self.vllm = vllm
        logger.info(
            "social.manager_initialized",
            platforms=list(platforms.keys()),
        )

    async def cross_post(
        self,
        content: str | SocialPost,
        platforms: list[str],
        *,
        adapt_per_platform: bool = True,
        link: str | None = None,
    ) -> dict[str, Any]:
        """Post same content to multiple platforms.
        
        Args:
            content: Base content (string or SocialPost)
            platforms: ["facebook", "linkedin", "twitter"]
            adapt_per_platform: If True, AI adapts content per platform
            link: Optional link to include
            
        Returns:
            {"facebook": {"post_id": "...", "status": "..."}, ...}
        """
        logger.info(
            "social.cross_post_start",
            platforms=platforms,
            adapt=adapt_per_platform,
        )

        # Convert string to SocialPost if needed
        base_post = SocialPost(content=content, link=link) if isinstance(content, str) else content

        # Adapt content per platform if requested
        if adapt_per_platform and self.vllm:
            adapted = await self._adapt_content_per_platform(
                base_post.content, platforms
            )
        else:
            adapted = {p: base_post.content for p in platforms}

        # Post to each platform
        results = {}
        for platform_name in platforms:
            platform = self.platforms.get(platform_name)
            if not platform:
                results[platform_name] = {"error": "Platform not configured"}
                continue

            try:
                adapted_post = SocialPost(
                    content=adapted[platform_name],
                    link=base_post.link,
                    hashtags=base_post.hashtags,
                )
                result = await platform.post(adapted_post)
                results[platform_name] = result
            except Exception as exc:
                logger.error(
                    "social.post_failed",
                    platform=platform_name,
                    error=str(exc),
                )
                results[platform_name] = {"error": str(exc), "status": "failed"}

        logger.info("social.cross_post_complete", results=results)
        return results

    async def _adapt_content_per_platform(
        self, base_content: str, platforms: list[str]
    ) -> dict[str, str]:
        """Use AI to adapt content for each platform's best practices."""
        if not self.vllm:
            return {p: base_content for p in platforms}

        # Platform-specific guidelines
        specs = {
            "facebook": {
                "max_length": 5000,
                "tone": "conversational",
                "emojis": "moderate",
                "style": "storytelling, questions, engagement",
            },
            "linkedin": {
                "max_length": 3000,
                "tone": "professional",
                "emojis": "minimal",
                "style": "insights, industry trends, expertise",
            },
            "twitter": {
                "max_length": 280,
                "tone": "concise",
                "emojis": "optional",
                "style": "hook first, thread if needed",
            },
            "tiktok": {
                "max_length": 2200,
                "tone": "casual",
                "emojis": "heavy",
                "style": "trending, hashtags, call-to-action",
            },
            "instagram": {
                "max_length": 2200,
                "tone": "visual-first",
                "emojis": "heavy",
                "style": "lifestyle, aesthetics, hashtags",
            },
        }

        adapted = {}
        for platform_name in platforms:
            spec = specs.get(platform_name, {})
            if not spec:
                adapted[platform_name] = base_content
                continue

            prompt = f"""Adapt this content for {platform_name.upper()}:

Original: {base_content}

Platform Guidelines:
- Max length: {spec.get('max_length', 1000)} characters
- Tone: {spec.get('tone', 'neutral')}
- Emoji usage: {spec.get('emojis', 'moderate')}
- Style: {spec.get('style', 'engaging')}

Adapt while preserving core message. Return ONLY the adapted text, no extra explanation."""

            try:
                adapted_text = await self.vllm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500,
                )
                adapted[platform_name] = adapted_text.strip()
            except Exception as exc:
                logger.error(
                    "social.adaptation_failed",
                    platform=platform_name,
                    error=str(exc),
                )
                adapted[platform_name] = base_content[:spec.get("max_length", 1000)]

        return adapted

    async def get_unified_analytics(
        self, time_period: str = "7d"
    ) -> dict[str, Any]:
        """Aggregate analytics across all platforms.
        
        Args:
            time_period: "24h"|"7d"|"30d"
            
        Returns:
            Unified metrics across platforms
        """
        logger.info("social.analytics_start", period=time_period)

        all_metrics = {}
        for platform_name, platform in self.platforms.items():
            try:
                # Get recent posts
                posts = await platform.get_posts(limit=20)

                # Get metrics for each post
                post_metrics = []
                for post in posts[:10]:  # Top 10 recent
                    post_id = post.get("post_id") or post.get("id")
                    if post_id:
                        try:
                            metrics = await platform.get_post_metrics(post_id)
                            post_metrics.append(metrics)
                        except Exception:
                            pass

                # Aggregate
                total_likes = sum(m.likes for m in post_metrics)
                total_comments = sum(m.comments for m in post_metrics)
                total_shares = sum(m.shares for m in post_metrics)
                total_reach = sum(m.reach for m in post_metrics)
                avg_engagement = (
                    sum(m.engagement_rate for m in post_metrics) / len(post_metrics)
                    if post_metrics
                    else 0
                )

                all_metrics[platform_name] = {
                    "posts_count": len(posts),
                    "total_likes": total_likes,
                    "total_comments": total_comments,
                    "total_shares": total_shares,
                    "total_reach": total_reach,
                    "avg_engagement_rate": round(avg_engagement, 2),
                    "posts_analyzed": len(post_metrics),
                }
            except Exception as exc:
                logger.error(
                    "social.analytics_failed",
                    platform=platform_name,
                    error=str(exc),
                )
                all_metrics[platform_name] = {"error": str(exc)}

        # Calculate totals
        total_engagement = sum(
            m.get("total_likes", 0) + m.get("total_comments", 0) + m.get("total_shares", 0)
            for m in all_metrics.values()
            if "error" not in m
        )

        return {
            "period": time_period,
            "platforms": all_metrics,
            "total_engagement": total_engagement,
            "platform_count": len([m for m in all_metrics.values() if "error" not in m]),
        }

    async def monitor_engagement(
        self, platforms: list[str] | None = None
    ) -> dict[str, Any]:
        """Monitor recent comments/mentions requiring response.
        
        Args:
            platforms: Specific platforms to monitor (None = all)
            
        Returns:
            Pending engagements by platform
        """
        target_platforms = platforms or list(self.platforms.keys())

        engagements = {}
        for platform_name in target_platforms:
            platform = self.platforms.get(platform_name)
            if not platform:
                continue

            try:
                # Get recent posts
                posts = await platform.get_posts(limit=10)

                # Get comments on each post
                all_comments = []
                for post in posts[:5]:  # Top 5 posts
                    post_id = post.get("post_id") or post.get("id")
                    if post_id:
                        comments = await platform.get_comments(post_id, limit=20)
                        all_comments.extend(comments)

                # Filter unanswered (simplified - in production, track replies)
                engagements[platform_name] = {
                    "total_comments": len(all_comments),
                    "recent_comments": [
                        {
                            "comment_id": c.comment_id,
                            "author": c.author,
                            "content": c.content[:100],
                            "timestamp": c.timestamp.isoformat(),
                            "likes": c.likes,
                        }
                        for c in all_comments[:10]
                    ],
                }
            except Exception as exc:
                logger.error(
                    "social.monitoring_failed",
                    platform=platform_name,
                    error=str(exc),
                )
                engagements[platform_name] = {"error": str(exc)}

        return engagements

    async def find_optimal_posting_times(self) -> dict[str, Any]:
        """Analyze engagement data to find best posting times per platform.
        
        Uses AI to analyze historical performance and recommend timing.
        """
        if not self.vllm:
            return {"error": "AI client not available"}

        analytics = await self.get_unified_analytics(time_period="30d")

        prompt = f"""Analyze this social media data and recommend optimal posting times:

{analytics}

For each platform, identify:
1. Best days of week (Mon-Sun)
2. Best times of day (morning/afternoon/evening with specific hours)
3. Reasoning based on engagement patterns

Return JSON:
{{
  "platform_name": {{
    "best_days": ["Monday", "Wednesday", "Friday"],
    "best_times": ["9:00-10:00", "14:00-15:00", "18:00-19:00"],
    "timezone": "Local business hours",
    "reasoning": "Why these times work",
    "avoid_times": ["Times to avoid"]
  }},
  ...
}}

Return ONLY JSON."""

        try:
            result = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return result
        except Exception as exc:
            logger.error("social.timing_analysis_failed", error=str(exc))
            return {"error": str(exc)}

    async def respond_to_comments(
        self,
        platform_name: str,
        comments: list[dict[str, Any]],
        *,
        auto_respond: bool = False,
    ) -> list[dict[str, Any]]:
        """Generate or post responses to comments.
        
        Args:
            platform_name: Platform to respond on
            comments: List of comments to respond to
            auto_respond: If True, automatically post responses
            
        Returns:
            List of generated/posted responses
        """
        if not self.vllm:
            return []

        platform = self.platforms.get(platform_name)
        if not platform:
            return []

        responses = []
        for comment in comments:
            comment_text = comment.get("content", "")
            comment_id = comment.get("comment_id", "")

            if not comment_text or not comment_id:
                continue

            # Generate response via AI
            prompt = f"""Generate a professional, helpful response to this social media comment:

Comment: "{comment_text}"

Platform: {platform_name}

Guidelines:
- Be friendly and professional
- Keep it concise (1-2 sentences)
- Thank them if positive, address concern if negative
- Add value, don't just acknowledge

Return only the response text, no quotes or extra formatting."""

            try:
                response_text = await self.vllm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150,
                )

                response_obj = {
                    "comment_id": comment_id,
                    "response": response_text.strip(),
                    "status": "generated",
                }

                # Auto-post if enabled
                if auto_respond:
                    success = await platform.reply_to_comment(
                        comment_id, response_text.strip()
                    )
                    response_obj["status"] = "posted" if success else "failed"

                responses.append(response_obj)
            except Exception as exc:
                logger.error(
                    "social.response_failed",
                    comment_id=comment_id,
                    error=str(exc),
                )

        return responses

    async def close_all(self) -> None:
        """Close all platform clients."""
        for platform in self.platforms.values():
            if hasattr(platform, "close"):
                await platform.close()
