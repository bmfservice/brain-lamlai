"""Braind Tools - Marketing intelligence and operations."""

from __future__ import annotations

from typing import Any

from core.agent.tool import BaseTool
from core.logging import get_logger
from services.marketing.manager import MarketingManager

logger = get_logger(__name__)


class ProductAnalysisTool(BaseTool):
    """Learn and analyze products."""

    name = "product_analysis"
    description = (
        "Học và phân tích sản phẩm: features, benefits, target audience, USPs, positioning. "
        "Dùng khi cần hiểu sâu về một sản phẩm (của mình hoặc đối thủ)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "Tên sản phẩm cần phân tích",
            },
            "documentation_url": {
                "type": "string",
                "description": "URL tài liệu sản phẩm (optional)",
            },
        },
        "required": ["product_name"],
    }

    def __init__(self, marketing: MarketingManager):
        self.marketing = marketing

    async def execute(
        self, product_name: str, documentation_url: str | None = None
    ) -> Any:
        urls = [documentation_url] if documentation_url else None
        profile = await self.marketing.product.learn_product(
            product_name, documentation_urls=urls, fetch_from_web=True
        )

        return {
            "product": profile.name,
            "category": profile.category,
            "description": profile.description,
            "features": profile.features[:10],
            "benefits": profile.benefits[:8],
            "usps": profile.unique_selling_points,
            "target_audience": profile.target_audience,
            "strengths": profile.strengths,
            "weaknesses": profile.weaknesses,
            "confidence": profile.confidence_score,
        }


class CompetitorResearchTool(BaseTool):
    """Research competitors."""

    name = "competitor_research"
    description = (
        "Nghiên cứu đối thủ cạnh tranh: features, pricing, strengths, weaknesses, threat level. "
        "Dùng khi cần intelligence về competitors."
    )
    parameters = {
        "type": "object",
        "properties": {
            "competitor_name": {
                "type": "string",
                "description": "Tên đối thủ cần nghiên cứu",
            },
            "deep_research": {
                "type": "boolean",
                "description": "True = nghiên cứu sâu (chậm hơn), False = nhanh (default True)",
            },
        },
        "required": ["competitor_name"],
    }

    def __init__(self, marketing: MarketingManager):
        self.marketing = marketing

    async def execute(
        self, competitor_name: str, deep_research: bool = True
    ) -> Any:
        profile = await self.marketing.competitor.research_competitor(
            competitor_name, deep_research=deep_research
        )

        return {
            "competitor": profile.name,
            "website": profile.website,
            "description": profile.description,
            "features": profile.features[:10],
            "strengths": profile.strengths,
            "weaknesses": profile.weaknesses,
            "threat_level": profile.threat_level,
            "recent_updates": profile.recent_updates[:5],
            "sentiment": profile.customer_reviews_sentiment,
        }


class MarketResearchTool(BaseTool):
    """Research markets."""

    name = "market_research"
    description = (
        "Nghiên cứu thị trường: market size, growth rate, trends, opportunities, key players. "
        "Dùng khi cần hiểu về một thị trường."
    )
    parameters = {
        "type": "object",
        "properties": {
            "market_name": {
                "type": "string",
                "description": "Tên thị trường (e.g., 'CRM software market')",
            },
        },
        "required": ["market_name"],
    }

    def __init__(self, marketing: MarketingManager):
        self.marketing = marketing

    async def execute(self, market_name: str) -> Any:
        insights = await self.marketing.market.research_market(
            market_name, deep_research=True
        )

        return {
            "market": insights.market_name,
            "size": insights.market_size,
            "growth_rate": insights.growth_rate,
            "trends": insights.trends[:5],
            "key_players": insights.key_players,
            "opportunities": insights.opportunities[:5],
            "threats": insights.threats,
            "confidence": insights.confidence_score,
        }


class CustomerPersonaTool(BaseTool):
    """Create customer personas."""

    name = "customer_personas"
    description = (
        "Tạo customer personas chi tiết: demographics, pain points, goals, buying triggers. "
        "Dùng khi cần hiểu target audience."
    )
    parameters = {
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "Sản phẩm cần tạo personas",
            },
            "num_personas": {
                "type": "integer",
                "description": "Số lượng personas (default 3)",
            },
        },
        "required": ["product_name"],
    }

    def __init__(self, marketing: MarketingManager):
        self.marketing = marketing

    async def execute(self, product_name: str, num_personas: int = 3) -> Any:
        personas = await self.marketing.customer.create_personas(
            product_name, num_personas=num_personas
        )

        return {
            "product": product_name,
            "personas": [
                {
                    "name": p.name,
                    "segment": p.segment,
                    "demographics": p.demographics,
                    "goals": p.goals,
                    "pain_points": p.pain_points,
                    "preferred_channels": p.preferred_channels,
                    "budget": p.budget_range,
                }
                for p in personas
            ],
        }


class MarketingCampaignTool(BaseTool):
    """Generate marketing campaign plans."""

    name = "marketing_campaign"
    description = (
        "Tạo chiến dịch marketing hoàn chỉnh: strategy, tactics, content calendar, budget. "
        "Dùng khi cần plan một campaign."
    )
    parameters = {
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "Sản phẩm cần chạy campaign",
            },
            "campaign_goal": {
                "type": "string",
                "enum": ["awareness", "consideration", "conversion", "retention"],
                "description": "Mục tiêu campaign",
            },
            "budget": {
                "type": "string",
                "description": "Ngân sách (e.g., '$10,000' or '10M VND')",
            },
            "duration": {
                "type": "string",
                "description": "Thời gian (e.g., '3 months', '6 weeks')",
            },
        },
        "required": ["product_name", "campaign_goal"],
    }

    def __init__(self, marketing: MarketingManager):
        self.marketing = marketing

    async def execute(
        self,
        product_name: str,
        campaign_goal: str,
        budget: str | None = None,
        duration: str = "3 months",
    ) -> Any:
        plan = await self.marketing.generate_campaign_plan(
            product_name,
            campaign_goal,
            budget=budget,
            duration=duration,
        )

        return plan


class CompetitiveIntelTool(BaseTool):
    """Generate competitive intelligence report."""

    name = "competitive_intelligence"
    description = (
        "Tạo báo cáo phân tích đối thủ cạnh tranh: SWOT, comparison matrix, recommendations. "
        "Dùng khi cần tổng hợp intelligence về competitors."
    )
    parameters = {
        "type": "object",
        "properties": {
            "our_product": {
                "type": "string",
                "description": "Sản phẩm của chúng ta",
            },
        },
        "required": ["our_product"],
    }

    def __init__(self, marketing: MarketingManager):
        self.marketing = marketing

    async def execute(self, our_product: str) -> Any:
        report = await self.marketing.competitive_intelligence_report(our_product)
        return report


class ProductOnboardingTool(BaseTool):
    """Complete product onboarding with full marketing intelligence."""

    name = "product_onboarding"
    description = (
        "Onboarding hoàn chỉnh cho sản phẩm: học product, tìm competitors, "
        "phân tích market, tạo personas, SWOT. "
        "Dùng khi cần setup marketing cho sản phẩm mới. "
        "Đây là workflow tổng hợp, chạy lâu (~5-10 phút)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "Tên sản phẩm",
            },
            "documentation_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URLs tài liệu sản phẩm (optional)",
            },
            "documentation_text": {
                "type": "string",
                "description": "Text tài liệu sản phẩm (optional)",
            },
        },
        "required": ["product_name"],
    }

    def __init__(self, marketing: MarketingManager):
        self.marketing = marketing

    async def execute(
        self,
        product_name: str,
        documentation_urls: list[str] | None = None,
        documentation_text: str | None = None,
    ) -> Any:
        result = await self.marketing.onboard_product(
            product_name,
            documentation_urls=documentation_urls,
            documentation_text=documentation_text,
        )

        return {
            "product": result["product"],
            "competitors": result["competitors"],
            "market": result["market"],
            "personas": result["personas"][:2],  # Show top 2
            "swot": {
                "strengths": result["swot"].get("strengths", [])[:3],
                "weaknesses": result["swot"].get("weaknesses", [])[:3],
                "opportunities": result["swot"].get("opportunities", [])[:3],
                "threats": result["swot"].get("threats", [])[:3],
            },
            "marketing_angles": result["marketing_angles"][:3],
            "message": (
                f"Onboarding complete. {len(result['competitors'])} competitors, "
                f"{len(result['personas'])} personas, "
                f"{len(result['marketing_angles'])} angles generated."
            ),
        }
