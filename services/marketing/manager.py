"""Marketing Manager - Orchestrates all marketing intelligence and operations."""

from __future__ import annotations

from typing import Any

from core.logging import get_logger
from core.vllm import VLLMClient
from services.ingest.service import IngestService
from services.marketing.competitor import CompetitorAnalyzer
from services.marketing.customer import CustomerTargeting
from services.marketing.market import MarketResearcher
from services.marketing.product import ProductAnalyzer
from services.rag.pipeline import UnifiedRetrievalService
from services.tools.web import WebSearchTool

logger = get_logger(__name__)


class MarketingManager:
    """Unified marketing intelligence and operations manager.
    
    This is the "Marketing Department" coordinator that:
    - Learns about products
    - Researches competitors
    - Analyzes markets
    - Identifies customers
    - Generates marketing strategies
    - Coordinates all marketing activities
    """

    def __init__(
        self,
        vllm: VLLMClient,
        retrieval: UnifiedRetrievalService,
        ingest: IngestService,
        web_tool: WebSearchTool,
    ):
        self.vllm = vllm
        self.retrieval = retrieval
        self.ingest = ingest
        self.web = web_tool

        # Initialize sub-managers
        self.product = ProductAnalyzer(vllm, retrieval, ingest, web_tool)
        self.competitor = CompetitorAnalyzer(vllm, retrieval, ingest, web_tool)
        self.market = MarketResearcher(vllm, retrieval, ingest, web_tool)
        self.customer = CustomerTargeting(vllm, retrieval, ingest, web_tool)

        logger.info("marketing.manager_initialized")

    async def onboard_product(
        self,
        product_name: str,
        documentation_urls: list[str] | None = None,
        documentation_text: str | None = None,
    ) -> dict[str, Any]:
        """Complete product onboarding process.
        
        This performs:
        1. Product learning and analysis
        2. Competitor discovery and research
        3. Market analysis
        4. Customer persona creation
        5. Strategy generation
        
        Args:
            product_name: Product to onboard
            documentation_urls: Product documentation URLs
            documentation_text: Product documentation text
            
        Returns:
            Complete marketing intelligence package
        """
        logger.info("marketing.onboard_start", product=product_name)

        # Step 1: Learn product
        logger.info("marketing.onboard_step", step="product_analysis")
        product_profile = await self.product.learn_product(
            product_name,
            documentation_urls=documentation_urls,
            documentation_text=documentation_text,
            fetch_from_web=True,
        )

        # Step 2: Discover and research competitors
        logger.info("marketing.onboard_step", step="competitor_discovery")
        competitors = await self.competitor.discover_competitors(
            product_name,
            product_profile.category,
            max_competitors=5,
        )

        competitor_profiles = []
        for competitor in competitors[:3]:  # Deep research top 3
            profile = await self.competitor.research_competitor(
                competitor, deep_research=True
            )
            competitor_profiles.append(profile)

        # Step 3: Market analysis
        logger.info("marketing.onboard_step", step="market_analysis")
        market_name = f"{product_profile.category} market"
        market_insights = await self.market.research_market(
            market_name, deep_research=True
        )

        # Step 4: Customer personas
        logger.info("marketing.onboard_step", step="persona_creation")
        personas = await self.customer.create_personas(product_name, num_personas=3)

        # Step 5: Competitive analysis
        logger.info("marketing.onboard_step", step="competitive_analysis")
        swot = await self.competitor.swot_analysis(product_name, competitors)

        # Step 6: Marketing angles
        logger.info("marketing.onboard_step", step="marketing_angles")
        angles = await self.product.generate_marketing_angles(
            product_name, campaign_goal="awareness"
        )

        logger.info("marketing.onboard_complete", product=product_name)

        return {
            "product": {
                "name": product_profile.name,
                "description": product_profile.description,
                "category": product_profile.category,
                "features": product_profile.features,
                "usps": product_profile.unique_selling_points,
                "target_audience": product_profile.target_audience,
                "strengths": product_profile.strengths,
                "weaknesses": product_profile.weaknesses,
            },
            "competitors": [
                {
                    "name": c.name,
                    "website": c.website,
                    "strengths": c.strengths,
                    "weaknesses": c.weaknesses,
                    "threat_level": c.threat_level,
                }
                for c in competitor_profiles
            ],
            "market": {
                "name": market_insights.market_name,
                "size": market_insights.market_size,
                "growth_rate": market_insights.growth_rate,
                "trends": market_insights.trends[:5],
                "opportunities": market_insights.opportunities[:5],
            },
            "personas": [
                {
                    "name": p.name,
                    "segment": p.segment,
                    "goals": p.goals,
                    "pain_points": p.pain_points,
                    "channels": p.preferred_channels,
                }
                for p in personas
            ],
            "swot": swot,
            "marketing_angles": angles[:5],
        }

    async def generate_campaign_plan(
        self,
        product_name: str,
        campaign_goal: str,  # awareness|consideration|conversion
        target_persona_id: str | None = None,
        budget: str | None = None,
        duration: str = "3 months",
    ) -> dict[str, Any]:
        """Generate complete marketing campaign plan.
        
        Args:
            product_name: Product to promote
            campaign_goal: Campaign objective
            target_persona_id: Specific persona to target (or all)
            budget: Campaign budget
            duration: Campaign duration
            
        Returns:
            Comprehensive campaign plan with tactics, content, channels
        """
        logger.info(
            "marketing.campaign_plan_start",
            product=product_name,
            goal=campaign_goal,
        )

        # Get product knowledge
        product_chunks = await self.retrieval.retrieve(
            f"product {product_name} features benefits audience", top_k=10
        )
        product_info = "\n".join([c.content[:500] for c in product_chunks])

        # Get personas
        personas = await self.customer.create_personas(product_name, num_personas=3)

        # Get market insights
        market_chunks = await self.retrieval.retrieve(
            f"market {product_name} trends opportunities", top_k=8
        )
        market_info = "\n".join([c.content[:500] for c in market_chunks])

        # Get marketing angles
        angles = await self.product.generate_marketing_angles(
            product_name, campaign_goal
        )

        # AI-generated campaign plan
        prompt = f"""Generate a comprehensive marketing campaign plan:

Product: {product_name}
{product_info}

Campaign Goal: {campaign_goal}
Budget: {budget or 'Not specified'}
Duration: {duration}

Target Personas:
{chr(10).join([f"- {p.name}: {p.segment}" for p in personas[:3]])}

Market Context:
{market_info}

Marketing Angles:
{chr(10).join([f"- {a.get('angle_name')}: {a.get('key_message')}" for a in angles[:3]])}

Generate detailed campaign plan JSON:
{{
  "campaign_name": "Campaign name",
  "objective": "Primary objective",
  "kpis": [
    {{
      "metric": "Metric name",
      "target": "Target value",
      "tracking": "How to track"
    }},
    ...
  ],
  "strategy": {{
    "positioning": "How we position product",
    "messaging": "Core messaging framework",
    "differentiation": "How we differentiate from competitors"
  }},
  "phases": [
    {{
      "phase": "1",
      "name": "Phase name (e.g., Awareness)",
      "duration": "Weeks 1-4",
      "goals": ["Goal 1", ...],
      "tactics": ["Tactic 1", ...],
      "channels": ["Channel 1", ...],
      "budget_allocation": "XX%"
    }},
    ...
  ],
  "channels": [
    {{
      "channel": "LinkedIn|Facebook|Email|Google Ads|etc.",
      "tactics": ["Tactic 1", ...],
      "content_types": ["Type 1", ...],
      "posting_frequency": "X times/week",
      "budget": "$XXX or XX%"
    }},
    ...
  ],
  "content_calendar": [
    {{
      "week": 1,
      "content_pieces": [
        {{
          "title": "Content title",
          "type": "Blog|Video|Infographic|etc.",
          "channel": "Where to publish",
          "goal": "awareness|consideration|conversion",
          "persona": "Target persona"
        }},
        ...
      ]
    }},
    ...
  ],
  "budget_breakdown": {{
    "paid_ads": "XX%",
    "content_production": "XX%",
    "tools_software": "XX%",
    "other": "XX%"
  }},
  "timeline": [
    {{
      "milestone": "Milestone name",
      "date": "Week X",
      "deliverables": ["Deliverable 1", ...]
    }},
    ...
  ],
  "success_metrics": [
    {{
      "metric": "Metric name",
      "baseline": "Current value",
      "target": "Target value",
      "measurement": "How measured"
    }},
    ...
  ],
  "risks": [
    {{
      "risk": "Risk description",
      "impact": "high|medium|low",
      "mitigation": "Mitigation strategy"
    }},
    ...
  ]
}}

Return ONLY valid JSON."""

        try:
            campaign_plan = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=3000,
            )

            logger.info(
                "marketing.campaign_plan_complete",
                product=product_name,
                phases=len(campaign_plan.get("phases", [])),
            )

            return campaign_plan
        except Exception as exc:
            logger.error("marketing.campaign_plan_failed", error=str(exc))
            return {"error": str(exc)}

    async def competitive_intelligence_report(
        self, our_product: str
    ) -> dict[str, Any]:
        """Generate comprehensive competitive intelligence report.
        
        Args:
            our_product: Our product name
            
        Returns:
            Detailed competitive intelligence report
        """
        logger.info("marketing.intel_report_start", product=our_product)

        # Get product info
        product_knowledge = await self.product.get_product_knowledge(our_product)

        if not product_knowledge.get("found"):
            return {
                "error": f"No knowledge about {our_product}. Run onboard_product() first."
            }

        # Get competitors from product profile
        product_chunks = await self.retrieval.retrieve(
            f"product {our_product} competitors", top_k=5
        )

        # Extract competitor names via AI
        content = "\n".join([c.content for c in product_chunks])
        prompt = f"""Extract competitor names from this content:

{content}

Return JSON: {{"competitors": ["Name1", "Name2", ...]}}"""

        try:
            response = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.1
            )
            competitors = response.get("competitors", [])[:5]
        except Exception:
            competitors = []

        if not competitors:
            return {"error": "No competitors found. More market research needed."}

        # Research each competitor
        competitor_intel = []
        for competitor in competitors:
            profile = await self.competitor.research_competitor(
                competitor, deep_research=False
            )
            competitor_intel.append(
                {
                    "name": profile.name,
                    "strengths": profile.strengths,
                    "weaknesses": profile.weaknesses,
                    "threat_level": profile.threat_level,
                    "recent_updates": profile.recent_updates,
                }
            )

        # SWOT analysis
        swot = await self.competitor.swot_analysis(our_product, competitors)

        # Comparison matrix
        comparison = await self.product.compare_with_competitors(
            our_product, competitors
        )

        logger.info(
            "marketing.intel_report_complete",
            product=our_product,
            competitors=len(competitor_intel),
        )

        return {
            "our_product": our_product,
            "competitors": competitor_intel,
            "swot": swot,
            "comparison": comparison,
            "recommendations": swot.get("strategic_recommendations", []),
        }

    async def market_opportunity_analysis(
        self, our_product: str, market_name: str
    ) -> dict[str, Any]:
        """Identify and prioritize market opportunities.
        
        Args:
            our_product: Our product name
            market_name: Market to analyze
            
        Returns:
            Prioritized opportunities with go-to-market recommendations
        """
        logger.info(
            "marketing.opportunity_analysis_start",
            product=our_product,
            market=market_name,
        )

        # Market research
        market_insights = await self.market.research_market(
            market_name, deep_research=True
        )

        # Identify opportunities
        opportunities = await self.market.identify_opportunities(
            our_product, market_name
        )

        # Prioritize via AI
        opp_text = "\n\n".join(
            [
                f"{i+1}. {o.get('opportunity')}: {o.get('market_gap')}"
                for i, o in enumerate(opportunities[:10])
            ]
        )

        prompt = f"""Prioritize these market opportunities for {our_product}:

{opp_text}

Market Context:
- Size: {market_insights.market_size}
- Growth: {market_insights.growth_rate}
- Trends: {', '.join([t.get('trend', '') for t in market_insights.trends[:3]])}

Prioritize and return JSON:
{{
  "top_priorities": [
    {{
      "rank": 1,
      "opportunity": "Opportunity name",
      "score": 1-10,
      "rationale": "Why this is priority",
      "quick_wins": ["Win 1", ...],
      "resources_needed": ["Resource 1", ...],
      "timeline": "X months",
      "expected_roi": "estimate"
    }},
    ...
  ]
}}

Return ONLY JSON with top 5 priorities."""

        try:
            prioritized = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.4
            )

            return {
                "market": market_name,
                "market_insights": {
                    "size": market_insights.market_size,
                    "growth": market_insights.growth_rate,
                    "trends": market_insights.trends[:5],
                },
                "opportunities": prioritized.get("top_priorities", []),
            }
        except Exception as exc:
            logger.error("marketing.opportunity_prioritization_failed", error=str(exc))
            return {
                "market": market_name,
                "opportunities": opportunities,
            }
