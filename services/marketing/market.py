"""Market Researcher - Analyze market trends, size, and opportunities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.logging import get_logger
from core.vllm import VLLMClient
from services.ingest.service import IngestService
from services.rag.pipeline import UnifiedRetrievalService
from services.tools.web import WebSearchTool

logger = get_logger(__name__)


@dataclass
class MarketInsights:
    """Market research insights."""

    market_name: str
    market_size: str  # estimated
    growth_rate: str  # YoY%
    trends: list[dict[str, Any]]
    key_players: list[str]
    customer_segments: list[dict[str, Any]]
    pain_points: list[str]
    opportunities: list[dict[str, Any]]
    threats: list[str]
    regulations: list[str]
    technology_trends: list[str]
    confidence_score: float


class MarketResearcher:
    """Research and analyze markets, trends, and opportunities.
    
    Capabilities:
    - Market size and growth analysis
    - Trend identification
    - Customer segment analysis
    - Opportunity assessment
    - Regulatory landscape
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

    async def research_market(
        self, market_name: str, deep_research: bool = True
    ) -> MarketInsights:
        """Comprehensive market research.
        
        Args:
            market_name: Market to research (e.g., "CRM software market")
            deep_research: If True, performs extensive web research
            
        Returns:
            Complete market insights
        """
        logger.info("market.research_start", market=market_name)

        if deep_research:
            await self._gather_market_intelligence(market_name)

        # Retrieve intelligence
        chunks = await self.retrieval.retrieve(
            f"market {market_name} trends size growth opportunities",
            top_k=20,
        )

        # AI analysis
        insights = await self._analyze_market(market_name, chunks)

        logger.info(
            "market.research_complete",
            market=market_name,
            trends=len(insights.trends),
        )

        return insights

    async def _gather_market_intelligence(self, market_name: str) -> None:
        """Gather market intelligence from web."""
        queries = [
            f"{market_name} size and growth 2026",
            f"{market_name} trends and predictions",
            f"{market_name} key players and market share",
            f"{market_name} customer segments",
            f"{market_name} challenges and opportunities",
            f"{market_name} statistics and data",
        ]

        for query in queries:
            try:
                results = await self.web.search(query, max_results=5)

                for result in results[:3]:
                    try:
                        page = await self.web.fetch_page_text(
                            result["url"], max_chars=8000
                        )

                        await self.ingest.ingest_text(
                            f"{page['title']}\n\n{page['text']}",
                            title=f"Market Research: {page['title']}",
                            source_type="market_research",
                            url=result["url"],
                            tags=[
                                "market",
                                market_name.lower().replace(" ", "_"),
                                "research",
                            ],
                            meta={
                                "market": market_name,
                                "source": "web_search",
                                "query": query,
                            },
                        )
                    except Exception as exc:
                        logger.warning(
                            "market.page_fetch_failed",
                            url=result.get("url"),
                            error=str(exc),
                        )
            except Exception as exc:
                logger.warning(
                    "market.search_failed", query=query, error=str(exc)
                )

    async def _analyze_market(
        self, market_name: str, chunks: list[Any]
    ) -> MarketInsights:
        """AI-powered market analysis."""
        content = "\n\n".join(
            [f"Source: {c.title}\n{c.content[:800]}" for c in chunks[:15]]
        )

        prompt = f"""Analyze this market and generate comprehensive insights:

Market: {market_name}

Research Data:
{content}

Generate detailed JSON analysis:
{{
  "market_name": "{market_name}",
  "market_size": "Estimated market size (e.g., $XXB, XXM users)",
  "growth_rate": "YoY growth rate (e.g., 15% CAGR)",
  "trends": [
    {{
      "trend": "Trend name",
      "description": "Description",
      "impact": "high|medium|low",
      "timeframe": "current|emerging|future"
    }},
    ...
  ],
  "key_players": ["Player 1", "Player 2", ...],
  "customer_segments": [
    {{
      "segment": "Segment name",
      "size": "Estimated size",
      "characteristics": "Key characteristics",
      "needs": ["Need 1", "Need 2"]
    }},
    ...
  ],
  "pain_points": ["Pain point 1", "Pain point 2", ...],
  "opportunities": [
    {{
      "opportunity": "Opportunity name",
      "description": "Description",
      "potential": "high|medium|low",
      "barriers": ["Barrier 1", ...]
    }},
    ...
  ],
  "threats": ["Threat 1", "Threat 2", ...],
  "regulations": ["Regulation 1", "Regulation 2", ...],
  "technology_trends": ["Tech trend 1", "Tech trend 2", ...],
  "confidence_score": 0.0-1.0
}}

Extract from available data. Use "Unknown" if missing.
Return ONLY valid JSON."""

        try:
            response = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.3
            )

            trends = []
            for t in response.get("trends", []):
                if isinstance(t, dict):
                    trends.append(t)

            segments = []
            for s in response.get("customer_segments", []):
                if isinstance(s, dict):
                    segments.append(s)

            opportunities = []
            for o in response.get("opportunities", []):
                if isinstance(o, dict):
                    opportunities.append(o)

            return MarketInsights(
                market_name=response.get("market_name", market_name),
                market_size=response.get("market_size", "Unknown"),
                growth_rate=response.get("growth_rate", "Unknown"),
                trends=trends,
                key_players=response.get("key_players", []),
                customer_segments=segments,
                pain_points=response.get("pain_points", []),
                opportunities=opportunities,
                threats=response.get("threats", []),
                regulations=response.get("regulations", []),
                technology_trends=response.get("technology_trends", []),
                confidence_score=float(response.get("confidence_score", 0.6)),
            )
        except Exception as exc:
            logger.error("market.analysis_failed", error=str(exc))
            return MarketInsights(
                market_name=market_name,
                market_size="Unknown",
                growth_rate="Unknown",
                trends=[],
                key_players=[],
                customer_segments=[],
                pain_points=[],
                opportunities=[],
                threats=[],
                regulations=[],
                technology_trends=[],
                confidence_score=0.0,
            )

    async def identify_opportunities(
        self, our_product: str, market_name: str
    ) -> list[dict[str, Any]]:
        """Identify market opportunities for our product."""
        # Get market insights
        market_chunks = await self.retrieval.retrieve(
            f"market {market_name} opportunities gaps unmet needs",
            top_k=10,
        )

        # Get product capabilities
        product_chunks = await self.retrieval.retrieve(
            f"product {our_product} features capabilities strengths",
            top_k=8,
        )

        market_data = "\n".join([c.content[:500] for c in market_chunks])
        product_data = "\n".join([c.content[:500] for c in product_chunks])

        prompt = f"""Identify market opportunities for our product:

Our Product: {our_product}
Product Capabilities:
{product_data}

Market: {market_name}
Market Insights:
{market_data}

Identify opportunities where our product can address market gaps:
{{
  "opportunities": [
    {{
      "opportunity": "Opportunity name",
      "market_gap": "What gap this fills",
      "our_fit": "How our product addresses it",
      "target_segment": "Who to target",
      "estimated_value": "high|medium|low",
      "competition_level": "low|medium|high",
      "go_to_market_strategy": "Recommended approach",
      "success_metrics": ["Metric 1", ...],
      "timeline": "immediate|3-6 months|6-12 months",
      "resources_needed": ["Resource 1", ...]
    }},
    ...
  ]
}}

Return ONLY JSON with 5-10 opportunities."""

        try:
            result = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.4
            )
            return result.get("opportunities", [])
        except Exception as exc:
            logger.error("market.opportunities_failed", error=str(exc))
            return []
