"""Competitor Analyzer - Research and track competitors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.logging import get_logger
from core.vllm import VLLMClient
from services.ingest.service import IngestService
from services.rag.pipeline import UnifiedRetrievalService
from services.tools.web import WebSearchTool

logger = get_logger(__name__)


@dataclass
class CompetitorProfile:
    """Complete competitor profile."""

    id: str
    name: str
    website: str | None
    description: str
    category: str
    features: list[str]
    pricing: dict[str, Any]
    target_audience: str
    strengths: list[str]
    weaknesses: list[str]
    market_share: str  # estimated
    recent_updates: list[dict[str, Any]]
    customer_reviews_sentiment: str
    threat_level: str  # low|medium|high|critical
    last_analyzed: datetime


class CompetitorAnalyzer:
    """Analyze competitors through web research and monitoring.
    
    Capabilities:
    - Discover competitors
    - Deep competitor research
    - Track competitor activities
    - SWOT analysis
    - Competitive intelligence gathering
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

    async def discover_competitors(
        self, our_product: str, our_category: str, max_competitors: int = 10
    ) -> list[str]:
        """Automatically discover competitors via web search.
        
        Args:
            our_product: Our product name
            our_category: Product category (e.g., "CRM software", "E-commerce platform")
            max_competitors: Maximum number of competitors to find
            
        Returns:
            List of competitor names
        """
        logger.info(
            "competitor.discover_start",
            product=our_product,
            category=our_category,
        )

        # Search strategies
        queries = [
            f"{our_category} alternatives to {our_product}",
            f"best {our_category} tools comparison",
            f"{our_product} vs competitors",
            f"top {our_category} software 2026",
        ]

        all_competitors = set()

        for query in queries:
            try:
                results = await self.web.search(query, max_results=10)

                # Extract competitor mentions from snippets and titles
                for result in results:
                    text = f"{result['title']} {result['snippet']}"

                    # Use AI to extract competitor names
                    prompt = f"""Extract competitor product names from this text:

"{text}"

Our product: {our_product}
Category: {our_category}

Rules:
- Only extract product/company names, not generic terms
- Exclude our own product: {our_product}
- Only products in the same category: {our_category}

Return JSON: {{"competitors": ["Name1", "Name2", ...]}}"""

                    try:
                        response = await self.vllm.chat_json(
                            [{"role": "user", "content": prompt}],
                            temperature=0.1,
                        )
                        competitors = response.get("competitors", [])
                        all_competitors.update(competitors)
                    except Exception as exc:
                        logger.warning(
                            "competitor.extract_failed", error=str(exc)
                        )

            except Exception as exc:
                logger.warning(
                    "competitor.search_failed", query=query, error=str(exc)
                )

        # Filter and rank by relevance
        competitors_list = list(all_competitors)[:max_competitors]

        logger.info(
            "competitor.discover_complete",
            found=len(competitors_list),
            competitors=competitors_list,
        )

        return competitors_list

    async def research_competitor(
        self, competitor_name: str, deep_research: bool = True
    ) -> CompetitorProfile:
        """Deep research on a specific competitor.
        
        Args:
            competitor_name: Competitor product/company name
            deep_research: If True, performs extensive web research
            
        Returns:
            Complete competitor profile
        """
        logger.info("competitor.research_start", competitor=competitor_name)

        if deep_research:
            await self._deep_web_research(competitor_name)

        # Retrieve all gathered intelligence
        chunks = await self.retrieval.retrieve(
            f"competitor {competitor_name} features pricing strengths weaknesses",
            top_k=20,
        )

        # AI analysis
        profile = await self._analyze_competitor(competitor_name, chunks)

        logger.info(
            "competitor.research_complete",
            competitor=competitor_name,
            threat_level=profile.threat_level,
        )

        return profile

    async def _deep_web_research(self, competitor_name: str) -> None:
        """Perform comprehensive web research on competitor."""
        research_queries = [
            f"{competitor_name} features and pricing",
            f"{competitor_name} review pros and cons",
            f"{competitor_name} customer complaints issues",
            f"{competitor_name} news updates 2026",
            f"{competitor_name} market share statistics",
            f"{competitor_name} target audience demographics",
        ]

        for query in research_queries:
            try:
                results = await self.web.search(query, max_results=5)

                for result in results[:3]:
                    try:
                        page = await self.web.fetch_page_text(
                            result["url"], max_chars=8000
                        )

                        await self.ingest.ingest_text(
                            f"{page['title']}\n\n{page['text']}",
                            title=f"Competitor Intel: {page['title']}",
                            source_type="competitor_research",
                            url=result["url"],
                            tags=[
                                "competitor",
                                competitor_name.lower().replace(" ", "_"),
                                "research",
                            ],
                            meta={
                                "competitor": competitor_name,
                                "source": "web_search",
                                "query": query,
                            },
                        )
                    except Exception as exc:
                        logger.warning(
                            "competitor.page_fetch_failed",
                            url=result.get("url"),
                            error=str(exc),
                        )
            except Exception as exc:
                logger.warning(
                    "competitor.search_failed", query=query, error=str(exc)
                )

    async def _analyze_competitor(
        self, competitor_name: str, chunks: list[Any]
    ) -> CompetitorProfile:
        """AI-powered competitor analysis."""
        content = "\n\n".join(
            [f"Source: {c.title}\n{c.content[:800]}" for c in chunks[:15]]
        )

        prompt = f"""Analyze this competitor and create a comprehensive intelligence profile:

Competitor: {competitor_name}

Gathered Intelligence:
{content}

Generate detailed JSON analysis:
{{
  "name": "{competitor_name}",
  "website": "https://...",
  "description": "Clear 2-3 sentence description",
  "category": "Product category",
  "features": ["Feature 1", "Feature 2", ...],
  "pricing": {{
    "model": "subscription|one-time|freemium|custom",
    "tiers": ["Tier 1", "Tier 2"],
    "starting_price": "$X/month",
    "notes": "Pricing observations"
  }},
  "target_audience": "Who they target (demographics, industries, etc.)",
  "strengths": ["Strength 1", "Strength 2", ...],
  "weaknesses": ["Weakness 1", "Weakness 2", ...],
  "market_share": "dominant|significant|moderate|small|unknown",
  "recent_updates": [
    {{"date": "2026-XX", "update": "Description", "impact": "high|medium|low"}},
    ...
  ],
  "customer_reviews_sentiment": "very positive|positive|neutral|negative|very negative",
  "threat_level": "low|medium|high|critical",
  "threat_reasoning": "Why this threat level"
}}

Base analysis on available intelligence. Use "Unknown" if data missing.
Return ONLY valid JSON."""

        try:
            response = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.3
            )

            return CompetitorProfile(
                id=competitor_name.lower().replace(" ", "_"),
                name=response.get("name", competitor_name),
                website=response.get("website"),
                description=response.get("description", ""),
                category=response.get("category", "Unknown"),
                features=response.get("features", []),
                pricing=response.get("pricing", {}),
                target_audience=response.get("target_audience", "Unknown"),
                strengths=response.get("strengths", []),
                weaknesses=response.get("weaknesses", []),
                market_share=response.get("market_share", "unknown"),
                recent_updates=response.get("recent_updates", []),
                customer_reviews_sentiment=response.get(
                    "customer_reviews_sentiment", "neutral"
                ),
                threat_level=response.get("threat_level", "medium"),
                last_analyzed=datetime.now(),
            )
        except Exception as exc:
            logger.error("competitor.analysis_failed", error=str(exc))
            return CompetitorProfile(
                id=competitor_name.lower().replace(" ", "_"),
                name=competitor_name,
                website=None,
                description="",
                category="Unknown",
                features=[],
                pricing={},
                target_audience="Unknown",
                strengths=[],
                weaknesses=[],
                market_share="unknown",
                recent_updates=[],
                customer_reviews_sentiment="neutral",
                threat_level="unknown",
                last_analyzed=datetime.now(),
            )

    async def monitor_competitor(
        self, competitor_name: str
    ) -> dict[str, Any]:
        """Monitor competitor for recent changes and updates.
        
        Returns:
            Recent activities, news, product updates
        """
        logger.info("competitor.monitor", competitor=competitor_name)

        # Search for recent news and updates
        queries = [
            f"{competitor_name} news last 30 days",
            f"{competitor_name} product update 2026",
            f"{competitor_name} announcement",
        ]

        updates = []

        for query in queries:
            try:
                results = await self.web.search(query, max_results=5)

                for result in results[:3]:
                    updates.append(
                        {
                            "title": result["title"],
                            "snippet": result["snippet"],
                            "url": result["url"],
                            "query": query,
                        }
                    )
            except Exception as exc:
                logger.warning(
                    "competitor.monitor_search_failed",
                    query=query,
                    error=str(exc),
                )

        # AI summarization of updates
        if updates:
            updates_text = "\n\n".join(
                [f"- {u['title']}: {u['snippet']}" for u in updates]
            )

            prompt = f"""Analyze these recent updates about {competitor_name}:

{updates_text}

Summarize:
1. Key changes or announcements
2. Potential impact on market
3. Recommended response for us

Return JSON:
{{
  "summary": "Brief summary of updates",
  "key_changes": ["Change 1", "Change 2", ...],
  "impact_assessment": "low|medium|high",
  "recommended_actions": ["Action 1", "Action 2", ...],
  "threats": ["Threat 1", ...],
  "opportunities": ["Opportunity 1", ...]
}}"""

            try:
                analysis = await self.vllm.chat_json(
                    [{"role": "user", "content": prompt}], temperature=0.3
                )

                return {
                    "competitor": competitor_name,
                    "updates": updates,
                    "analysis": analysis,
                }
            except Exception as exc:
                logger.error("competitor.analysis_failed", error=str(exc))

        return {"competitor": competitor_name, "updates": updates, "analysis": {}}

    async def swot_analysis(
        self, our_product: str, competitors: list[str]
    ) -> dict[str, Any]:
        """Generate SWOT analysis comparing us with competitors.
        
        Args:
            our_product: Our product name
            competitors: List of competitor names
            
        Returns:
            Comprehensive SWOT analysis
        """
        # Gather intelligence on all parties
        all_products = [our_product] + competitors

        intelligence = {}
        for product in all_products:
            chunks = await self.retrieval.retrieve(
                f"product {product} features strengths weaknesses market",
                top_k=8,
            )
            intelligence[product] = "\n".join([c.content[:600] for c in chunks])

        # AI-powered SWOT
        prompt = f"""Generate a comprehensive SWOT analysis:

Our Product: {our_product}
{intelligence.get(our_product, 'Limited data')}

Competitors: {', '.join(competitors)}
{"".join([f"\n\n{comp}:\n{intelligence.get(comp, 'Limited data')}" for comp in competitors])}

Generate detailed SWOT analysis JSON:
{{
  "strengths": [
    {{
      "strength": "What we do well",
      "evidence": "Supporting evidence",
      "competitive_advantage": "How this beats competitors"
    }},
    ...
  ],
  "weaknesses": [
    {{
      "weakness": "What we lack",
      "impact": "high|medium|low",
      "competitor_advantage": "Who does this better",
      "mitigation": "How to address"
    }},
    ...
  ],
  "opportunities": [
    {{
      "opportunity": "Market opportunity",
      "potential": "high|medium|low",
      "actions": ["Action 1", "Action 2"],
      "timeline": "immediate|short-term|long-term"
    }},
    ...
  ],
  "threats": [
    {{
      "threat": "Competitive threat",
      "severity": "critical|high|medium|low",
      "source": "Which competitor",
      "response": ["Response 1", "Response 2"]
    }},
    ...
  ],
  "strategic_recommendations": ["Recommendation 1", ...]
}}

Return ONLY JSON."""

        try:
            swot = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.3
            )
            return swot
        except Exception as exc:
            logger.error("competitor.swot_failed", error=str(exc))
            return {"error": str(exc)}
