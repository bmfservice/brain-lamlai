"""Product Analyzer - Learn and analyze own products."""

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
class ProductProfile:
    """Complete product profile with all analyzed information."""

    id: str
    name: str
    description: str
    category: str
    features: list[str]
    benefits: list[str]
    target_audience: dict[str, Any]
    pricing: dict[str, Any]
    unique_selling_points: list[str]
    use_cases: list[str]
    competitors: list[str]
    strengths: list[str]
    weaknesses: list[str]
    market_position: str
    keywords: list[str]
    sentiment: str  # positive, neutral, negative
    confidence_score: float


class ProductAnalyzer:
    """Analyze and learn about company's own products.
    
    Capabilities:
    - Ingest product documentation
    - Extract features, benefits, USPs
    - Identify target audience
    - Analyze positioning
    - Generate marketing angles
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

    async def learn_product(
        self,
        product_name: str,
        *,
        documentation_urls: list[str] | None = None,
        documentation_text: str | None = None,
        fetch_from_web: bool = True,
    ) -> ProductProfile:
        """Learn everything about a product from various sources.
        
        Args:
            product_name: Name of the product
            documentation_urls: List of documentation/product page URLs
            documentation_text: Direct product documentation text
            fetch_from_web: Whether to search web for additional info
            
        Returns:
            Complete product profile with analysis
        """
        logger.info("product.learn_start", product=product_name)

        # Step 1: Ingest documentation
        if documentation_text:
            await self.ingest.ingest_text(
                documentation_text,
                title=f"Product Documentation: {product_name}",
                source_type="product_docs",
                tags=["product", product_name.lower()],
                meta={"product": product_name, "doc_type": "official"},
            )

        if documentation_urls:
            for url in documentation_urls:
                try:
                    page_data = await self.web.fetch_page_text(url, max_chars=10000)
                    await self.ingest.ingest_text(
                        page_data["text"],
                        title=page_data.get("title") or f"Product: {product_name}",
                        source_type="product_docs",
                        url=url,
                        tags=["product", product_name.lower()],
                        meta={"product": product_name, "doc_type": "official"},
                    )
                except Exception as exc:
                    logger.warning(
                        "product.fetch_url_failed", url=url, error=str(exc)
                    )

        # Step 2: Search web for reviews, comparisons, mentions
        if fetch_from_web:
            await self._learn_from_web(product_name)

        # Step 3: Retrieve all product knowledge
        chunks = await self.retrieval.retrieve(
            f"product information features benefits {product_name}", top_k=20
        )

        # Step 4: AI analysis
        profile = await self._analyze_product(product_name, chunks)

        logger.info(
            "product.learn_complete",
            product=product_name,
            features=len(profile.features),
            usps=len(profile.unique_selling_points),
        )

        return profile

    async def _learn_from_web(self, product_name: str) -> None:
        """Search web and ingest product-related content."""
        queries = [
            f"{product_name} features and benefits",
            f"{product_name} review comparison",
            f"{product_name} use cases examples",
            f"{product_name} pricing plans",
        ]

        for query in queries:
            try:
                results = await self.web.search(query, max_results=5)
                for result in results[:3]:  # Top 3 per query
                    try:
                        page = await self.web.fetch_page_text(
                            result["url"], max_chars=8000
                        )
                        await self.ingest.ingest_text(
                            f"{page['title']}\n\n{page['text']}",
                            title=page["title"],
                            source_type="web",
                            url=result["url"],
                            tags=["product", product_name.lower(), "research"],
                            meta={"product": product_name, "source": "web_search"},
                        )
                    except Exception as exc:
                        logger.warning(
                            "product.web_ingest_failed",
                            url=result.get("url"),
                            error=str(exc),
                        )
            except Exception as exc:
                logger.warning(
                    "product.web_search_failed", query=query, error=str(exc)
                )

    async def _analyze_product(
        self, product_name: str, chunks: list[Any]
    ) -> ProductProfile:
        """Use AI to analyze product information and create profile."""
        # Aggregate content
        content = "\n\n".join(
            [f"Source: {c.title}\n{c.content[:800]}" for c in chunks[:15]]
        )

        prompt = f"""Analyze this product and create a comprehensive profile:

Product Name: {product_name}

Information gathered:
{content}

Generate a detailed JSON analysis with:
{{
  "name": "{product_name}",
  "description": "Clear 2-3 sentence description",
  "category": "Product category (e.g., SaaS, E-commerce, Service)",
  "features": ["Feature 1", "Feature 2", ...],
  "benefits": ["Benefit 1", "Benefit 2", ...],
  "target_audience": {{
    "segments": ["Segment 1", "Segment 2"],
    "demographics": "Age, profession, etc.",
    "psychographics": "Pain points, goals, behaviors"
  }},
  "pricing": {{
    "model": "subscription|one-time|freemium|custom",
    "tiers": ["Free", "Pro", "Enterprise"],
    "notes": "Any pricing details"
  }},
  "unique_selling_points": ["USP 1", "USP 2", "USP 3"],
  "use_cases": ["Use case 1", "Use case 2", ...],
  "competitors": ["Competitor 1", "Competitor 2", ...],
  "strengths": ["Strength 1", "Strength 2", ...],
  "weaknesses": ["Weakness 1", "Weakness 2", ...],
  "market_position": "market leader|challenger|niche|emerging",
  "keywords": ["keyword1", "keyword2", ...],
  "sentiment": "positive|neutral|negative",
  "confidence_score": 0.0-1.0
}}

Extract from available information. If data is missing, use "Unknown" but infer \
intelligently from context.
Return ONLY valid JSON, no markdown."""

        try:
            response = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.3
            )

            # Parse into ProductProfile
            return ProductProfile(
                id=response.get("name", product_name).lower().replace(" ", "_"),
                name=response.get("name", product_name),
                description=response.get("description", ""),
                category=response.get("category", "Unknown"),
                features=response.get("features", []),
                benefits=response.get("benefits", []),
                target_audience=response.get("target_audience", {}),
                pricing=response.get("pricing", {}),
                unique_selling_points=response.get("unique_selling_points", []),
                use_cases=response.get("use_cases", []),
                competitors=response.get("competitors", []),
                strengths=response.get("strengths", []),
                weaknesses=response.get("weaknesses", []),
                market_position=response.get("market_position", "Unknown"),
                keywords=response.get("keywords", []),
                sentiment=response.get("sentiment", "neutral"),
                confidence_score=float(response.get("confidence_score", 0.7)),
            )
        except Exception as exc:
            logger.error("product.analysis_failed", error=str(exc))
            # Return minimal profile
            return ProductProfile(
                id=product_name.lower().replace(" ", "_"),
                name=product_name,
                description="",
                category="Unknown",
                features=[],
                benefits=[],
                target_audience={},
                pricing={},
                unique_selling_points=[],
                use_cases=[],
                competitors=[],
                strengths=[],
                weaknesses=[],
                market_position="Unknown",
                keywords=[],
                sentiment="neutral",
                confidence_score=0.0,
            )

    async def get_product_knowledge(self, product_name: str) -> dict[str, Any]:
        """Retrieve existing knowledge about a product from memory."""
        chunks = await self.retrieval.retrieve(
            f"product {product_name} features benefits", top_k=10
        )

        if not chunks:
            return {
                "found": False,
                "message": (
                    f"No knowledge about {product_name} in memory. "
                    "Use learn_product() first."
                ),
            }

        return {
            "found": True,
            "product": product_name,
            "chunks": [
                {
                    "title": c.title,
                    "excerpt": c.content[:300],
                    "score": c.score,
                    "url": c.meta.get("url") if isinstance(c.meta, dict) else None,
                }
                for c in chunks
            ],
        }

    async def compare_with_competitors(
        self, our_product: str, competitors: list[str]
    ) -> dict[str, Any]:
        """Compare our product with competitors across key dimensions."""
        # First ensure we have knowledge about all products
        all_products = [our_product] + competitors

        comparison_data = {}
        for product in all_products:
            chunks = await self.retrieval.retrieve(
                f"product {product} features pricing target audience", top_k=5
            )
            comparison_data[product] = "\n".join([c.content[:500] for c in chunks])

        # AI-powered comparison
        prompt = f"""Compare these products and generate a competitive analysis:

Our Product: {our_product}
{comparison_data.get(our_product, 'No data')}

Competitors: {', '.join(competitors)}
{"".join([f"\n\n{comp}:\n{comparison_data.get(comp, 'No data')}" for comp in competitors])}

Generate detailed JSON comparison:
{{
  "dimensions": [
    {{
      "name": "Features",
      "our_rating": 1-5,
      "competitor_ratings": {{"Comp1": 1-5, "Comp2": 1-5}},
      "analysis": "What we do better/worse",
      "recommendations": ["Recommendation 1", ...]
    }},
    {{
      "name": "Pricing",
      ...
    }},
    {{
      "name": "Target Audience",
      ...
    }},
    {{
      "name": "User Experience",
      ...
    }},
    {{
      "name": "Market Position",
      ...
    }}
  ],
  "overall_summary": "Overall competitive standing",
  "opportunities": ["Opportunity 1", "Opportunity 2", ...],
  "threats": ["Threat 1", "Threat 2", ...],
  "recommended_actions": ["Action 1", "Action 2", ...]
}}

Return ONLY JSON."""

        try:
            comparison = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.3
            )
            return comparison
        except Exception as exc:
            logger.error("product.comparison_failed", error=str(exc))
            return {"error": str(exc)}

    async def generate_marketing_angles(
        self, product_name: str, campaign_goal: str = "awareness"
    ) -> list[dict[str, Any]]:
        """Generate marketing angles/messaging for the product.
        
        Args:
            product_name: Product to generate angles for
            campaign_goal: awareness|consideration|conversion|retention
            
        Returns:
            List of marketing angles with messaging
        """
        # Get product knowledge
        chunks = await self.retrieval.retrieve(
            f"product {product_name} features benefits audience", top_k=8
        )

        content = "\n\n".join([f"{c.title}\n{c.content[:400]}" for c in chunks])

        prompt = f"""Generate marketing angles for this product:

Product: {product_name}
Campaign Goal: {campaign_goal}

Product Information:
{content}

Generate 5-7 distinct marketing angles, each with:
{{
  "angles": [
    {{
      "angle_name": "Angle name (e.g., Time-Saver, Cost-Effective, Innovation Leader)",
      "target_segment": "Which audience segment this appeals to",
      "key_message": "Core message (1-2 sentences)",
      "supporting_points": ["Point 1", "Point 2", "Point 3"],
      "emotional_appeal": "Fear|Trust|Belonging|Pride|Joy",
      "proof_points": ["Statistic/testimonial 1", ...],
      "cta": "Recommended call-to-action",
      "channels": ["LinkedIn", "Facebook", "Email", ...],
      "content_ideas": ["Idea 1", "Idea 2", ...]
    }},
    ...
  ]
}}

Return ONLY JSON."""

        try:
            result = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.7
            )
            return result.get("angles", [])
        except Exception as exc:
            logger.error("product.angles_failed", error=str(exc))
            return []
