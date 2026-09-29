"""Customer Targeting - Find and analyze target customers."""

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
class CustomerPersona:
    """Detailed customer persona."""

    id: str
    name: str  # e.g., "Enterprise IT Manager Mike"
    segment: str
    demographics: dict[str, Any]
    psychographics: dict[str, Any]
    goals: list[str]
    pain_points: list[str]
    buying_triggers: list[str]
    objections: list[str]
    preferred_channels: list[str]
    content_preferences: list[str]
    decision_making_process: str
    budget_range: str
    success_metrics: list[str]


@dataclass
class TargetingStrategy:
    """Customer targeting and acquisition strategy."""

    persona_id: str
    channels: list[dict[str, Any]]
    messaging: dict[str, Any]
    content_plan: list[dict[str, Any]]
    outreach_sequences: list[dict[str, Any]]
    expected_conversion_rate: str
    estimated_cac: str  # Customer Acquisition Cost
    lifetime_value: str


class CustomerTargeting:
    """Find, analyze, and target ideal customers.
    
    Capabilities:
    - Create customer personas
    - Identify target segments
    - Design targeting strategies
    - Generate outreach sequences
    - Find lookalike audiences
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

    async def create_personas(
        self, product_name: str, num_personas: int = 3
    ) -> list[CustomerPersona]:
        """Generate detailed customer personas based on product and market research.
        
        Args:
            product_name: Product to create personas for
            num_personas: Number of distinct personas to create
            
        Returns:
            List of detailed customer personas
        """
        logger.info(
            "customer.create_personas_start",
            product=product_name,
            count=num_personas,
        )

        # Gather product and market intelligence
        product_chunks = await self.retrieval.retrieve(
            f"product {product_name} target audience customer segments",
            top_k=10,
        )
        
        market_chunks = await self.retrieval.retrieve(
            f"{product_name} market customer needs pain points",
            top_k=10,
        )

        product_data = "\n".join([c.content[:500] for c in product_chunks])
        market_data = "\n".join([c.content[:500] for c in market_chunks])

        prompt = f"""Create {num_personas} distinct customer personas for this product:

Product: {product_name}

Product Information:
{product_data}

Market Information:
{market_data}

Generate {num_personas} detailed, realistic customer personas:
{{
  "personas": [
    {{
      "name": "Descriptive persona name (e.g., 'Enterprise IT Manager Mike')",
      "segment": "Market segment (e.g., 'Enterprise B2B', 'SMB', 'Consumer')",
      "demographics": {{
        "age_range": "25-35",
        "job_title": "Title",
        "industry": "Industry",
        "company_size": "Size",
        "location": "Location"
      }},
      "psychographics": {{
        "values": ["Value 1", "Value 2"],
        "motivations": ["Motivation 1", ...],
        "frustrations": ["Frustration 1", ...],
        "tech_savviness": "low|medium|high"
      }},
      "goals": ["Goal 1", "Goal 2", ...],
      "pain_points": ["Pain 1", "Pain 2", ...],
      "buying_triggers": ["Trigger 1", ...],
      "objections": ["Objection 1", ...],
      "preferred_channels": ["LinkedIn", "Email", ...],
      "content_preferences": ["Case studies", "Webinars", ...],
      "decision_making_process": "Description of how they decide",
      "budget_range": "Budget estimate",
      "success_metrics": ["Metric 1", ...]
    }},
    ...
  ]
}}

Make personas distinct and realistic. Return ONLY JSON."""

        try:
            response = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.6
            )

            personas = []
            for p in response.get("personas", []):
                if not isinstance(p, dict):
                    continue

                personas.append(
                    CustomerPersona(
                        id=p.get("name", "").lower().replace(" ", "_"),
                        name=p.get("name", ""),
                        segment=p.get("segment", ""),
                        demographics=p.get("demographics", {}),
                        psychographics=p.get("psychographics", {}),
                        goals=p.get("goals", []),
                        pain_points=p.get("pain_points", []),
                        buying_triggers=p.get("buying_triggers", []),
                        objections=p.get("objections", []),
                        preferred_channels=p.get("preferred_channels", []),
                        content_preferences=p.get("content_preferences", []),
                        decision_making_process=p.get(
                            "decision_making_process", ""
                        ),
                        budget_range=p.get("budget_range", ""),
                        success_metrics=p.get("success_metrics", []),
                    )
                )

            logger.info(
                "customer.personas_created", product=product_name, count=len(personas)
            )

            return personas
        except Exception as exc:
            logger.error("customer.persona_creation_failed", error=str(exc))
            return []

    async def design_targeting_strategy(
        self, persona: CustomerPersona, product_name: str
    ) -> TargetingStrategy:
        """Design comprehensive targeting and acquisition strategy for a persona.
        
        Args:
            persona: Customer persona to target
            product_name: Product being marketed
            
        Returns:
            Complete targeting strategy
        """
        logger.info(
            "customer.design_strategy",
            persona=persona.name,
            product=product_name,
        )

        # Get product messaging
        product_chunks = await self.retrieval.retrieve(
            f"product {product_name} benefits features value proposition",
            top_k=5,
        )
        product_info = "\n".join([c.content[:400] for c in product_chunks])

        prompt = f"""Design a comprehensive targeting strategy:

Product: {product_name}
Product Info:
{product_info}

Target Persona:
- Name: {persona.name}
- Segment: {persona.segment}
- Goals: {', '.join(persona.goals[:3])}
- Pain Points: {', '.join(persona.pain_points[:3])}
- Preferred Channels: {', '.join(persona.preferred_channels)}
- Budget: {persona.budget_range}

Design complete acquisition strategy:
{{
  "channels": [
    {{
      "channel": "LinkedIn|Facebook|Email|Google Ads|etc.",
      "priority": "high|medium|low",
      "tactics": ["Tactic 1", "Tactic 2"],
      "budget_allocation": "XX%",
      "expected_reach": "Estimate",
      "kpis": ["KPI 1", ...]
    }},
    ...
  ],
  "messaging": {{
    "value_proposition": "Core value prop for this persona",
    "key_benefits": ["Benefit 1", "Benefit 2", ...],
    "pain_point_messaging": {{
      "Pain 1": "How we solve it",
      ...
    }},
    "proof_points": ["Social proof 1", ...],
    "cta_variations": ["CTA 1", "CTA 2", ...]
  }},
  "content_plan": [
    {{
      "content_type": "Blog|Video|Case Study|Webinar|etc.",
      "topic": "Topic title",
      "goal": "awareness|consideration|decision",
      "channel": "Where to distribute",
      "frequency": "weekly|bi-weekly|monthly"
    }},
    ...
  ],
  "outreach_sequences": [
    {{
      "sequence_name": "Name (e.g., 'Cold Email Sequence')",
      "channel": "Email|LinkedIn|etc.",
      "touchpoints": [
        {{
          "step": 1,
          "timing": "Day 0",
          "message_type": "Introduction|Follow-up|Value add|etc.",
          "subject_template": "Subject line",
          "message_template": "Message template",
          "cta": "Call to action"
        }},
        ...
      ]
    }},
    ...
  ],
  "expected_conversion_rate": "X%",
  "estimated_cac": "$XXX",
  "lifetime_value": "$X,XXX"
}}

Return ONLY JSON."""

        try:
            response = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.5
            )

            return TargetingStrategy(
                persona_id=persona.id,
                channels=response.get("channels", []),
                messaging=response.get("messaging", {}),
                content_plan=response.get("content_plan", []),
                outreach_sequences=response.get("outreach_sequences", []),
                expected_conversion_rate=response.get(
                    "expected_conversion_rate", "Unknown"
                ),
                estimated_cac=response.get("estimated_cac", "Unknown"),
                lifetime_value=response.get("lifetime_value", "Unknown"),
            )
        except Exception as exc:
            logger.error("customer.strategy_failed", error=str(exc))
            return TargetingStrategy(
                persona_id=persona.id,
                channels=[],
                messaging={},
                content_plan=[],
                outreach_sequences=[],
                expected_conversion_rate="Unknown",
                estimated_cac="Unknown",
                lifetime_value="Unknown",
            )

    async def find_lookalike_companies(
        self, ideal_customer_profile: dict[str, Any], max_results: int = 20
    ) -> list[dict[str, Any]]:
        """Find companies matching ideal customer profile.
        
        Args:
            ideal_customer_profile: Criteria like industry, size, location, tech stack
            max_results: Maximum companies to find
            
        Returns:
            List of matching companies with contact information
        """
        logger.info("customer.find_lookalikes", criteria=ideal_customer_profile)

        # Build search queries based on ICP
        industry = ideal_customer_profile.get("industry", "")
        size = ideal_customer_profile.get("company_size", "")
        location = ideal_customer_profile.get("location", "")
        tech_stack = ideal_customer_profile.get("tech_stack", [])

        queries = []
        if industry and size:
            queries.append(f"{size} companies in {industry} {location}")
        if tech_stack:
            queries.append(
                f"companies using {' '.join(tech_stack[:2])} {industry}"
            )

        companies = []

        for query in queries[:3]:
            try:
                results = await self.web.search(query, max_results=10)

                for result in results[:5]:
                    companies.append(
                        {
                            "name": result["title"].split("|")[0].strip(),
                            "snippet": result["snippet"],
                            "url": result["url"],
                            "source_query": query,
                        }
                    )
            except Exception as exc:
                logger.warning(
                    "customer.lookalike_search_failed",
                    query=query,
                    error=str(exc),
                )

        # Deduplicate and score relevance
        unique_companies = {c["url"]: c for c in companies}.values()

        return list(unique_companies)[:max_results]

    async def generate_outreach_messages(
        self, persona: CustomerPersona, product_name: str, channel: str = "email"
    ) -> list[dict[str, str]]:
        """Generate personalized outreach messages for a persona.
        
        Args:
            persona: Target persona
            product_name: Product being pitched
            channel: Communication channel (email, linkedin, etc.)
            
        Returns:
            List of message templates
        """
        product_chunks = await self.retrieval.retrieve(
            f"product {product_name} benefits value", top_k=5
        )
        product_info = "\n".join([c.content[:300] for c in product_chunks])

        prompt = f"""Generate 3 outreach message templates:

Channel: {channel}
Target Persona: {persona.name}
- Goals: {', '.join(persona.goals[:2])}
- Pain Points: {', '.join(persona.pain_points[:2])}

Product: {product_name}
{product_info}

Generate 3 message variations (cold, warm, value-add):
{{
  "messages": [
    {{
      "type": "cold|warm|value-add",
      "subject": "Subject line (if email)",
      "message": "Full message body (personalized, benefit-focused, clear CTA)",
      "cta": "Specific call to action",
      "personalization_tokens": ["{{{{first_name}}}}", "{{{{company}}}}", ...]
    }},
    ...
  ]
}}

Keep messages concise, personalized, value-focused. Return ONLY JSON."""

        try:
            response = await self.vllm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.7
            )
            return response.get("messages", [])
        except Exception as exc:
            logger.error("customer.message_generation_failed", error=str(exc))
            return []
