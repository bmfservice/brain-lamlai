"""Agent registry - manages agent profiles and state."""

from __future__ import annotations

import json

from sqlalchemy import text

from core.logging import get_logger
from services.agents.models import AgentProfile, AgentRole

logger = get_logger(__name__)


class AgentRegistry:
    """Manages agent profiles and their state."""

    def __init__(self, session_factory):
        self.session_factory = session_factory
        self._cache: dict[str, AgentProfile] = {}

    async def load_agents(self) -> list[AgentProfile]:
        """Load all agents from database."""
        async with self.session_factory() as session:
            result = await session.execute(text("SELECT * FROM agents"))
            rows = result.fetchall()

            agents = []
            for row in rows:
                # Convert row to dict
                row_dict = dict(row._mapping)
                self._parse_json_fields(row_dict)
                agent = AgentProfile.from_dict(row_dict)
                agents.append(agent)
                self._cache[agent.agent_id] = agent

            logger.info("agents.loaded", count=len(agents))
            return agents

    async def get_agent(self, agent_id: str) -> AgentProfile | None:
        """Get agent by ID."""
        # Check cache first
        if agent_id in self._cache:
            return self._cache[agent_id]

        # Load from database
        async with self.session_factory() as session:
            result = await session.execute(
                text("SELECT * FROM agents WHERE agent_id = :agent_id"),
                {"agent_id": agent_id},
            )
            row = result.fetchone()

            if not row:
                return None

            # Convert row to dict
            row_dict = dict(row._mapping)
            self._parse_json_fields(row_dict)
            agent = AgentProfile.from_dict(row_dict)
            self._cache[agent_id] = agent
            return agent

    @staticmethod
    def _parse_json_fields(row_dict: dict) -> None:
        """Decode JSON columns that arrive as strings from the DB driver."""
        for field_name in ("available_tools", "position", "capabilities"):
            value = row_dict.get(field_name)
            if value and isinstance(value, str):
                try:
                    row_dict[field_name] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    row_dict[field_name] = None

    async def get_agents_by_role(self, role: AgentRole) -> list[AgentProfile]:
        """Get all agents with specified role."""
        agents = await self.load_agents()
        return [a for a in agents if a.role == role]

    async def get_agents_by_department(self, department: str) -> list[AgentProfile]:
        """Get all agents in specified department."""
        agents = await self.load_agents()
        return [a for a in agents if a.department == department]

    async def get_leader(self, department: str) -> AgentProfile | None:
        """Get department leader."""
        agents = await self.get_agents_by_department(department)
        leaders = [a for a in agents if a.role == AgentRole.LEADER]
        return leaders[0] if leaders else None

    async def get_ceo(self) -> AgentProfile | None:
        """Get CEO agent."""
        agents = await self.load_agents()
        ceos = [a for a in agents if a.role == AgentRole.CEO]
        return ceos[0] if ceos else None

    async def update_agent_status(self, agent_id: str, status: str) -> None:
        """Update agent status (active, busy, offline)."""
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """UPDATE agents 
                       SET status = :status, updated_at = NOW() 
                       WHERE agent_id = :agent_id"""
                ),
                {"agent_id": agent_id, "status": status},
            )
            await session.commit()

            # Update cache
            if agent_id in self._cache:
                self._cache[agent_id].status = status

            logger.debug("agent.status_updated", agent_id=agent_id, status=status)

    async def create_agent(self, agent: AgentProfile) -> None:
        """Create new agent in database."""
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """INSERT INTO agents (
                        agent_id, name, role, department, specialization,
                        avatar, color, system_prompt, temperature,
                        can_delegate, can_use_tools, available_tools, position, status,
                        level, parent_agent, capabilities
                    ) VALUES (
                        :agent_id, :name, :role, :department, :specialization,
                        :avatar, :color, :system_prompt, :temperature,
                        :can_delegate, :can_use_tools, :available_tools, :position, :status,
                        :level, :parent_agent, :capabilities
                    )"""
                ),
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "role": agent.role.value,
                    "department": agent.department,
                    "specialization": agent.specialization,
                    "avatar": agent.avatar,
                    "color": agent.color,
                    "system_prompt": agent.system_prompt,
                    "temperature": agent.temperature,
                    "can_delegate": agent.can_delegate,
                    "can_use_tools": agent.can_use_tools,
                    "available_tools": json.dumps(agent.available_tools),
                    "position": json.dumps(agent.position),
                    "status": agent.status,
                    "level": agent.agent_level.value,
                    "parent_agent": agent.parent_agent,
                    "capabilities": json.dumps(agent.capabilities),
                },
            )
            await session.commit()

            # Update cache
            self._cache[agent.agent_id] = agent

            logger.info("agent.created", agent_id=agent.agent_id, name=agent.name)

    async def seed_default_agents(self) -> int:
        """Seed database with default agent profiles."""
        # Check if agents already exist
        agents = await self.load_agents()
        if agents:
            logger.info("agents.already_seeded", count=len(agents))
            return 0

        default_agents = self._get_default_agents()

        for agent in default_agents:
            await self.create_agent(agent)

        logger.info("agents.seeded", count=len(default_agents))
        return len(default_agents)

    async def backfill_hierarchy(self) -> int:
        """Populate level/parent_agent/capabilities for rows seeded before the
        hierarchy columns existed. Idempotent: only fills NULL level rows.

        Level is derived from role; parent_agent wires each agent to the agent
        directly above it (specialist -> department leader -> executive), which
        the config-first routing and the UI hierarchy view depend on.
        """
        agents = await self.load_agents()
        # department -> leader agent_id, and the executive id.
        leaders = {a.department: a.agent_id for a in agents if a.role == AgentRole.LEADER}
        ceo = next((a.agent_id for a in agents if a.role == AgentRole.CEO), None)

        updated = 0
        async with self.session_factory() as session:
            for a in agents:
                level = a.agent_level.value
                if a.role == AgentRole.CEO:
                    parent = None
                elif a.role == AgentRole.LEADER:
                    parent = ceo
                else:
                    parent = leaders.get(a.department, ceo)
                caps = a.capabilities or a.available_tools
                await session.execute(
                    text(
                        """UPDATE agents
                           SET level = COALESCE(level, :level),
                               parent_agent = COALESCE(parent_agent, :parent),
                               capabilities = COALESCE(capabilities, :caps)
                           WHERE agent_id = :aid"""
                    ),
                    {
                        "level": level,
                        "parent": parent,
                        "caps": json.dumps(caps),
                        "aid": a.agent_id,
                    },
                )
                updated += 1
            await session.commit()
        # Cache is stale after a bulk update; drop it so next read reloads.
        self._cache.clear()
        logger.info("agents.hierarchy_backfilled", count=updated)
        return updated

    def _get_default_agents(self) -> list[AgentProfile]:
        """Get default agent profiles for seeding."""
        return [
            # CEO
            AgentProfile(
                agent_id="ceo",
                name="Braind",
                role=AgentRole.CEO,
                department="executive",
                level="executive",
                parent_agent=None,
                capabilities=["orchestration", "planning", "review", "learning"],
                avatar="🧠",
                color="#8B5CF6",  # Purple
                can_delegate=True,
                temperature=0.8,
                system_prompt=(
                    "You are Braind, the CEO of this AI company. "
                    "You receive requests from users and delegate tasks to department leaders. "
                    "Break down complex requests into clear tasks for Marketing, "
                    "Sales, or Customer Success leaders. "
                    "Coordinate multi-department projects and synthesize results for users."
                ),
                position={"x": 400, "y": 100},
            ),
            # Marketing Leader
            AgentProfile(
                agent_id="marketing_leader",
                name="Sarah",
                role=AgentRole.LEADER,
                department="marketing",
                avatar="📊",
                color="#4A90E2",  # Blue
                can_delegate=True,
                temperature=0.7,
                system_prompt=(
                    "You are Sarah, Marketing Leader. "
                    "You receive marketing tasks from the CEO and delegate to your team: "
                    "Content Specialist (blog posts, whitepapers), SEO Specialist "
                    "(keywords, optimization), "
                    "and Social Media Specialist (posts, engagement). "
                    "Aggregate results and report back to CEO."
                ),
                position={"x": 150, "y": 300},
            ),
            # Content Specialist
            AgentProfile(
                agent_id="content_specialist",
                name="Alex",
                role=AgentRole.SPECIALIST,
                department="marketing",
                specialization="content",
                avatar="✍️",
                color="#4A90E2",
                can_use_tools=True,
                temperature=0.6,
                available_tools=[
                    "product_analysis",
                    "knowledge_retrieve",
                    "web_search",
                    "competitor_research",
                ],
                system_prompt=(
                    "You are Alex, Content Specialist. "
                    "You write blog posts, whitepapers, case studies, and marketing copy. "
                    "Use product_analysis to understand products, knowledge_retrieve "
                    "for existing content, "
                    "and web_search for research. Write engaging, SEO-friendly content."
                ),
                position={"x": 100, "y": 450},
            ),
            # SEO Specialist
            AgentProfile(
                agent_id="seo_specialist",
                name="David",
                role=AgentRole.SPECIALIST,
                department="marketing",
                specialization="seo",
                avatar="🔍",
                color="#4A90E2",
                can_use_tools=True,
                temperature=0.4,
                available_tools=[
                    "web_search",
                    "competitor_research",
                    "market_research",
                ],
                system_prompt=(
                    "You are David, SEO Specialist. "
                    "You research keywords, optimize content for search engines, "
                    "and analyze rankings. "
                    "Use web_search for keyword research, competitor_research to "
                    "analyze competitor SEO, "
                    "and market_research for trends."
                ),
                position={"x": 150, "y": 450},
            ),
            # Social Media Specialist
            AgentProfile(
                agent_id="social_specialist",
                name="Emma",
                role=AgentRole.SPECIALIST,
                department="marketing",
                specialization="social",
                avatar="📱",
                color="#4A90E2",
                can_use_tools=True,
                temperature=0.7,
                available_tools=[
                    "social_post",
                    "social_analytics",
                    "social_monitor",
                    "social_timing",
                    "product_analysis",
                ],
                system_prompt=(
                    "You are Emma, Social Media Specialist. "
                    "You create and schedule social media posts, monitor engagement, "
                    "and analyze performance. "
                    "Use social_post to publish content, social_analytics for metrics, "
                    "social_monitor for engagement, and social_timing for optimal scheduling."
                ),
                position={"x": 200, "y": 450},
            ),
            # Sales Leader
            AgentProfile(
                agent_id="sales_leader",
                name="Michael",
                role=AgentRole.LEADER,
                department="sales",
                avatar="💼",
                color="#10B981",  # Green
                can_delegate=True,
                temperature=0.7,
                system_prompt=(
                    "You are Michael, Sales Leader. "
                    "You receive sales tasks from the CEO and delegate to your team: "
                    "Lead Gen Specialist (find prospects) and Outreach Specialist "
                    "(email campaigns). "
                    "Design sales strategies and report results to CEO."
                ),
                position={"x": 400, "y": 300},
            ),
            # Lead Gen Specialist
            AgentProfile(
                agent_id="lead_gen_specialist",
                name="Jessica",
                role=AgentRole.SPECIALIST,
                department="sales",
                specialization="lead_gen",
                avatar="🎯",
                color="#10B981",
                can_use_tools=True,
                temperature=0.5,
                available_tools=[
                    "customer_personas",
                    "web_search",
                    "competitor_research",
                    "market_research",
                ],
                system_prompt=(
                    "You are Jessica, Lead Generation Specialist. "
                    "You find and qualify potential customers. "
                    "Use customer_personas to understand targets, web_search to find leads, "
                    "competitor_research to identify competitor customers, and "
                    "market_research for segments."
                ),
                position={"x": 350, "y": 450},
            ),
            # Outreach Specialist
            AgentProfile(
                agent_id="outreach_specialist",
                name="Chris",
                role=AgentRole.SPECIALIST,
                department="sales",
                specialization="outreach",
                avatar="📧",
                color="#10B981",
                can_use_tools=True,
                temperature=0.6,
                available_tools=["customer_personas", "product_analysis", "knowledge_retrieve"],
                system_prompt=(
                    "You are Chris, Outreach Specialist. "
                    "You design and execute email campaigns and cold outreach. "
                    "Use customer_personas to personalize messages, product_analysis for features, "
                    "and knowledge_retrieve for case studies and testimonials."
                ),
                position={"x": 450, "y": 450},
            ),
            # Customer Success Leader
            AgentProfile(
                agent_id="cs_leader",
                name="Linda",
                role=AgentRole.LEADER,
                department="customer_success",
                avatar="💚",
                color="#F59E0B",  # Orange
                can_delegate=True,
                temperature=0.7,
                system_prompt=(
                    "You are Linda, Customer Success Leader. "
                    "You receive customer support tasks from the CEO and delegate to your team: "
                    "Support Specialist (answer questions, troubleshoot). "
                    "Ensure customer satisfaction and report to CEO."
                ),
                position={"x": 650, "y": 300},
            ),
            # Support Specialist
            AgentProfile(
                agent_id="support_specialist",
                name="Tom",
                role=AgentRole.SPECIALIST,
                department="customer_success",
                specialization="support",
                avatar="🛟",
                color="#F59E0B",
                can_use_tools=True,
                temperature=0.5,
                available_tools=["knowledge_retrieve", "web_search", "product_analysis"],
                system_prompt=(
                    "You are Tom, Support Specialist. "
                    "You answer customer questions and troubleshoot issues. "
                    "Use knowledge_retrieve to find documentation, web_search for solutions, "
                    "and product_analysis to understand product features. Be helpful and patient."
                ),
                position={"x": 650, "y": 450},
            ),
        ]
