"""Read-only aggregation of live workflow state and progress reporting.

Every value this module returns is derived from committed rows in
``agents``, ``agent_tasks``, ``agent_reports`` and ``agent_audit_events``.
Nothing is simulated: an agent with no live task is reported ``idle``, not
``working`` (spec §36). Where the schema carries no deadline column, this
module reports the signal it actually has rather than inventing one — see
``_attention_count`` and ``first_pass_rate``.

This view is a projection used for progress reporting in chat, workflow
tracking, and diagnostics. It never writes, so it can be queried without
competing with the orchestration engine for row locks.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import bindparam, text

from core.logging import get_logger
from services.agents.domain import AgentStatus, TaskState

logger = get_logger(__name__)

#: A task in one of these states is finished; its agent is no longer on it.
_TERMINAL = ("completed", "failed")

#: Agent status implied by the state of the task it currently holds.
_STATE_TO_AGENT_STATUS: dict[str, AgentStatus] = {
    TaskState.CREATED.value: AgentStatus.WAITING,
    TaskState.ANALYZING.value: AgentStatus.ANALYZING,
    TaskState.PLANNED.value: AgentStatus.THINKING,
    TaskState.DISPATCHED.value: AgentStatus.THINKING,
    TaskState.IN_PROGRESS.value: AgentStatus.WORKING,
    TaskState.REVIEWING.value: AgentStatus.REVIEWING,
    TaskState.WAITING_FOR_USER.value: AgentStatus.WAITING,
    TaskState.CONTINUING.value: AgentStatus.WORKING,
    TaskState.FINAL_VALIDATION.value: AgentStatus.REVIEWING,
    TaskState.BLOCKED.value: AgentStatus.BLOCKED,
    TaskState.REWORK_REQUIRED.value: AgentStatus.WAITING,
    TaskState.FAILED.value: AgentStatus.ERROR,
    TaskState.COMPLETED.value: AgentStatus.COMPLETED,
}

#: When an agent holds several live tasks, the most "active" state wins so the
#: roster shows what the agent is really doing rather than an arbitrary row.
_STATE_SALIENCE: dict[str, int] = {
    TaskState.IN_PROGRESS.value: 100,
    TaskState.ANALYZING.value: 95,
    TaskState.CONTINUING.value: 90,
    TaskState.REVIEWING.value: 85,
    TaskState.FINAL_VALIDATION.value: 80,
    TaskState.BLOCKED.value: 75,
    TaskState.WAITING_FOR_USER.value: 70,
    TaskState.DISPATCHED.value: 60,
    TaskState.PLANNED.value: 55,
    TaskState.REWORK_REQUIRED.value: 50,
    TaskState.CREATED.value: 20,
}

#: Fallback progress for a task with no children to measure against.
_STATE_PROGRESS: dict[str, int] = {
    TaskState.CREATED.value: 0,
    TaskState.ANALYZING.value: 10,
    TaskState.PLANNED.value: 20,
    TaskState.DISPATCHED.value: 30,
    TaskState.REWORK_REQUIRED.value: 35,
    TaskState.BLOCKED.value: 40,
    TaskState.IN_PROGRESS.value: 50,
    TaskState.CONTINUING.value: 60,
    TaskState.WAITING_FOR_USER.value: 75,
    TaskState.REVIEWING.value: 80,
    TaskState.FINAL_VALIDATION.value: 90,
    TaskState.COMPLETED.value: 100,
    TaskState.FAILED.value: 0,
}

#: States that mean a human or an operator has to intervene. The schema has no
#: deadline column, so the UI reports this instead of a fabricated "overdue".
_ATTENTION_STATES = (
    TaskState.BLOCKED.value,
    TaskState.FAILED.value,
    TaskState.REWORK_REQUIRED.value,
)

#: How long a just-finished agent keeps showing COMPLETED before going IDLE.
_RECENT_COMPLETION_WINDOW = timedelta(minutes=10)


def _iso(value: Any) -> str | None:
    """Return an ISO timestamp for a datetime, or ``None``."""
    return value.isoformat() if isinstance(value, datetime) else None


def _loads(value: Any, default: Any) -> Any:
    """Decode a JSON column that some drivers hand back as ``str``."""
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return default
    return value


def _pct(part: int, whole: int) -> int:
    """Return ``part/whole`` as a rounded percentage, 0 when undefined."""
    return round(100 * part / whole) if whole else 0


class OfficeService:
    """Projects live orchestration state into structured progress reports for chat and workflow."""

    def __init__(self, session_factory, vllm=None) -> None:
        self.session_factory = session_factory
        self.vllm = vllm

    # ------------------------------------------------------------------ util
    async def _rows(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a read query and return plain dict rows.

        Sequence-valued parameters are bound as expanding IN-lists, so callers
        can write ``WHERE state IN :states`` and pass a tuple.
        """
        params = params or {}
        stmt = text(sql)
        expanding = [k for k, v in params.items() if isinstance(v, (list, tuple, set))]
        if expanding:
            stmt = stmt.bindparams(*(bindparam(k, expanding=True) for k in expanding))
            params = {k: list(v) if k in expanding else v for k, v in params.items()}
        async with self.session_factory() as session:
            result = await session.execute(stmt, params)
            return [dict(r._mapping) for r in result.fetchall()]

    # -------------------------------------------------------------- overview
    async def overview(self, activity_limit: int = 40) -> dict[str, Any]:
        """Return everything the Executive Dashboard renders in one snapshot."""
        agents = await self._agent_roster()
        counts = await self._task_counts()
        focus = await self._focus_root()
        return {
            "generated_at": datetime.now().isoformat(),
            "system": await self._system_health(),
            "kpis": counts,
            "flow": await self._flow(focus, agents),
            "agents": agents,
            "departments": await self._departments(agents),
            "activity": await self._activity(activity_limit),
            "analytics": await self._analytics(counts),
            "focus_root_id": focus["task_id"] if focus else None,
        }

    async def _system_health(self) -> dict[str, Any]:
        """Report which backing services actually answered, not a fixed banner."""
        llm = embedding = None
        if self.vllm is not None:
            try:
                llm = await self.vllm.is_llm_available()
                embedding = await self.vllm.is_embedding_available()
            except Exception as exc:  # health probe must never break the page
                logger.warning("office.health_probe_failed", error=str(exc))
                llm = embedding = False
        checks = [c for c in (llm, embedding) if c is not None]
        operational = all(checks) if checks else True
        return {
            "database": True,
            "llm": llm,
            "embedding": embedding,
            "status": "operational" if operational else "degraded",
        }

    async def _task_counts(self) -> dict[str, int]:
        """Aggregate task counts by lifecycle state across every workflow."""
        rows = await self._rows(
            "SELECT state, task_type, COUNT(*) AS n FROM agent_tasks GROUP BY state, task_type"
        )
        by_state: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for r in rows:
            by_state[r["state"]] = by_state.get(r["state"], 0) + r["n"]
            by_type[r["task_type"]] = by_type.get(r["task_type"], 0) + r["n"]

        active_states = (
            TaskState.ANALYZING.value,
            TaskState.PLANNED.value,
            TaskState.DISPATCHED.value,
            TaskState.IN_PROGRESS.value,
            TaskState.CONTINUING.value,
            TaskState.REVIEWING.value,
            TaskState.FINAL_VALIDATION.value,
        )
        return {
            "total": sum(by_state.values()),
            "in_progress": sum(by_state.get(s, 0) for s in active_states),
            "completed": by_state.get(TaskState.COMPLETED.value, 0),
            "blocked": by_state.get(TaskState.BLOCKED.value, 0),
            "failed": by_state.get(TaskState.FAILED.value, 0),
            "waiting_user": by_state.get(TaskState.WAITING_FOR_USER.value, 0),
            "pending": by_state.get(TaskState.CREATED.value, 0),
            "rework": by_state.get(TaskState.REWORK_REQUIRED.value, 0),
            "root_tasks": by_type.get("root", 0),
            "department_tasks": by_type.get("department", 0),
            "subtasks": by_type.get("subtask", 0),
        }

    async def _agent_roster(self) -> list[dict[str, Any]]:
        """Return every agent with a runtime status derived from its real work."""
        agents = await self._rows(
            """SELECT agent_id, name, role, department, level, avatar, color,
                      specialization, status, parent_agent
               FROM agents ORDER BY name"""
        )
        live = await self._rows(
            """SELECT task_id, task_type, state, department, assigned_to, objective,
                      task_description, root_task_id, created_at, started_at
               FROM agent_tasks
               WHERE assigned_to IS NOT NULL AND state NOT IN :terminal
               ORDER BY created_at DESC""",
            {"terminal": _TERMINAL},
        )
        finished = await self._rows(
            """SELECT assigned_to, MAX(completed_at) AS last_completed
               FROM agent_tasks
               WHERE assigned_to IS NOT NULL AND state = :done AND completed_at IS NOT NULL
               GROUP BY assigned_to""",
            {"done": TaskState.COMPLETED.value},
        )
        last_done = {r["assigned_to"]: r["last_completed"] for r in finished}

        current: dict[str, dict[str, Any]] = {}
        for task in live:
            holder = task["assigned_to"]
            best = current.get(holder)
            salience = _STATE_SALIENCE.get(task["state"], 0)
            if best is None or salience > _STATE_SALIENCE.get(best["state"], 0):
                current[holder] = task

        progress = await self._progress_map([t["task_id"] for t in current.values()])
        now = datetime.now()
        roster = []
        for agent in agents:
            task = current.get(agent["agent_id"])
            done_at = last_done.get(agent["agent_id"])
            if task is not None:
                status = _STATE_TO_AGENT_STATUS.get(task["state"], AgentStatus.WORKING)
            elif isinstance(done_at, datetime) and now - done_at <= _RECENT_COMPLETION_WINDOW:
                status = AgentStatus.COMPLETED
            else:
                status = AgentStatus.IDLE
            roster.append(
                {
                    "agent_id": agent["agent_id"],
                    "name": agent["name"],
                    "role": agent["role"],
                    "department": agent["department"],
                    "level": agent["level"] or _level_from_role(agent["role"]),
                    "avatar": agent["avatar"],
                    "color": agent["color"],
                    "specialization": agent["specialization"],
                    "parent_agent": agent["parent_agent"],
                    "runtime_status": status.value,
                    "current_task": task["task_id"] if task else None,
                    "current_task_type": task["task_type"] if task else None,
                    "current_task_state": task["state"] if task else None,
                    "current_objective": (
                        task["objective"] or task["task_description"] if task else None
                    ),
                    "current_root_id": task["root_task_id"] if task else None,
                    "started_at": _iso(task["started_at"]) if task else None,
                    "progress": progress.get(task["task_id"], 0) if task else 0,
                    "last_completed_at": _iso(done_at),
                }
            )
        return roster

    async def _progress_map(self, task_ids: list[str]) -> dict[str, int]:
        """Return completion percentage per task, measured from its children."""
        if not task_ids:
            return {}
        rows = await self._rows(
            """SELECT t.task_id, t.state,
                      COUNT(c.task_id) AS children,
                      COUNT(c.task_id) FILTER (WHERE c.state = :done) AS done
               FROM agent_tasks t
               LEFT JOIN agent_tasks c ON c.parent_task_id = t.task_id
               WHERE t.task_id IN :ids
               GROUP BY t.task_id, t.state""",
            {"ids": tuple(task_ids), "done": TaskState.COMPLETED.value},
        )
        return {r["task_id"]: _progress_of(r["state"], r["children"], r["done"]) for r in rows}

    async def _focus_root(self) -> dict[str, Any] | None:
        """Return the root task the dashboard should feature.

        Prefers the newest workflow that is still live, so the flow panel shows
        work in motion; falls back to the newest finished one.
        """
        live = await self._rows(
            """SELECT * FROM agent_tasks
               WHERE task_type = 'root' AND state NOT IN :terminal
               ORDER BY created_at DESC LIMIT 1""",
            {"terminal": _TERMINAL},
        )
        if live:
            return live[0]
        recent = await self._rows(
            "SELECT * FROM agent_tasks WHERE task_type = 'root' ORDER BY created_at DESC LIMIT 1"
        )
        return recent[0] if recent else None

    async def _flow(
        self, root: dict[str, Any] | None, agents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build the USER → BRAIND → LEADERS → SPECIALISTS → REPORTS pipeline."""
        by_id = {a["agent_id"]: a for a in agents}
        ceo = next((a for a in agents if a["level"] == "executive"), None)
        if root is None:
            return {
                "root": None,
                "braind": _flow_agent(ceo),
                "leaders": [],
                "specialists": {"active": 0, "total": _count_level(agents, "specialist")},
                "reports": {"submitted": 0, "expected": 0},
                "review": {"state": None, "decision": None},
            }

        root_id = root["task_id"]
        tree = await self._rows(
            """SELECT task_id, task_type, state, department, assigned_to, objective,
                      task_description, parent_task_id, created_at, started_at, completed_at
               FROM agent_tasks WHERE root_task_id = :rid OR task_id = :rid
               ORDER BY created_at""",
            {"rid": root_id},
        )
        dept_tasks = [t for t in tree if t["task_type"] == "department"]
        subtasks = [t for t in tree if t["task_type"] == "subtask"]
        progress = await self._progress_map([t["task_id"] for t in dept_tasks])

        reports = await self._rows(
            """SELECT report_type, COUNT(*) AS n FROM agent_reports
               WHERE root_task_id = :rid GROUP BY report_type""",
            {"rid": root_id},
        )
        submitted = next((r["n"] for r in reports if r["report_type"] == "department"), 0)

        leaders = [
            {
                "task_id": t["task_id"],
                "department": t["department"],
                "state": t["state"],
                "objective": t["objective"] or t["task_description"],
                "progress": progress.get(t["task_id"], 0),
                "agent": _flow_agent(by_id.get(t["assigned_to"])),
            }
            for t in dept_tasks
        ]
        active_subtasks = [t for t in subtasks if t["state"] not in _TERMINAL]
        return {
            "root": {
                "task_id": root_id,
                "user_request": root.get("user_request"),
                "state": root["state"],
                "priority": root.get("priority"),
                "created_at": _iso(root.get("created_at")),
            },
            "braind": _flow_agent(by_id.get(root.get("assigned_to")) or ceo),
            "leaders": leaders,
            "specialists": {
                "active": len({t["assigned_to"] for t in active_subtasks if t["assigned_to"]}),
                "total": _count_level(agents, "specialist"),
                "working_on": len(active_subtasks),
            },
            "reports": {"submitted": submitted, "expected": len(dept_tasks)},
            "review": {
                "state": root["state"],
                "in_review": root["state"]
                in (TaskState.REVIEWING.value, TaskState.FINAL_VALIDATION.value),
            },
        }

    async def _departments(self, agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Roll task counts and agent activity up to the department level."""
        rows = await self._rows(
            """SELECT department, state, COUNT(*) AS n FROM agent_tasks
               WHERE department IS NOT NULL GROUP BY department, state"""
        )
        totals: dict[str, dict[str, int]] = {}
        for r in rows:
            bucket = totals.setdefault(r["department"], {})
            bucket[r["state"]] = bucket.get(r["state"], 0) + r["n"]

        departments = sorted(
            {a["department"] for a in agents if a["department"] and a["department"] != "executive"}
            | set(totals)
        )
        out = []
        for dept in departments:
            states = totals.get(dept, {})
            members = [a for a in agents if a["department"] == dept]
            total = sum(states.values())
            completed = states.get(TaskState.COMPLETED.value, 0)
            leader = next((a for a in members if a["level"] == "leader"), None)
            out.append(
                {
                    "department": dept,
                    "leader": leader["name"] if leader else None,
                    "leader_id": leader["agent_id"] if leader else None,
                    "agents_total": len(members),
                    "agents_active": sum(
                        1 for a in members if a["runtime_status"] != AgentStatus.IDLE.value
                    ),
                    "tasks_total": total,
                    "completed": completed,
                    "in_progress": sum(
                        n for s, n in states.items() if s not in _TERMINAL and s != "created"
                    ),
                    "blocked": sum(states.get(s, 0) for s in _ATTENTION_STATES),
                    "progress_pct": _pct(completed, total),
                }
            )
        return out

    async def _activity(self, limit: int) -> list[dict[str, Any]]:
        """Return the newest audit events, enriched with task and agent names."""
        events = await self._rows(
            """SELECT event_id, action, actor, task_id, root_task_id, previous_status,
                      new_status, reason, created_at
               FROM agent_audit_events ORDER BY created_at DESC LIMIT :limit""",
            {"limit": limit},
        )
        task_ids = tuple({e["task_id"] for e in events if e["task_id"]})
        titles: dict[str, dict[str, Any]] = {}
        if task_ids:
            for r in await self._rows(
                """SELECT task_id, task_type, department, objective, task_description
                   FROM agent_tasks WHERE task_id IN :ids""",
                {"ids": task_ids},
            ):
                titles[r["task_id"]] = r

        actor_ids = tuple({e["actor"] for e in events if e["actor"]})
        names: dict[str, str] = {}
        if actor_ids:
            for r in await self._rows(
                "SELECT agent_id, name FROM agents WHERE agent_id IN :ids", {"ids": actor_ids}
            ):
                names[r["agent_id"]] = r["name"]

        out = []
        for e in events:
            task = titles.get(e["task_id"] or "", {})
            out.append(
                {
                    "event_id": e["event_id"],
                    "created_at": _iso(e["created_at"]),
                    "action": e["action"],
                    "actor": e["actor"],
                    "actor_name": names.get(e["actor"] or "", e["actor"]),
                    "task_id": e["task_id"],
                    "root_task_id": e["root_task_id"],
                    "task_type": task.get("task_type"),
                    "department": task.get("department"),
                    "task_title": task.get("objective") or task.get("task_description"),
                    "previous_status": e["previous_status"],
                    "new_status": e["new_status"],
                    "reason": e["reason"],
                }
            )
        return out

    async def _analytics(self, counts: dict[str, int]) -> dict[str, Any]:
        """Compute the lower dashboard's rates, distributions and 7-day trend."""
        finished = counts["completed"] + counts["failed"]
        by_dept = await self._rows(
            """SELECT department,
                      COUNT(*) AS total,
                      COUNT(*) FILTER (WHERE state = :done) AS completed
               FROM agent_tasks WHERE department IS NOT NULL
               GROUP BY department ORDER BY total DESC""",
            {"done": TaskState.COMPLETED.value},
        )
        dept_total = sum(r["total"] for r in by_dept)

        by_state = await self._rows(
            "SELECT state, COUNT(*) AS n FROM agent_tasks GROUP BY state ORDER BY n DESC"
        )
        trend = await self._rows(
            """SELECT d::date AS day,
                      (SELECT COUNT(*) FROM agent_tasks
                        WHERE created_at >= d AND created_at < d + INTERVAL '1 day') AS created,
                      (SELECT COUNT(*) FROM agent_tasks
                        WHERE completed_at >= d AND completed_at < d + INTERVAL '1 day')
                        AS completed
               FROM generate_series(
                    date_trunc('day', NOW()) - INTERVAL '6 days',
                    date_trunc('day', NOW()), INTERVAL '1 day') AS d
               ORDER BY day"""
        )
        duration = await self._rows(
            """SELECT AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) AS seconds
               FROM agent_tasks
               WHERE state = :done AND started_at IS NOT NULL AND completed_at IS NOT NULL""",
            {"done": TaskState.COMPLETED.value},
        )
        avg_seconds = duration[0]["seconds"] if duration and duration[0]["seconds"] else None

        roots = await self._rows(
            """SELECT COUNT(*) AS total,
                      COUNT(*) FILTER (WHERE rework_count = 0) AS clean
               FROM agent_tasks WHERE task_type = 'root' AND state = :done""",
            {"done": TaskState.COMPLETED.value},
        )
        root_done = roots[0]["total"] if roots else 0
        root_clean = roots[0]["clean"] if roots else 0

        agents_total = (await self._rows("SELECT COUNT(*) AS n FROM agents"))[0]["n"]
        engaged = (
            await self._rows(
                """SELECT COUNT(DISTINCT assigned_to) AS n FROM agent_tasks
                   WHERE assigned_to IS NOT NULL AND state NOT IN :terminal""",
                {"terminal": _TERMINAL},
            )
        )[0]["n"]

        return {
            "completion_rate": _pct(counts["completed"], finished),
            "agent_utilization": _pct(engaged, agents_total),
            "first_pass_rate": _pct(root_clean, root_done),
            "avg_completion_seconds": round(float(avg_seconds), 1) if avg_seconds else None,
            "tasks_by_department": [
                {
                    "department": r["department"],
                    "total": r["total"],
                    "completed": r["completed"],
                    "share_pct": _pct(r["total"], dept_total),
                }
                for r in by_dept
            ],
            "status_distribution": [
                {"state": r["state"], "count": r["n"], "share_pct": _pct(r["n"], counts["total"])}
                for r in by_state
            ],
            "performance_trend": [
                {
                    "day": (
                        r["day"].isoformat() if hasattr(r["day"], "isoformat") else str(r["day"])
                    ),
                    "created": r["created"],
                    "completed": r["completed"],
                }
                for r in trend
            ],
        }

    # ------------------------------------------------------------ task board
    async def task_board(self, root_task_id: str | None = None) -> dict[str, Any]:
        """Return the Task Command Center: summary cards plus the task forest."""
        summary = await self._board_summary()
        roots = await self._rows(
            """SELECT task_id, user_request, state, priority, created_at, completed_at,
                      confirmation_required
               FROM agent_tasks WHERE task_type = 'root'
               ORDER BY created_at DESC LIMIT 50"""
        )
        selected = root_task_id or (roots[0]["task_id"] if roots else None)
        tree: list[dict[str, Any]] = []
        if selected:
            tree = await self._tree(selected)
        return {
            "summary": summary,
            "roots": [
                {
                    "task_id": r["task_id"],
                    "user_request": r["user_request"],
                    "state": r["state"],
                    "priority": r["priority"],
                    "created_at": _iso(r["created_at"]),
                    "completed_at": _iso(r["completed_at"]),
                    "confirmation_required": r["confirmation_required"],
                }
                for r in roots
            ],
            "selected_root_id": selected,
            "tree": tree,
        }

    async def _board_summary(self) -> dict[str, Any]:
        """Summary cards for the command center, from real columns only."""
        counts = await self._task_counts()
        today = await self._rows(
            """SELECT COUNT(*) AS n FROM agent_tasks
               WHERE state = :done AND completed_at >= date_trunc('day', NOW())""",
            {"done": TaskState.COMPLETED.value},
        )
        attention = await self._rows(
            "SELECT COUNT(*) AS n FROM agent_tasks WHERE state IN :states",
            {"states": _ATTENTION_STATES},
        )
        duration = await self._rows(
            """SELECT AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) AS seconds
               FROM agent_tasks
               WHERE state = :done AND started_at IS NOT NULL AND completed_at IS NOT NULL""",
            {"done": TaskState.COMPLETED.value},
        )
        seconds = duration[0]["seconds"] if duration and duration[0]["seconds"] else None
        return {
            "root_tasks": counts["root_tasks"],
            "department_tasks": counts["department_tasks"],
            "subtasks": counts["subtasks"],
            "completed_today": today[0]["n"] if today else 0,
            "needs_attention": attention[0]["n"] if attention else 0,
            "avg_completion_seconds": round(float(seconds), 1) if seconds else None,
        }

    async def _tree(self, root_task_id: str) -> list[dict[str, Any]]:
        """Return every task under a root, each annotated with real progress."""
        rows = await self._rows(
            """SELECT t.task_id, t.parent_task_id, t.task_type, t.state, t.department,
                      t.assigned_to, t.assigned_by, t.objective, t.task_description,
                      t.user_request, t.priority, t.rework_count, t.created_at,
                      t.started_at, t.completed_at,
                      COUNT(c.task_id) AS children,
                      COUNT(c.task_id) FILTER (WHERE c.state = :done) AS children_done
               FROM agent_tasks t
               LEFT JOIN agent_tasks c ON c.parent_task_id = t.task_id
               WHERE t.root_task_id = :rid OR t.task_id = :rid
               GROUP BY t.task_id
               ORDER BY t.created_at""",
            {"rid": root_task_id, "done": TaskState.COMPLETED.value},
        )
        names = {
            r["agent_id"]: r["name"] for r in await self._rows("SELECT agent_id, name FROM agents")
        }
        return [
            {
                "task_id": r["task_id"],
                "parent_task_id": r["parent_task_id"],
                "task_type": r["task_type"],
                "state": r["state"],
                "department": r["department"],
                "assigned_to": r["assigned_to"],
                "assigned_name": names.get(r["assigned_to"] or "", r["assigned_to"]),
                "title": r["objective"] or r["task_description"] or r["user_request"],
                "priority": r["priority"],
                "rework_count": r["rework_count"],
                "progress": _progress_of(r["state"], r["children"], r["children_done"]),
                "children": r["children"],
                "created_at": _iso(r["created_at"]),
                "started_at": _iso(r["started_at"]),
                "completed_at": _iso(r["completed_at"]),
            }
            for r in rows
        ]

    # ----------------------------------------------------------- task detail
    async def task_detail(self, task_id: str) -> dict[str, Any] | None:
        """Return one task with its timeline, reports and acceptance criteria."""
        rows = await self._rows(
            "SELECT * FROM agent_tasks WHERE task_id = :tid", {"tid": task_id}
        )
        if not rows:
            return None
        task = rows[0]
        root_id = task["root_task_id"] or task["task_id"]

        children = await self._rows(
            """SELECT task_id, state, department, assigned_to, objective, task_description
               FROM agent_tasks WHERE parent_task_id = :tid ORDER BY created_at""",
            {"tid": task_id},
        )
        done = sum(1 for c in children if c["state"] == TaskState.COMPLETED.value)

        timeline = await self._rows(
            """SELECT action, actor, previous_status, new_status, reason, created_at
               FROM agent_audit_events
               WHERE task_id = :tid OR parent_task_id = :tid
               ORDER BY created_at""",
            {"tid": task_id},
        )
        reports = await self._rows(
            """SELECT report_id, task_id, agent_id, report_type, status, confidence,
                      payload, created_at
               FROM agent_reports
               WHERE task_id = :tid OR department_task_id = :tid
               ORDER BY created_at""",
            {"tid": task_id},
        )
        names = {
            r["agent_id"]: r["name"] for r in await self._rows("SELECT agent_id, name FROM agents")
        }
        return {
            "task": {
                "task_id": task["task_id"],
                "root_task_id": root_id,
                "parent_task_id": task["parent_task_id"],
                "task_type": task["task_type"],
                "state": task["state"],
                "department": task["department"],
                "assigned_to": task["assigned_to"],
                "assigned_name": names.get(task["assigned_to"] or "", task["assigned_to"]),
                "assigned_by": task["assigned_by"],
                "assigned_by_name": names.get(task["assigned_by"] or "", task["assigned_by"]),
                "title": task["objective"] or task["task_description"] or task["user_request"],
                "description": task["task_description"],
                "instructions": task["instructions"],
                "expected_output": task["expected_output"],
                "user_request": task["user_request"],
                "priority": task["priority"],
                "rework_count": task["rework_count"],
                "confirmation_required": task["confirmation_required"],
                "acceptance_criteria": _loads(task["acceptance_criteria"], []),
                "constraints": _loads(task["constraints"], []),
                "result": _loads(task["result"], None),
                "error": task["error"],
                "progress": _progress_of(task["state"], len(children), done),
                "created_at": _iso(task["created_at"]),
                "started_at": _iso(task["started_at"]),
                "completed_at": _iso(task["completed_at"]),
            },
            "children": [
                {
                    "task_id": c["task_id"],
                    "state": c["state"],
                    "department": c["department"],
                    "assigned_to": c["assigned_to"],
                    "assigned_name": names.get(c["assigned_to"] or "", c["assigned_to"]),
                    "title": c["objective"] or c["task_description"],
                }
                for c in children
            ],
            "timeline": [
                {
                    "action": e["action"],
                    "actor": e["actor"],
                    "actor_name": names.get(e["actor"] or "", e["actor"]),
                    "previous_status": e["previous_status"],
                    "new_status": e["new_status"],
                    "reason": e["reason"],
                    "created_at": _iso(e["created_at"]),
                }
                for e in timeline
            ],
            "reports": [
                {
                    "report_id": r["report_id"],
                    "task_id": r["task_id"],
                    "agent_id": r["agent_id"],
                    "agent_name": names.get(r["agent_id"] or "", r["agent_id"]),
                    "report_type": r["report_type"],
                    "status": r["status"],
                    "confidence": r["confidence"],
                    "payload": _loads(r["payload"], {}),
                    "created_at": _iso(r["created_at"]),
                }
                for r in reports
            ],
        }

    # ---------------------------------------------------------- agent detail
    async def agent_detail(self, agent_id: str) -> dict[str, Any] | None:
        """Return one agent's live assignment, history and recent activity."""
        roster = await self._agent_roster()
        agent = next((a for a in roster if a["agent_id"] == agent_id), None)
        if agent is None:
            return None

        tasks = await self._rows(
            """SELECT task_id, task_type, state, department, objective, task_description,
                      created_at, started_at, completed_at
               FROM agent_tasks WHERE assigned_to = :aid
               ORDER BY created_at DESC LIMIT 25""",
            {"aid": agent_id},
        )
        reports = await self._rows(
            """SELECT report_id, task_id, report_type, status, confidence, created_at
               FROM agent_reports WHERE agent_id = :aid ORDER BY created_at DESC LIMIT 15""",
            {"aid": agent_id},
        )
        events = await self._rows(
            """SELECT action, task_id, previous_status, new_status, reason, created_at
               FROM agent_audit_events WHERE actor = :aid ORDER BY created_at DESC LIMIT 25""",
            {"aid": agent_id},
        )
        completed = sum(1 for t in tasks if t["state"] == TaskState.COMPLETED.value)
        return {
            "agent": agent,
            "stats": {
                "tasks_total": len(tasks),
                "tasks_completed": completed,
                "reports_submitted": len(reports),
                "completion_rate": _pct(completed, len(tasks)),
            },
            "tasks": [
                {
                    "task_id": t["task_id"],
                    "task_type": t["task_type"],
                    "state": t["state"],
                    "department": t["department"],
                    "title": t["objective"] or t["task_description"],
                    "created_at": _iso(t["created_at"]),
                    "completed_at": _iso(t["completed_at"]),
                }
                for t in tasks
            ],
            "reports": [
                {
                    "report_id": r["report_id"],
                    "task_id": r["task_id"],
                    "report_type": r["report_type"],
                    "status": r["status"],
                    "confidence": r["confidence"],
                    "created_at": _iso(r["created_at"]),
                }
                for r in reports
            ],
            "activity": [
                {
                    "action": e["action"],
                    "task_id": e["task_id"],
                    "previous_status": e["previous_status"],
                    "new_status": e["new_status"],
                    "reason": e["reason"],
                    "created_at": _iso(e["created_at"]),
                }
                for e in events
            ],
        }


def _progress_of(state: str, children: int, children_done: int) -> int:
    """Return a task's completion percentage.

    Children are the honest measure when a task has them; otherwise the
    lifecycle state is the only signal the schema carries.
    """
    if state == TaskState.COMPLETED.value:
        return 100
    if state == TaskState.FAILED.value:
        return 0
    if children:
        return _pct(children_done, children)
    return _STATE_PROGRESS.get(state, 0)


def _level_from_role(role: str | None) -> str:
    """Derive a hierarchy level for rows seeded before the ``level`` column."""
    return {"CEO": "executive", "LEADER": "leader"}.get((role or "").upper(), "specialist")


def _count_level(agents: list[dict[str, Any]], level: str) -> int:
    """Count roster members at one hierarchy level."""
    return sum(1 for a in agents if a["level"] == level)


def _flow_agent(agent: dict[str, Any] | None) -> dict[str, Any] | None:
    """Reduce a roster entry to the fields the flow cards render."""
    if agent is None:
        return None
    return {
        "agent_id": agent["agent_id"],
        "name": agent["name"],
        "role": agent["role"],
        "department": agent["department"],
        "level": agent["level"],
        "avatar": agent["avatar"],
        "runtime_status": agent["runtime_status"],
        "current_task": agent["current_task"],
        "progress": agent["progress"],
    }
