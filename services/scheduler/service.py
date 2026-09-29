"""Braind Scheduler Service.

In-process task scheduler replacing n8n. Uses APScheduler (AsyncIOScheduler)
to run recurring jobs that keep the braind fresh:

- RSS fetch:  pull latest articles from configured feeds -> ingest into brain
- Web scrape: fetch + clean a URL -> ingest as a document
- Vault sync: re-scan the Obsidian vault for changes
- Digest:     weekly/daily summary of new knowledge

Jobs are registered in code (simple, transparent) and their state (last run,
config) persisted in the `scheduled_tasks` table for observability.
"""

from __future__ import annotations

from collections.abc import Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# Default content types allowed by the writer
DEFAULT_FEEDS: list[dict] = [
    {"name": "Vietnam+ Tech", "url": "https://www.vietnamplus.vn/technology.rss"},
    {"name": "Mozilla Blog", "url": "https://blog.mozilla.org/feed/"},
    {"name": "Datasette", "url": "https://simonwillison.net/atom/everything/"},
]

DEFAULT_DIGEST_PROMPT = (
    "Tổng hợp các kiến thức, tin tức mới nhất mà hệ thống đã học trong thời gian qua. "
    "Trả về 5 mục nổi bật nhất, mỗi mục 2-3 câu, kèm nguồn."
)


class BraindScheduler:
    """APScheduler wrapper for running Braind background jobs."""

    def __init__(
        self, session_factory, ingest, retrieval, writer, maintenance=None, consolidation=None
    ):
        self.session_factory = session_factory
        self.ingest = ingest
        self.retrieval = retrieval
        self.writer = writer
        self.maintenance = maintenance
        # Giai đoạn 5, Layer 7: sleep-time distillation + weight breathing. Set
        # by the composition root; the cron job is registered only when present.
        self.consolidation = consolidation
        # Set by the composition root once the agent exists (the agent depends
        # on the tool registry, which is built after the scheduler).
        self.agent = None
        self._scheduler = AsyncIOScheduler(timezone=settings.scheduler_timezone)
        self._jobs: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register + start the scheduler."""
        self._register_default_jobs()
        if self._jobs:
            self._scheduler.start()
            logger.info("scheduler.started", jobs=[j["name"] for j in self._jobs])
        else:
            logger.info("scheduler.no_jobs")

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("scheduler.stopped")

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def _register_default_jobs(self) -> None:
        # RSS refresh (default every 60 min)
        if settings.rss_refresh_minutes and settings.rss_refresh_minutes > 0:
            self.add_interval_job(
                name="rss-fetch",
                task_type="rss_fetch",
                func=self._job_fetch_rss,
                minutes=settings.rss_refresh_minutes,
            )

        # Vault sync (default every 10 min, cheap due to hash-skip)
        self.add_interval_job(
            name="vault-sync",
            task_type="vault_sync",
            func=self._job_sync_vault,
            minutes=5,
        )

        # Daily digest at 07:03 (off-peak minute)
        self.add_cron_job(
            name="daily-digest",
            task_type="digest",
            func=self._job_daily_digest,
            hour=settings.digest_hour or 7,
            minute=3,
        )

        # Maintenance window: dedup + contradiction scan + re-embed NULL vectors.
        maintenance_hour, _, maintenance_minute = settings.maintenance_time.partition(":")
        self.add_cron_job(
            name="maintenance",
            task_type="maintenance",
            func=self._job_maintenance,
            hour=int(maintenance_hour or 0),
            minute=int(maintenance_minute or 0),
        )

        # Giai đoạn 5, Layer 7: consolidation runs after maintenance (default
        # 00:30) so it distils the decisions the day produced and re-breathes
        # edge weights. Registered only when the service is wired and enabled.
        if self.consolidation is not None and settings.consolidation_enabled:
            cons_hour, _, cons_minute = settings.consolidation_time.partition(":")
            self.add_cron_job(
                name="consolidation",
                task_type="consolidation",
                func=self._job_consolidation,
                hour=int(cons_hour or 0),
                minute=int(cons_minute or 30),
            )

    async def run_job(self, name: str) -> None:
        """Run one registered job on demand.

        Raises ``KeyError`` when the name is not a registered job, so the API
        layer can turn that into a 404 without reaching into private methods.
        """
        return await self._runnable_jobs()[name]()

    def _runnable_jobs(self) -> dict[str, Callable]:
        return {
            "rss-fetch": self._job_fetch_rss,
            "vault-sync": self._job_sync_vault,
            "daily-digest": self._job_daily_digest,
            "maintenance": self._job_maintenance,
            "consolidation": self._job_consolidation,
        }

    def add_interval_job(
        self,
        name: str,
        task_type: str,
        func: Callable,
        minutes: int = 60,
    ) -> None:
        job = self._scheduler.add_job(
            func,
            trigger=IntervalTrigger(minutes=minutes),
            id=name,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._jobs.append({"name": name, "task_type": task_type, "job_id": job.id})
        logger.info("scheduler.job_added", name=name, task_type=task_type, minutes=minutes)

    def add_cron_job(
        self,
        name: str,
        task_type: str,
        func: Callable,
        hour: int,
        minute: int,
        day_of_week: str | None = None,
    ) -> None:
        job = self._scheduler.add_job(
            func,
            trigger=CronTrigger(hour=hour, minute=minute, day_of_week=day_of_week),
            id=name,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._jobs.append({"name": name, "task_type": task_type, "job_id": job.id})
        logger.info(
            "scheduler.job_added", name=name, task_type=task_type, cron=f"{minute} {hour} * * *"
        )

    def get_jobs(self) -> list[dict]:
        out = []
        for j in self._jobs:
            out.append(
                {
                    "name": j["name"],
                    "task_type": j["task_type"],
                    "next_run": self._next_run(j["job_id"]),
                }
            )
        return out

    def _next_run(self, job_id: str) -> str | None:
        job = self._scheduler.get_job(job_id)
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
        return None

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

    async def _job_fetch_rss(self) -> None:
        """Fetch configured RSS feeds and ingest unread items."""
        from services.scheduler.rss import fetch_feed

        feeds = list(DEFAULT_FEEDS)
        # Merge any feeds stored in config
        extra = getattr(settings, "rss_feeds", None) or []
        if isinstance(extra, list):
            for f in extra:
                if isinstance(f, dict) and f.get("url"):
                    feeds.append(f)
                elif isinstance(f, str):
                    feeds.append({"name": "Custom", "url": f})

        total_new = 0
        for feed_cfg in feeds:
            try:
                items = await fetch_feed(feed_cfg["url"], limit=10)
            except Exception as exc:
                logger.warning("rss.fetch_error", url=feed_cfg["url"], error=str(exc))
                continue

            for item in items:
                try:
                    added = await self.ingest.ingest_text(
                        item.content,
                        title=item.title,
                        source_type="rss",
                        url=item.url,
                        tags=["rss", feed_cfg.get("name", "news").lower()],
                        meta={"published": item.published},
                    )
                    total_new += added.added
                except Exception as exc:
                    logger.error("rss.ingest_error", title=item.title, error=str(exc))
        logger.info("rss.job_done", total_new=total_new, feeds=len(feeds))

    async def _job_sync_vault(self) -> None:
        """Re-sync Obsidian vault."""
        try:
            res = await self.ingest.sync_vault()
            logger.info("vault.job_done", added=res.added, updated=res.updated, skipped=res.skipped)
        except Exception as exc:
            logger.error("vault.job_error", error=str(exc))

    async def _job_daily_digest(self) -> None:
        """Generate a daily digest of freshly-learned knowledge."""
        if self.agent is None:
            logger.info("digest.skipped_no_agent")
            return
        try:
            recent = await self.retrieval.recent_knowledge(limit=20)
            context = (
                "\n\n".join(f"--- {d.title} ---\n{d.content[:500]}" for d in recent)
                or "(Chưa có kiến thức mới nào được học hôm nay.)"
            )

            prompt = (
                "Hôm nay hệ thống Braind đã học được những nội dung sau. Hãy viết bản tóm tắt "
                "ngắn gọn 5 mục nổi bật nhất, mỗi mục 2-3 câu, kèm tiêu đề nguồn.\n\n"
                f"{context}"
            )
            digest = await self.agent.chat(prompt)
            logger.info(
                "digest.generated",
                conversation_id=digest.get("conversation_id"),
                content=digest.get("content", "")[:120],
            )
        except Exception as exc:
            logger.error("digest.error", error=str(exc), exc_info=True)

    async def _job_maintenance(self) -> None:
        """Run the maintenance window: report dedup/contradictions, re-embed gaps."""
        if self.maintenance is None:
            logger.info("maintenance.skipped_no_service")
            return
        try:
            report = await self.maintenance.run()
            reindexed = await self.maintenance.reindex(report.reindex_queue)
            logger.info(
                "maintenance.job_done",
                duplicates=len(report.duplicates),
                contradictions=len(report.contradictions),
                reindex_queued=len(report.reindex_queue),
                reindexed=reindexed,
            )
        except Exception as exc:
            logger.error("maintenance.error", error=str(exc), exc_info=True)

    async def _job_consolidation(self) -> None:
        """Giai đoạn 5, Layer 7: replay the day's decisions into the distilled
        classifier, re-breathe edge weights, quarantine contradictions."""
        if self.consolidation is None:
            logger.info("consolidation.skipped_no_service")
            return
        try:
            report = await self.consolidation.run()
            logger.info(
                "consolidation.job_complete",
                retrained=report.retrained,
                new_samples=report.new_samples,
                model_version=report.model_version,
                edges_reweighted=report.edges_reweighted,
                flagged_review=report.flagged_review,
                quarantined=report.quarantined,
                failed=report.steps_failed,
            )
        except Exception as exc:
            logger.error("consolidation.error", error=str(exc), exc_info=True)
