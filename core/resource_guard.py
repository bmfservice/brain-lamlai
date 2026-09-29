"""Runtime VRAM and resource guard (CLAUDE.md §3, §9).

Monitors GPU VRAM and utilization across dual RTX 3090s via nvidia-smi.
Enforces the mandatory >= 5GB free VRAM reserve per GPU constraint at runtime.
Automatically throttles embedding batch size, concurrency, or initiates offload
when available memory falls below the configured threshold.
Persists periodic telemetry samples to `gpu_resource_samples`.
Degrades gracefully with logged warnings if nvidia-smi is unavailable (no crash).
"""

from __future__ import annotations

import asyncio
import datetime
import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core import database as db
from core.logging import get_logger
from services.settings.resolver import SettingsResolver
from services.settings.store import SettingsStore

logger = get_logger(__name__)

# Search candidates for nvidia-smi across WSL, Windows, and standard Linux
_NVIDIA_SMI_CANDIDATES: tuple[str, ...] = (
    "nvidia-smi",
    "/usr/lib/wsl/lib/nvidia-smi",
    "/usr/bin/nvidia-smi",
    "/bin/nvidia-smi",
    "nvidia-smi.exe",
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)


@dataclass(frozen=True)
class GPUInfo:
    """Snapshot of a single GPU's runtime metrics."""

    gpu_index: int
    total_vram_bytes: int
    used_vram_bytes: int
    free_vram_bytes: int
    utilization_percent: float | None = None
    temperature_celsius: float | None = None
    is_degraded: bool = False

    @property
    def total_vram_mib(self) -> float:
        return self.total_vram_bytes / (1024 * 1024)

    @property
    def used_vram_mib(self) -> float:
        return self.used_vram_bytes / (1024 * 1024)

    @property
    def free_vram_mib(self) -> float:
        return self.free_vram_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_index": self.gpu_index,
            "total_vram_bytes": self.total_vram_bytes,
            "used_vram_bytes": self.used_vram_bytes,
            "free_vram_bytes": self.free_vram_bytes,
            "total_vram_mib": round(self.total_vram_mib, 2),
            "used_vram_mib": round(self.used_vram_mib, 2),
            "free_vram_mib": round(self.free_vram_mib, 2),
            "utilization_percent": self.utilization_percent,
            "temperature_celsius": self.temperature_celsius,
            "is_degraded": self.is_degraded,
        }


@dataclass(frozen=True)
class GuardAdjustment:
    """Current throttling state computed by ResourceGuard."""

    degraded: bool
    reason: str | None
    min_free_vram_mib: float
    reserve_threshold_mib: int
    configured_batch_size: int
    effective_batch_size: int
    configured_concurrency: int
    effective_concurrency: int
    gpus: list[GPUInfo]

    def to_dict(self) -> dict[str, Any]:
        return {
            "degraded": self.degraded,
            "reason": self.reason,
            "min_free_vram_mib": round(self.min_free_vram_mib, 2),
            "reserve_threshold_mib": self.reserve_threshold_mib,
            "configured_batch_size": self.configured_batch_size,
            "effective_batch_size": self.effective_batch_size,
            "configured_concurrency": self.configured_concurrency,
            "effective_concurrency": self.effective_concurrency,
            "gpus": [g.to_dict() for g in self.gpus],
        }


class ResourceGuard:
    """Runtime VRAM & hardware resource guard."""

    def __init__(
        self,
        session_factory=None,
        *,
        settings_resolver: SettingsResolver | None = None,
        settings_store: SettingsStore | None = None,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.session_factory = session_factory
        self.settings_store = settings_store or (
            SettingsStore(session_factory) if session_factory else None
        )
        self.settings_resolver = settings_resolver or (
            SettingsResolver(session_factory, self.settings_store)
            if session_factory
            else None
        )
        self.timeout_seconds = timeout_seconds
        self._cached_gpu_info: list[GPUInfo] | None = None
        self._cache_timestamp: float = 0.0
        self._smi_executable: str | None = None

    def _locate_nvidia_smi(self) -> str | None:
        """Find a working nvidia-smi binary on the system."""
        if self._smi_executable is not None:
            return self._smi_executable

        for candidate in _NVIDIA_SMI_CANDIDATES:
            if shutil.which(candidate) or os.path.exists(candidate):
                self._smi_executable = candidate
                return candidate

        return None

    async def _get_setting_int(self, key: str, fallback_hard: int) -> int:
        """Resolve an integer setting dynamically through the settings hierarchy."""
        if self.settings_resolver is not None:
            try:
                val, _ = await self.settings_resolver.resolve(key)
                if val is not None:
                    return int(val)
            except Exception as exc:
                logger.debug("resource_guard.resolve_setting_error", key=key, error=str(exc))
        return fallback_hard

    async def _execute_nvidia_smi(self) -> str:
        """Run nvidia-smi query and return raw CSV output."""
        binary = self._locate_nvidia_smi()
        if not binary:
            raise FileNotFoundError("nvidia-smi executable not found on system PATH or standard locations")

        cmd = [
            binary,
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError as err:
            try:
                proc.kill()
            except Exception:
                pass
            raise TimeoutError(f"nvidia-smi timed out after {self.timeout_seconds}s") from err

        if proc.returncode != 0:
            err_msg = stderr_bytes.decode(errors="replace").strip()
            raise RuntimeError(f"nvidia-smi exited with code {proc.returncode}: {err_msg}")

        return stdout_bytes.decode("utf-8", errors="replace").strip()

    def _parse_smi_output(self, output: str) -> list[GPUInfo]:
        """Parse CSV rows from nvidia-smi query output."""
        results: list[GPUInfo] = []
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            try:
                idx = int(parts[0])
                total_mib = int(float(parts[1]))
                used_mib = int(float(parts[2]))
                free_mib = int(float(parts[3]))

                # Convert MiB to Bytes (1 MiB = 1024 * 1024 Bytes)
                total_bytes = total_mib * 1024 * 1024
                used_bytes = used_mib * 1024 * 1024
                free_bytes = free_mib * 1024 * 1024

                utilization: float | None = None
                if len(parts) >= 5 and parts[4] and parts[4] != "[N/A]":
                    try:
                        utilization = float(parts[4])
                    except ValueError:
                        utilization = None

                temp: float | None = None
                if len(parts) >= 6 and parts[5] and parts[5] != "[N/A]":
                    try:
                        temp = float(parts[5])
                    except ValueError:
                        temp = None

                results.append(
                    GPUInfo(
                        gpu_index=idx,
                        total_vram_bytes=total_bytes,
                        used_vram_bytes=used_bytes,
                        free_vram_bytes=free_bytes,
                        utilization_percent=utilization,
                        temperature_celsius=temp,
                        is_degraded=False,
                    )
                )
            except (ValueError, IndexError) as parse_err:
                logger.warning("resource_guard.parse_row_failed", line=line, error=str(parse_err))

        return results

    async def query_gpu_info(self, *, force_refresh: bool = False) -> list[GPUInfo]:
        """Query GPU info with short TTL caching and graceful fallback."""
        now = asyncio.get_running_loop().time()
        interval = await self._get_setting_int("gpu_monitor_interval_seconds", 5)

        if not force_refresh and self._cached_gpu_info is not None and (now - self._cache_timestamp) < interval:
            return self._cached_gpu_info

        try:
            raw_output = await self._execute_nvidia_smi()
            parsed = self._parse_smi_output(raw_output)
            if not parsed:
                raise ValueError("nvidia-smi returned empty GPU table")
            self._cached_gpu_info = parsed
            self._cache_timestamp = now
            return parsed
        except Exception as exc:
            logger.warning(
                "resource_guard.nvidia_smi_unavailable",
                error=str(exc),
                message="nvidia-smi query failed; operating in degraded fallback mode",
            )
            # Graceful degraded fallback (e.g. CPU or unmonitored GPU state, no crash)
            fallback_vram = 24 * 1024 * 1024 * 1024  # 24GB nominal
            degraded_info = [
                GPUInfo(
                    gpu_index=0,
                    total_vram_bytes=fallback_vram,
                    used_vram_bytes=fallback_vram // 2,
                    free_vram_bytes=fallback_vram // 2,
                    utilization_percent=None,
                    temperature_celsius=None,
                    is_degraded=True,
                )
            ]
            self._cached_gpu_info = degraded_info
            self._cache_timestamp = now
            return degraded_info

    async def check_and_adjust_limits(self) -> GuardAdjustment:
        """Check current VRAM headroom against the mandatory reserve threshold and adjust limits.

        If any GPU has free_vram < gpu_reserve_free_mib (default/min 5120 MiB per CLAUDE.md §3),
        automatically throttles batch size and concurrency, logging resource_guard.degraded.
        """
        gpus = await self.query_gpu_info()
        reserve_threshold_mib = await self._get_setting_int("gpu_reserve_free_mib", 5120)
        embed_batch_size = await self._get_setting_int("embed_batch_size", 16)
        max_parallel_tasks = await self._get_setting_int("max_parallel_tasks", 4)

        min_free_mib = min((g.free_vram_mib for g in gpus), default=float("inf"))

        degraded = False
        reason: str | None = None
        effective_batch = embed_batch_size
        effective_concurrency = max_parallel_tasks

        # Check for fallback degraded mode
        if any(g.is_degraded for g in gpus):
            degraded = True
            reason = "nvidia-smi unavailable (running in unmonitored fallback)"
            effective_batch = max(1, embed_batch_size // 2)
            effective_concurrency = max(1, max_parallel_tasks // 2)
            logger.warning(
                "resource_guard.degraded",
                reason=reason,
                effective_batch_size=effective_batch,
                effective_concurrency=effective_concurrency,
            )
        elif min_free_mib < reserve_threshold_mib:
            degraded = True
            shortage_mib = reserve_threshold_mib - min_free_mib
            reason = (
                f"VRAM reserve violated: min free VRAM is {min_free_mib:.1f} MiB, "
                f"below required {reserve_threshold_mib} MiB ({shortage_mib:.1f} MiB deficit)"
            )
            # Automatic throttling: halve the batch size and reduce concurrency
            effective_batch = max(1, embed_batch_size // 2)
            effective_concurrency = max(1, max_parallel_tasks // 2)

            logger.warning(
                "resource_guard.degraded",
                reason=reason,
                min_free_vram_mib=min_free_mib,
                reserve_threshold_mib=reserve_threshold_mib,
                adjusted_batch_size=effective_batch,
                adjusted_concurrency=effective_concurrency,
            )

        return GuardAdjustment(
            degraded=degraded,
            reason=reason,
            min_free_vram_mib=min_free_mib,
            reserve_threshold_mib=reserve_threshold_mib,
            configured_batch_size=embed_batch_size,
            effective_batch_size=effective_batch,
            configured_concurrency=max_parallel_tasks,
            effective_concurrency=effective_concurrency,
            gpus=gpus,
        )

    async def record_sample(self, session_factory=None) -> list[uuid.UUID]:
        """Sample current GPU metrics and persist rows to `gpu_resource_samples`.

        Uses BigInteger byte columns to store exact > 2^31-1 byte counts without overflow.
        """
        factory = session_factory or self.session_factory
        if factory is None:
            return []

        gpus = await self.query_gpu_info(force_refresh=True)
        sample_ids: list[uuid.UUID] = []
        now = datetime.datetime.now(datetime.UTC)

        async with factory() as session:
            for g in gpus:
                sample_id = uuid.uuid4()
                sample_ids.append(sample_id)
                stmt = pg_insert(db.GPUResourceSample).values(
                    id=sample_id,
                    gpu_index=g.gpu_index,
                    total_vram_bytes=g.total_vram_bytes,
                    used_vram_bytes=g.used_vram_bytes,
                    free_vram_bytes=g.free_vram_bytes,
                    utilization_percent=g.utilization_percent,
                    temperature_celsius=g.temperature_celsius,
                    sampled_at=now,
                )
                await session.execute(stmt)
            await session.commit()

        logger.info("resource_guard.sample_recorded", count=len(sample_ids))
        return sample_ids

    async def get_latest_samples(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve recent samples for HUD/dashboard tiles."""
        if not self.session_factory:
            # Fall back to live info
            gpus = await self.query_gpu_info()
            return [g.to_dict() for g in gpus]

        async with self.session_factory() as session:
            stmt = (
                select(db.GPUResourceSample)
                .order_by(db.GPUResourceSample.sampled_at.desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "id": str(r.id),
                    "gpu_index": r.gpu_index,
                    "total_vram_bytes": r.total_vram_bytes,
                    "used_vram_bytes": r.used_vram_bytes,
                    "free_vram_bytes": r.free_vram_bytes,
                    "total_vram_mib": round(r.total_vram_bytes / (1024 * 1024), 2),
                    "used_vram_mib": round(r.used_vram_bytes / (1024 * 1024), 2),
                    "free_vram_mib": round(r.free_vram_bytes / (1024 * 1024), 2),
                    "utilization_percent": r.utilization_percent,
                    "temperature_celsius": r.temperature_celsius,
                    "sampled_at": r.sampled_at.isoformat() if r.sampled_at else None,
                }
                for r in rows
            ]
