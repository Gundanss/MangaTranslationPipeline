from __future__ import annotations

import asyncio
import json
import os
import signal
import traceback
from pathlib import Path
from typing import Any

from .db import Database
from .engine import CoreEngine, rerender, trim_runtime_memory
from .providers import OllamaProvider, create_provider
from .secret_store import SecretStore


class TaskManager:
    def __init__(self, database: Database, secrets: SecretStore):
        self.database = database
        self.secrets = secrets
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        self.model_lock = asyncio.Lock()
        self.image_locks: dict[str, asyncio.Lock] = {}
        self.shutdown_requested = False
        self.stop_scheduled = False
        self.active_task_id: str | None = None
        self.active_image_id: str | None = None
        self.active_manual_edits = 0

    def start(self) -> None:
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())

    def _build_engine(
        self,
        config: dict[str, Any],
        progress_callback,
        log_callback,
    ) -> CoreEngine:
        settings = self.secrets.get()
        provider = create_provider(config["provider"], settings, config.get("ollama_model"))
        polish_provider = None
        if config.get("polish_with_ollama"):
            polish_model = config.get("polish_model") or config.get("ollama_model")
            if not polish_model:
                raise RuntimeError("开启 Ollama 润色时必须选择润色模型")
            polish_provider = OllamaProvider(settings["ollama_base_url"], polish_model)
        return CoreEngine(
            provider=provider,
            source_language=config["source_language"],
            target_language=config["target_language"],
            polish_provider=polish_provider,
            render_direction=config.get("render_direction", "auto"),
            render_alignment=config.get("render_alignment", "auto"),
            font_size=config.get("font_size"),
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

    async def enqueue(self, task_id: str) -> None:
        await self.queue.put(task_id)

    @property
    def is_shutting_down(self) -> bool:
        return self.shutdown_requested

    def _image_lock(self, image_id: str) -> asyncio.Lock:
        lock = self.image_locks.get(image_id)
        if lock is None:
            lock = asyncio.Lock()
            self.image_locks[image_id] = lock
        return lock

    def _schedule_process_stop(self) -> None:
        if self.stop_scheduled:
            return
        self.stop_scheduled = True
        loop = asyncio.get_running_loop()
        loop.call_later(0.35, lambda: os.kill(os.getpid(), signal.SIGTERM))

    def _finalize_shutdown_if_idle(self) -> None:
        if (
            self.shutdown_requested
            and not self.stop_scheduled
            and self.active_task_id is None
            and self.active_image_id is None
            and self.active_manual_edits == 0
            and self.queue.empty()
        ):
            self._schedule_process_stop()

    def _stop_task(
        self,
        task_id: str,
        images: list[dict[str, Any]],
        *,
        start_index: int,
        completed_files: int,
        progress: float,
        reason: str,
        log_message: str,
    ) -> int:
        stopped_count = 0
        for image in images[start_index:]:
            self.database.update_image(
                image["id"],
                status="stopped",
                stage="shutdown",
                error=reason,
            )
            stopped_count += 1
        self.database.update_task(
            task_id,
            status="stopped",
            current_stage="shutdown",
            completed_files=completed_files,
            progress=progress,
            error=reason,
        )
        self.database.log(task_id, "WARNING", "shutdown", log_message)
        return stopped_count

    async def request_shutdown(self) -> dict[str, Any]:
        if self.shutdown_requested:
            self._finalize_shutdown_if_idle()
            return {
                "already_requested": True,
                "active_task_id": self.active_task_id,
            }

        self.shutdown_requested = True
        stopped_tasks = 0

        while True:
            try:
                task_id = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.queue.task_done()
            task = self.database.get_task(task_id)
            if not task:
                continue
            self._stop_task(
                task_id,
                task["images"],
                start_index=0,
                completed_files=task.get("completed_files", 0),
                progress=task.get("progress", 0.0),
                reason="服务手动停止，任务未开始执行",
                log_message="服务手动停止：任务未开始执行",
            )
            stopped_tasks += 1

        if self.active_task_id:
            stage = "shutdown"
            message = (
                "收到停止服务请求：当前图片处理完成后停止"
                if self.active_image_id
                else "收到停止服务请求：当前任务即将停止"
            )
            self.database.log(self.active_task_id, "WARNING", stage, message)

        self._finalize_shutdown_if_idle()
        return {
            "already_requested": False,
            "active_task_id": self.active_task_id,
            "active_image_id": self.active_image_id,
            "stopped_tasks": stopped_tasks,
        }

    async def _worker(self) -> None:
        while True:
            task_id = await self.queue.get()
            try:
                await self._process_task(task_id)
            except Exception as exc:
                self.database.update_task(
                    task_id, status="failed", current_stage="error", error=str(exc)
                )
                self.database.log(
                    task_id,
                    "ERROR",
                    "error",
                    str(exc),
                    details={"traceback": traceback.format_exc()},
                )
            finally:
                self.queue.task_done()

    async def _process_task(self, task_id: str) -> None:
        task = self.database.get_task(task_id)
        if not task:
            return
        self.active_task_id = task_id
        config = task["config"]
        self.database.update_task(
            task_id, status="running", current_stage="starting", error=None
        )
        self.database.log(task_id, "INFO", "starting", "任务开始处理")
        failures = 0
        total = max(1, task["total_files"])
        processed = 0

        try:
            for index, image in enumerate(task["images"]):
                if self.shutdown_requested and self.active_image_id is None:
                    remaining = self._stop_task(
                        task_id,
                        task["images"],
                        start_index=index,
                        completed_files=processed,
                        progress=processed / total,
                        reason=f"服务手动停止，剩余 {len(task['images']) - index} 张未继续处理",
                        log_message=(
                            f"服务手动停止：已处理 {processed} 张，失败 {failures} 张，"
                            f"剩余 {len(task['images']) - index} 张未继续处理"
                        ),
                    )
                    if remaining == 0:
                        self.database.update_task(
                            task_id,
                            status="stopped",
                            current_stage="shutdown",
                            completed_files=processed,
                            progress=processed / total,
                            error="服务手动停止",
                        )
                    return

                image_id = image["id"]
                self.active_image_id = image_id
                last_stage: str | None = None

                async def progress(stage: str, fraction: float):
                    nonlocal last_stage
                    overall = (processed + fraction) / total
                    self.database.update_image(
                        image_id, status="running", stage=stage, progress=fraction
                    )
                    self.database.update_task(
                        task_id, current_stage=stage, progress=overall
                    )
                    if stage != last_stage:
                        self.database.log(
                            task_id,
                            "INFO",
                            stage,
                            f"进入处理阶段：{stage}",
                            image_id=image_id,
                        )
                        last_stage = stage

                async def log(
                    level: str,
                    stage: str,
                    message: str,
                    details: dict[str, Any] | None,
                ):
                    self.database.log(
                        task_id,
                        level,
                        stage,
                        message,
                        image_id=image_id,
                        details=details,
                    )

                engine = self._build_engine(config, progress, log)
                self.database.log(
                    task_id,
                    "INFO",
                    "starting",
                    f"开始处理：{image['relative_path']}",
                    image_id=image_id,
                )
                try:
                    async with self.model_lock:
                        async with self._image_lock(image_id):
                            regions = await engine.process(
                                Path(image["input_path"]),
                                Path(image["output_path"]),
                                Path(image["context_path"]),
                                Path(image["regions_path"]),
                            )
                    self.database.update_image(
                        image_id, status="completed", stage="saved", progress=1.0
                    )
                    self.database.log(
                        task_id,
                        "INFO",
                        "saved",
                        f"已保存结果，共 {len(regions)} 个文本区域",
                        image_id=image_id,
                    )
                except Exception as exc:
                    failures += 1
                    self.database.update_image(
                        image_id, status="failed", stage="error", error=str(exc)
                    )
                    self.database.log(
                        task_id,
                        "ERROR",
                        "error",
                        f"{image['relative_path']}：{exc}",
                        image_id=image_id,
                        details={"traceback": traceback.format_exc()},
                    )
                finally:
                    processed += 1
                    self.active_image_id = None
                    if index == total - 1 or processed % 5 == 0:
                        trim_runtime_memory()
                    self.database.update_task(task_id, completed_files=processed)

                if self.shutdown_requested:
                    remaining = len(task["images"]) - processed
                    if remaining > 0:
                        self._stop_task(
                            task_id,
                            task["images"],
                            start_index=processed,
                            completed_files=processed,
                            progress=processed / total,
                            reason=f"服务手动停止，剩余 {remaining} 张未继续处理",
                            log_message=(
                                f"服务手动停止：已处理 {processed} 张，失败 {failures} 张，"
                                f"剩余 {remaining} 张未继续处理"
                            ),
                        )
                        return

            status = "completed_with_errors" if failures else "completed"
            self.database.update_task(
                task_id,
                status=status,
                current_stage="finished",
                progress=1.0,
                error=f"{failures} 张图片处理失败" if failures else None,
            )
            self.database.log(
                task_id,
                "WARNING" if failures else "INFO",
                "finished",
                f"任务完成：成功 {task['total_files'] - failures}，失败 {failures}",
            )
        finally:
            self.active_image_id = None
            self.active_task_id = None
            self._finalize_shutdown_if_idle()

    async def rerender_image(
        self, image: dict[str, Any], updates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.active_manual_edits += 1
        try:
            async with self.model_lock:
                async with self._image_lock(image["id"]):
                    return await rerender(
                        Path(image["context_path"]),
                        Path(image["regions_path"]),
                        Path(image["output_path"]),
                        Path(image["input_path"]),
                        updates,
                    )
        finally:
            self.active_manual_edits -= 1
            self._finalize_shutdown_if_idle()

    async def reprocess_image(
        self,
        image: dict[str, Any],
        updates: list[dict[str, Any]],
        changed_indices: list[int],
    ) -> list[dict[str, Any]]:
        task = self.database.get_task(image["task_id"])
        if not task:
            raise RuntimeError("图片所属任务不存在")

        async def progress(stage: str, fraction: float):
            self.database.update_image(
                image["id"], status="running", stage=stage, progress=fraction
            )

        async def log(
            level: str,
            stage: str,
            message: str,
            details: dict[str, Any] | None,
        ):
            self.database.log(
                image["task_id"],
                level,
                stage,
                message,
                image_id=image["id"],
                details=details,
            )

        self.active_manual_edits += 1
        try:
            async with self.model_lock:
                async with self._image_lock(image["id"]):
                    engine = self._build_engine(task["config"], progress, log)
                    regions = await engine.reprocess_regions(
                        Path(image["input_path"]),
                        Path(image["output_path"]),
                        Path(image["context_path"]),
                        Path(image["regions_path"]),
                        updates,
                        changed_indices,
                    )
                    self.database.update_image(
                        image["id"], status="completed", stage="saved", progress=1.0
                    )
                    return regions
        finally:
            self.active_manual_edits -= 1
            self._finalize_shutdown_if_idle()
