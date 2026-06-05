from __future__ import annotations

import asyncio
import json
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
        self.processing_lock = asyncio.Lock()

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

    async def _worker(self) -> None:
        while True:
            task_id = await self.queue.get()
            try:
                async with self.processing_lock:
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
        config = task["config"]
        self.database.update_task(
            task_id, status="running", current_stage="starting", error=None
        )
        self.database.log(task_id, "INFO", "starting", "任务开始处理")
        failures = 0
        total = max(1, task["total_files"])

        for completed, image in enumerate(task["images"]):
            image_id = image["id"]
            last_stage: str | None = None

            async def progress(stage: str, fraction: float):
                nonlocal last_stage
                overall = (completed + fraction) / total
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
                    task_id, level, stage, message, image_id=image_id, details=details
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
                if completed == total - 1 or (completed + 1) % 5 == 0:
                    trim_runtime_memory()
                self.database.update_task(task_id, completed_files=completed + 1)

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

    async def rerender_image(
        self, image: dict[str, Any], updates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        async with self.processing_lock:
            return await rerender(
                Path(image["context_path"]),
                Path(image["regions_path"]),
                Path(image["output_path"]),
                Path(image["input_path"]),
                updates,
            )

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

        async with self.processing_lock:
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
