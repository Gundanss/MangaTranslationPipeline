from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DATABASE_PATH, ensure_runtime_dirs


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path = DATABASE_PATH):
        ensure_runtime_dirs()
        self.path = path
        self._write_lock = threading.Lock()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    total_files INTEGER NOT NULL DEFAULT 0,
                    completed_files INTEGER NOT NULL DEFAULT 0,
                    progress REAL NOT NULL DEFAULT 0,
                    current_stage TEXT NOT NULL DEFAULT 'queued',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(id),
                    relative_path TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    context_path TEXT NOT NULL,
                    regions_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    progress REAL NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT 'queued',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(id),
                    image_id TEXT,
                    level TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_images_task ON images(task_id);
                CREATE INDEX IF NOT EXISTS idx_logs_task_id ON logs(task_id, id);
                """
            )

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._write_lock, self.connect() as conn:
            conn.execute(sql, params)
            conn.commit()

    def create_task(self, task: dict[str, Any], images: list[dict[str, Any]]) -> None:
        timestamp = now_iso()
        with self._write_lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks
                (id, name, status, config_json, output_dir, total_files,
                 created_at, updated_at)
                VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    task["id"],
                    task["name"],
                    json.dumps(task["config"], ensure_ascii=False),
                    task["output_dir"],
                    len(images),
                    timestamp,
                    timestamp,
                ),
            )
            conn.executemany(
                """
                INSERT INTO images
                (id, task_id, relative_path, input_path, output_path, context_path,
                 regions_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        image["id"],
                        task["id"],
                        image["relative_path"],
                        image["input_path"],
                        image["output_path"],
                        image["context_path"],
                        image["regions_path"],
                        timestamp,
                        timestamp,
                    )
                    for image in images
                ],
            )
            conn.commit()

    def update_task(self, task_id: str, **values: Any) -> None:
        values["updated_at"] = now_iso()
        fields = ", ".join(f"{key} = ?" for key in values)
        self._execute(
            f"UPDATE tasks SET {fields} WHERE id = ?",
            tuple(values.values()) + (task_id,),
        )

    def update_image(self, image_id: str, **values: Any) -> None:
        values["updated_at"] = now_iso()
        fields = ", ".join(f"{key} = ?" for key in values)
        self._execute(
            f"UPDATE images SET {fields} WHERE id = ?",
            tuple(values.values()) + (image_id,),
        )

    def log(
        self,
        task_id: str,
        level: str,
        stage: str,
        message: str,
        image_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO logs
            (task_id, image_id, level, stage, message, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                image_id,
                level,
                stage,
                message,
                json.dumps(details, ensure_ascii=False) if details else None,
                now_iso(),
            ),
        )

    def _one(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def _all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task = self._one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        if not task:
            return None
        task["config"] = json.loads(task.pop("config_json"))
        task["images"] = self._all(
            "SELECT * FROM images WHERE task_id = ? ORDER BY relative_path",
            (task_id,),
        )
        return task

    def list_tasks(self, limit: int = 30) -> list[dict[str, Any]]:
        tasks = self._all(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        for task in tasks:
            task["config"] = json.loads(task.pop("config_json"))
        return tasks

    def get_image(self, image_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM images WHERE id = ?", (image_id,))

    def get_logs(self, task_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        logs = self._all(
            "SELECT * FROM logs WHERE task_id = ? AND id > ? ORDER BY id LIMIT 1000",
            (task_id, after_id),
        )
        for log in logs:
            details_json = log.pop("details_json")
            log["details"] = json.loads(details_json) if details_json else None
        return logs
