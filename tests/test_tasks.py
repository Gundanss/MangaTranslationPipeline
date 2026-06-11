import asyncio
from pathlib import Path

from manga_pipeline import tasks
from manga_pipeline.tasks import TaskManager


class FakeSecrets:
    def get(self):
        return {}


class FakeDatabase:
    def __init__(self, task_specs):
        self.tasks = {}
        self.images = {}
        self.logs = []
        for task_id, image_specs in task_specs.items():
            image_rows = []
            for image_id, path in image_specs:
                row = {
                    "id": image_id,
                    "task_id": task_id,
                    "relative_path": f"{image_id}.png",
                    "input_path": str(path),
                    "output_path": str(path.with_name(f"{image_id}-out.png")),
                    "context_path": str(path.with_name(f"{image_id}.pkl")),
                    "regions_path": str(path.with_name(f"{image_id}.json")),
                    "status": "queued",
                    "progress": 0.0,
                    "stage": "queued",
                    "error": None,
                }
                image_rows.append(row)
                self.images[image_id] = row
            self.tasks[task_id] = {
                "id": task_id,
                "name": task_id,
                "config": {"provider": "ollama", "source_language": "ja", "target_language": "zh-CN"},
                "status": "queued",
                "progress": 0.0,
                "current_stage": "queued",
                "completed_files": 0,
                "total_files": len(image_rows),
                "error": None,
                "images": image_rows,
            }

    def get_task(self, task_id):
        task = self.tasks.get(task_id)
        if not task:
            return None
        return {
            **task,
            "images": [dict(self.images[row["id"]]) for row in task["images"]],
        }

    def update_task(self, task_id, **values):
        self.tasks[task_id].update(values)

    def update_image(self, image_id, **values):
        self.images[image_id].update(values)

    def log(self, task_id, level, stage, message, image_id=None, details=None):
        self.logs.append((task_id, level, stage, message, image_id, details))


def test_manual_edit_runs_between_batch_images(tmp_path, monkeypatch):
    order = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    edit_started = asyncio.Event()
    release_edit = asyncio.Event()

    first_path = tmp_path / "image-1.png"
    second_path = tmp_path / "image-2.png"
    database = FakeDatabase(
        {"task-1": [("image-1", first_path), ("image-2", second_path)]}
    )
    manager = TaskManager(database, FakeSecrets())

    class FakeEngine:
        async def process(self, input_path, *_args):
            image_id = Path(input_path).stem
            order.append(f"process:{image_id}")
            if image_id == "image-1":
                first_started.set()
                await release_first.wait()
            return []

    manager._build_engine = lambda *_args: FakeEngine()

    async def fake_rerender(*_args):
        order.append("edit:image-1")
        edit_started.set()
        await release_edit.wait()
        return []

    monkeypatch.setattr(tasks, "rerender", fake_rerender)

    async def scenario():
        batch_task = asyncio.create_task(manager._process_task("task-1"))
        await first_started.wait()
        edit_task = asyncio.create_task(
            manager.rerender_image(database.images["image-1"], [])
        )
        await asyncio.sleep(0)
        release_first.set()
        await edit_started.wait()
        release_edit.set()
        await edit_task
        await batch_task

    asyncio.run(scenario())

    assert order == ["process:image-1", "edit:image-1", "process:image-2"]


def test_shutdown_marks_queued_tasks_stopped_and_schedules_process_exit(tmp_path):
    database = FakeDatabase(
        {
            "task-1": [("image-1", tmp_path / "image-1.png")],
            "task-2": [("image-2", tmp_path / "image-2.png")],
        }
    )
    manager = TaskManager(database, FakeSecrets())
    scheduled = []
    manager._schedule_process_stop = lambda: scheduled.append("stop")

    async def scenario():
        await manager.enqueue("task-1")
        await manager.enqueue("task-2")
        result = await manager.request_shutdown()
        return result

    result = asyncio.run(scenario())

    assert result["stopped_tasks"] == 2
    assert scheduled == ["stop"]
    assert database.tasks["task-1"]["status"] == "stopped"
    assert database.tasks["task-2"]["status"] == "stopped"
    assert database.images["image-1"]["status"] == "stopped"
    assert database.images["image-2"]["status"] == "stopped"


def test_shutdown_finishes_current_image_then_stops_remaining_images(tmp_path):
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    order = []

    database = FakeDatabase(
        {
            "task-1": [
                ("image-1", tmp_path / "image-1.png"),
                ("image-2", tmp_path / "image-2.png"),
            ]
        }
    )
    manager = TaskManager(database, FakeSecrets())
    scheduled = []
    manager._schedule_process_stop = lambda: scheduled.append("stop")

    class FakeEngine:
        async def process(self, input_path, *_args):
            image_id = Path(input_path).stem
            order.append(f"process:{image_id}")
            if image_id == "image-1":
                first_started.set()
                await release_first.wait()
            return []

    manager._build_engine = lambda *_args: FakeEngine()

    async def scenario():
        task = asyncio.create_task(manager._process_task("task-1"))
        await first_started.wait()
        result = await manager.request_shutdown()
        release_first.set()
        await task
        return result

    result = asyncio.run(scenario())

    assert result["active_task_id"] == "task-1"
    assert order == ["process:image-1"]
    assert scheduled == ["stop"]
    assert database.tasks["task-1"]["status"] == "stopped"
    assert database.images["image-1"]["status"] == "completed"
    assert database.images["image-2"]["status"] == "stopped"


def test_resume_run_processes_only_requeued_images_and_keeps_old_failures(tmp_path):
    processed = []
    database = FakeDatabase(
        {
            "task-1": [
                ("image-1", tmp_path / "image-1.png"),
                ("image-2", tmp_path / "image-2.png"),
                ("image-3", tmp_path / "image-3.png"),
                ("image-4", tmp_path / "image-4.png"),
            ]
        }
    )
    database.tasks["task-1"].update(
        {"status": "completed_with_errors", "completed_files": 2, "progress": 0.5}
    )
    database.images["image-1"].update(status="failed", stage="error", error="old failure")
    database.images["image-2"].update(status="completed", stage="saved", progress=1.0)
    database.images["image-3"].update(status="queued", stage="retry", progress=0.0, error=None)
    database.images["image-4"].update(status="completed", stage="saved", progress=1.0)

    manager = TaskManager(database, FakeSecrets())

    class FakeEngine:
        async def process(self, input_path, *_args):
            processed.append(Path(input_path).stem)
            return []

    manager._build_engine = lambda *_args: FakeEngine()

    asyncio.run(manager._process_task("task-1"))

    assert processed == ["image-3"]
    assert database.images["image-1"]["status"] == "failed"
    assert database.images["image-2"]["status"] == "completed"
    assert database.images["image-3"]["status"] == "completed"
    assert database.images["image-4"]["status"] == "completed"
    assert database.tasks["task-1"]["status"] == "completed_with_errors"
    assert database.tasks["task-1"]["completed_files"] == 4
    assert database.tasks["task-1"]["error"] == "1 张图片未成功处理"


def test_stop_task_only_marks_queued_images_when_resuming(tmp_path):
    database = FakeDatabase(
        {
            "task-1": [
                ("image-1", tmp_path / "image-1.png"),
                ("image-2", tmp_path / "image-2.png"),
                ("image-3", tmp_path / "image-3.png"),
            ]
        }
    )
    manager = TaskManager(database, FakeSecrets())
    images = database.get_task("task-1")["images"]
    database.images["image-1"]["status"] = "completed"
    database.images["image-3"]["status"] = "failed"
    images[0]["status"] = "completed"
    images[1]["status"] = "queued"
    images[2]["status"] = "failed"

    stopped = manager._stop_task(
        "task-1",
        images,
        start_index=0,
        completed_files=1,
        progress=1 / 3,
        reason="stop",
        log_message="stop",
    )

    assert stopped == 1
    assert database.images["image-1"]["status"] == "completed"
    assert database.images["image-2"]["status"] == "stopped"
    assert database.images["image-3"]["status"] == "failed"
