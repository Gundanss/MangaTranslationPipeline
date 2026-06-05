import asyncio
from pathlib import Path

from manga_pipeline import tasks
from manga_pipeline.tasks import TaskManager


class FakeSecrets:
    def get(self):
        return {}


class FakeDatabase:
    def __init__(self, image_paths):
        self.images = [
            {
                "id": image_id,
                "task_id": "task-1",
                "relative_path": f"{image_id}.png",
                "input_path": str(path),
                "output_path": str(path.with_name(f"{image_id}-out.png")),
                "context_path": str(path.with_name(f"{image_id}.pkl")),
                "regions_path": str(path.with_name(f"{image_id}.json")),
            }
            for image_id, path in image_paths
        ]
        self.task = {
            "id": "task-1",
            "config": {"provider": "ollama"},
            "total_files": len(self.images),
            "images": self.images,
        }
        self.image_updates = []
        self.task_updates = []
        self.logs = []

    def get_task(self, task_id):
        return self.task if task_id == self.task["id"] else None

    def update_task(self, task_id, **values):
        self.task_updates.append((task_id, values))

    def update_image(self, image_id, **values):
        self.image_updates.append((image_id, values))

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
    database = FakeDatabase([("image-1", first_path), ("image-2", second_path)])
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
        edit_task = asyncio.create_task(manager.rerender_image(database.images[0], []))
        await asyncio.sleep(0)
        release_first.set()
        await edit_started.wait()
        release_edit.set()
        await edit_task
        await batch_task

    asyncio.run(scenario())

    assert order == ["process:image-1", "edit:image-1", "process:image-2"]
