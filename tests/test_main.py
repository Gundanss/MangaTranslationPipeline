import asyncio
import json
from types import SimpleNamespace

import httpx
from fastapi import HTTPException

from manga_pipeline import main


class FakeManager:
    def __init__(self, *, shutting_down=False):
        self.is_shutting_down = shutting_down
        self.active_task_id = None
        self.active_manual_edits = 0
        self.shutdown_calls = 0
        self.enqueued = []

    def start(self):
        return None

    async def request_shutdown(self):
        self.shutdown_calls += 1
        self.is_shutting_down = True
        return {"ok": True}

    async def enqueue(self, task_id):
        self.enqueued.append(task_id)


async def request_json(path, *, method="GET", client_host="127.0.0.1", **kwargs):
    transport = httpx.ASGITransport(
        app=main.app,
        client=(client_host, 4321),
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.request(method, path, **kwargs)
    return response


def test_health_reports_shutdown_flag(monkeypatch):
    fake_manager = FakeManager(shutting_down=True)
    monkeypatch.setattr(main, "manager", fake_manager)

    response = asyncio.run(request_json("/api/health"))

    assert response.status_code == 200
    assert response.json()["shutting_down"] is True


def test_index_disables_cache():
    response = asyncio.run(request_json("/"))

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_static_assets_disable_cache():
    response = asyncio.run(request_json("/static/app.js"))

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_shutdown_route_rejects_non_local_requests():
    request = SimpleNamespace(client=SimpleNamespace(host="10.0.0.8"))

    try:
        main._require_local_request(request)
    except HTTPException as exc:
        assert exc.status_code == 403
        assert "仅允许本机请求" in exc.detail
    else:
        raise AssertionError("non-local request should be rejected")


def test_shutdown_route_accepts_local_request_and_runs_background_task(monkeypatch):
    fake_manager = FakeManager()
    monkeypatch.setattr(main, "manager", fake_manager)

    response = asyncio.run(request_json("/api/system/shutdown", method="POST"))

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert fake_manager.shutdown_calls == 1


def test_mutation_routes_reject_when_service_is_shutting_down(monkeypatch):
    fake_manager = FakeManager(shutting_down=True)
    monkeypatch.setattr(main, "manager", fake_manager)

    create_response = asyncio.run(
        request_json(
            "/api/tasks",
            method="POST",
            files={"files": ("page.png", b"png", "image/png")},
            data={
                "config_json": json.dumps(
                    {
                        "name": "测试",
                        "source_language": "ja",
                        "target_language": "zh-CN",
                        "provider": "google",
                    },
                    ensure_ascii=False,
                ),
                "relative_paths": "page.png",
            },
        )
    )
    rerender_response = asyncio.run(
        request_json(
            "/api/images/any-image/rerender",
            method="POST",
            json={"regions": []},
        )
    )
    reprocess_response = asyncio.run(
        request_json(
            "/api/images/any-image/reprocess-regions",
            method="POST",
            json={"regions": [], "changed_indices": []},
        )
    )
    resume_response = asyncio.run(
        request_json(
            "/api/tasks/task-1/resume-from-image",
            method="POST",
            json={"image_id": "image-1"},
        )
    )

    assert create_response.status_code == 503
    assert rerender_response.status_code == 503
    assert reprocess_response.status_code == 503
    assert resume_response.status_code == 503


def test_normalize_region_json_adds_default_mask_dilation():
    regions, changed = main._normalize_region_json(
        [
            {
                "index": 0,
                "bbox": [0, 0, 10, 10],
                "enabled": True,
                "text": "原文",
                "translation": "译文",
            }
        ]
    )

    assert changed is True
    assert regions[0]["mask_dilation_offset"] == 20
    assert regions[0]["angle"] == 0


class FakeDatabase:
    def __init__(self, config):
        self.config = config
        self.logs = []

    def get_image(self, image_id):
        if image_id != "image-1":
            return None
        return {"id": "image-1", "task_id": "task-1"}

    def get_task(self, task_id):
        if task_id != "task-1":
            return None
        return {"id": "task-1", "config": self.config, "images": []}

    def log(self, task_id, level, stage, message, image_id=None, details=None):
        self.logs.append(
            {
                "task_id": task_id,
                "level": level,
                "stage": stage,
                "message": message,
                "image_id": image_id,
                "details": details,
            }
        )


class ResumeDatabase:
    def __init__(self, task_status="completed_with_errors"):
        self.logs = []
        self.images = {
            "image-1": {
                "id": "image-1",
                "task_id": "task-1",
                "relative_path": "001.png",
                "status": "failed",
                "progress": 0.52,
                "stage": "error",
                "error": "network",
            },
            "image-2": {
                "id": "image-2",
                "task_id": "task-1",
                "relative_path": "002.png",
                "status": "completed",
                "progress": 1.0,
                "stage": "saved",
                "error": None,
            },
            "image-3": {
                "id": "image-3",
                "task_id": "task-1",
                "relative_path": "003.png",
                "status": "queued",
                "progress": 0.0,
                "stage": "queued",
                "error": None,
            },
        }
        self.task = {
            "id": "task-1",
            "name": "task-1",
            "status": task_status,
            "config": {
                "provider": "google",
                "source_language": "ja",
                "target_language": "zh-CN",
            },
            "total_files": 3,
            "completed_files": 3,
            "progress": 1.0,
            "current_stage": "finished",
            "error": "1 张图片处理失败",
            "output_dir": "/tmp/out",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
        }

    def get_image(self, image_id):
        image = self.images.get(image_id)
        return dict(image) if image else None

    def get_task(self, task_id):
        if task_id != "task-1":
            return None
        return {
            **self.task,
            "images": [dict(self.images[key]) for key in ("image-1", "image-2", "image-3")],
        }

    def update_task(self, task_id, **values):
        assert task_id == "task-1"
        self.task.update(values)

    def update_image(self, image_id, **values):
        self.images[image_id].update(values)

    def log(self, task_id, level, stage, message, image_id=None, details=None):
        self.logs.append(
            {
                "task_id": task_id,
                "level": level,
                "stage": stage,
                "message": message,
                "image_id": image_id,
                "details": details,
            }
        )


class FakeSecrets:
    def __init__(self, values=None):
        self.values = {
            "ollama_base_url": "http://localhost:11434",
            "google_api_key": "",
            "microsoft_api_key": "",
            "microsoft_region": "",
            "microsoft_endpoint": "https://api.cognitive.microsofttranslator.com",
            "last_ollama_model": "",
        }
        self.values.update(values or {})

    def get(self):
        return self.values


class FakeProvider:
    def __init__(self):
        self.log_callback = None

    def set_log_callback(self, callback):
        self.log_callback = callback

    async def translate(self, texts, source, target):
        return [f"{source}->{target}:{text}" for text in texts]


def test_translate_region_machine_uses_task_online_provider(monkeypatch):
    fake_database = FakeDatabase(
        {"provider": "google", "source_language": "ja", "target_language": "zh-CN"}
    )
    captured = {}
    monkeypatch.setattr(main, "manager", FakeManager())
    monkeypatch.setattr(main, "database", fake_database)
    monkeypatch.setattr(main, "secrets", FakeSecrets())

    def fake_create_provider(name, settings, ollama_model):
        captured.update({"name": name, "ollama_model": ollama_model})
        return FakeProvider()

    monkeypatch.setattr(main, "create_provider", fake_create_provider)

    response = asyncio.run(
        request_json(
            "/api/images/image-1/translate-region",
            method="POST",
            json={"mode": "machine", "text": "勉強しなさい"},
        )
    )

    assert response.status_code == 200
    assert response.json()["translation"] == "ja->zh-CN:勉強しなさい"
    assert captured == {"name": "google", "ollama_model": None}
    assert fake_database.logs[-1]["stage"] == "translation"


def test_resume_from_image_requeues_selected_and_later_unfinished_images(monkeypatch):
    fake_manager = FakeManager()
    fake_database = ResumeDatabase()
    monkeypatch.setattr(main, "manager", fake_manager)
    monkeypatch.setattr(main, "database", fake_database)

    response = asyncio.run(
        request_json(
            "/api/tasks/task-1/resume-from-image",
            method="POST",
            json={"image_id": "image-1"},
        )
    )

    assert response.status_code == 200
    assert fake_manager.enqueued == ["task-1"]
    assert fake_database.images["image-1"]["status"] == "queued"
    assert fake_database.images["image-1"]["stage"] == "retry"
    assert fake_database.images["image-2"]["status"] == "completed"
    assert fake_database.images["image-3"]["status"] == "queued"
    assert fake_database.task["status"] == "queued"
    assert fake_database.task["current_stage"] == "retry"
    assert fake_database.task["completed_files"] == 1
    assert fake_database.task["progress"] == 1 / 3
    assert fake_database.logs[-1]["stage"] == "retry"


def test_resume_from_image_rejects_running_task(monkeypatch):
    monkeypatch.setattr(main, "manager", FakeManager())
    monkeypatch.setattr(main, "database", ResumeDatabase(task_status="running"))

    response = asyncio.run(
        request_json(
            "/api/tasks/task-1/resume-from-image",
            method="POST",
            json={"image_id": "image-1"},
        )
    )

    assert response.status_code == 409
    assert "才能续跑" in response.json()["detail"]


def test_resume_from_image_returns_404_for_missing_image(monkeypatch):
    monkeypatch.setattr(main, "manager", FakeManager())
    monkeypatch.setattr(main, "database", ResumeDatabase())

    response = asyncio.run(
        request_json(
            "/api/tasks/task-1/resume-from-image",
            method="POST",
            json={"image_id": "missing-image"},
        )
    )

    assert response.status_code == 404


def test_translate_region_machine_rejects_ollama_task(monkeypatch):
    monkeypatch.setattr(main, "manager", FakeManager())
    monkeypatch.setattr(
        main,
        "database",
        FakeDatabase(
            {
                "provider": "ollama",
                "source_language": "ja",
                "target_language": "zh-CN",
                "ollama_model": "task-model",
            }
        ),
    )
    monkeypatch.setattr(main, "secrets", FakeSecrets())

    response = asyncio.run(
        request_json(
            "/api/images/image-1/translate-region",
            method="POST",
            json={"mode": "machine", "text": "勉強しなさい"},
        )
    )

    assert response.status_code == 400
    assert "机器翻译只复用" in response.json()["detail"]


def test_translate_region_ollama_uses_recent_model_fallback(monkeypatch):
    captured = {}
    monkeypatch.setattr(main, "manager", FakeManager())
    monkeypatch.setattr(
        main,
        "database",
        FakeDatabase(
            {"provider": "google", "source_language": "ja", "target_language": "zh-CN"}
        ),
    )
    monkeypatch.setattr(main, "secrets", FakeSecrets({"last_ollama_model": "last-model"}))

    def fake_create_provider(name, settings, ollama_model):
        captured.update({"name": name, "ollama_model": ollama_model})
        return FakeProvider()

    monkeypatch.setattr(main, "create_provider", fake_create_provider)

    response = asyncio.run(
        request_json(
            "/api/images/image-1/translate-region",
            method="POST",
            json={"mode": "ollama", "text": "勉強しなさい"},
        )
    )

    assert response.status_code == 200
    assert captured == {"name": "ollama", "ollama_model": "last-model"}
