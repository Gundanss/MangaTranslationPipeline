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

    def start(self):
        return None

    async def request_shutdown(self):
        self.shutdown_calls += 1
        self.is_shutting_down = True
        return {"ok": True}


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

    assert create_response.status_code == 503
    assert rerender_response.status_code == 503
    assert reprocess_response.status_code == 503
