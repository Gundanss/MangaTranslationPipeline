from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    MODEL_DIR,
    OUTPUT_DIR,
    ROOT_DIR,
    STATIC_DIR,
    UPLOAD_DIR,
    VENDOR_CORE_DIR,
    ensure_runtime_dirs,
    safe_name,
    safe_relative_path,
)
from .db import Database
from .providers import (
    TranslationError,
    create_provider,
    list_ollama_models,
    sanitize_translation_text,
)
from .schemas import (
    ReprocessRegionsRequest,
    RerenderRequest,
    SettingsUpdate,
    TaskConfig,
    TranslateRegionRequest,
)
from .secret_store import SecretStore
from .tasks import TaskManager

ensure_runtime_dirs()
database = Database()
secrets = SecretStore()
manager = TaskManager(database, secrets)
LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.start()
    yield


app = FastAPI(title="漫画翻译流水线", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _image_public(image: dict) -> dict:
    return {
        "id": image["id"],
        "relative_path": image["relative_path"],
        "status": image["status"],
        "progress": image["progress"],
        "stage": image["stage"],
        "error": image["error"],
        "original_url": f"/api/images/{image['id']}/file/original",
        "result_url": f"/api/images/{image['id']}/file/result",
        "regions_url": f"/api/images/{image['id']}/regions",
    }


def _task_public(task: dict) -> dict:
    result = {
        key: task[key]
        for key in (
            "id",
            "name",
            "status",
            "total_files",
            "completed_files",
            "progress",
            "current_stage",
            "error",
            "output_dir",
            "created_at",
            "updated_at",
        )
    }
    result["config"] = task["config"]
    if "images" in task:
        result["images"] = [_image_public(image) for image in task["images"]]
    return result


def _normalize_region_json(regions: list[dict]) -> tuple[list[dict], bool]:
    changed = False
    normalized: list[dict] = []
    for index, region in enumerate(regions):
        item = dict(region)
        bbox = item.get("bbox") or item.get("render_bbox") or item.get("ocr_bbox")
        if item.get("index") != index:
            item["index"] = index
            changed = True
        if "ocr_bbox" not in item and bbox:
            item["ocr_bbox"] = bbox
            changed = True
        if "render_bbox" not in item and bbox:
            item["render_bbox"] = bbox
            changed = True
        if item.get("bbox") != item.get("render_bbox") and item.get("render_bbox"):
            item["bbox"] = item["render_bbox"]
            changed = True
        if "enabled" not in item:
            item["enabled"] = True
            changed = True
        if "mask_dilation_offset" not in item:
            item["mask_dilation_offset"] = 20
            changed = True
        if "angle" not in item:
            item["angle"] = 0
            changed = True
        cleaned = sanitize_translation_text(item.get("translation", ""))
        if cleaned != item.get("translation", ""):
            item["translation"] = cleaned
            changed = True
        normalized.append(item)
    return normalized, changed


async def _validate_config(config: TaskConfig) -> None:
    settings = secrets.get()
    try:
        create_provider(config.provider, settings, config.ollama_model)
    except TranslationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if config.provider == "ollama" or config.polish_with_ollama:
        try:
            models = await list_ollama_models(settings["ollama_base_url"])
        except TranslationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        names = {model["name"] for model in models}
        selected = (
            config.ollama_model
            if config.provider == "ollama"
            else config.polish_model
        )
        if not selected:
            raise HTTPException(status_code=400, detail="必须选择一个 Ollama 模型")
        if selected not in names:
            raise HTTPException(
                status_code=400,
                detail=f"本机 Ollama 中不存在模型：{selected}。系统不会自动下载模型。",
            )


def _require_local_request(request: Request) -> None:
    host = request.client.host if request.client else None
    if host not in LOCAL_CLIENTS:
        raise HTTPException(status_code=403, detail="仅允许本机请求停止本地服务")


def _ensure_service_writable() -> None:
    if manager.is_shutting_down:
        raise HTTPException(status_code=503, detail="服务正在停止，暂不接受新的处理请求")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    required_models = {
        "ctd": MODEL_DIR / "detection" / "comictextdetector.pt.onnx",
        "ocr": MODEL_DIR / "ocr" / "ocr_ar_48px.ckpt",
        "ocr_dictionary": MODEL_DIR / "ocr" / "alphabet-all-v7.txt",
        "inpainting": MODEL_DIR / "inpainting" / "lama_large_512px.ckpt",
    }
    return {
        "ok": True,
        "shutting_down": manager.is_shutting_down,
        "core_present": (VENDOR_CORE_DIR / "manga_translator" / "__init__.py").exists(),
        "models": {name: path.exists() for name, path in required_models.items()},
        "output_dir": str(OUTPUT_DIR),
    }


@app.get("/api/settings")
async def get_settings():
    return secrets.public()


@app.put("/api/settings")
async def update_settings(update: SettingsUpdate):
    return secrets.update(update)


@app.get("/api/ollama/models")
async def ollama_models():
    settings = secrets.get()
    try:
        models = await list_ollama_models(settings["ollama_base_url"])
    except TranslationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"models": models, "base_url": settings["ollama_base_url"]}


@app.post("/api/tasks")
async def create_task(
    files: list[UploadFile] = File(...),
    config_json: str = Form(...),
    relative_paths: list[str] = Form(default=[]),
):
    _ensure_service_writable()
    try:
        config = TaskConfig.model_validate_json(config_json)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"任务配置无效：{exc}") from exc
    await _validate_config(config)
    if not files:
        raise HTTPException(status_code=400, detail="请至少选择一张图片")

    task_id = uuid.uuid4().hex[:12]
    task_name = safe_name(config.name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    output_dir = OUTPUT_DIR / f"{stamp}-{task_name}"
    upload_dir = UPLOAD_DIR / task_id
    private_dir = output_dir / ".pipeline"
    images: list[dict] = []
    used_paths: set[Path] = set()

    for index, upload in enumerate(files):
        given_path = (
            relative_paths[index]
            if index < len(relative_paths)
            else upload.filename or f"image-{index + 1}.png"
        )
        try:
            relative = safe_relative_path(given_path, upload.filename or "image.png")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if relative in used_paths:
            relative = relative.with_stem(f"{relative.stem}-{index + 1}")
        used_paths.add(relative)
        image_id = uuid.uuid4().hex[:12]
        input_path = upload_dir / relative
        output_path = output_dir / relative
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(await upload.read())
        images.append(
            {
                "id": image_id,
                "relative_path": relative.as_posix(),
                "input_path": str(input_path),
                "output_path": str(output_path),
                "context_path": str(private_dir / "contexts" / f"{image_id}.pkl"),
                "regions_path": str(private_dir / "regions" / f"{image_id}.json"),
            }
        )

    database.create_task(
        {
            "id": task_id,
            "name": task_name,
            "config": config.model_dump(),
            "output_dir": str(output_dir),
        },
        images,
    )
    selected_model = config.ollama_model or config.polish_model
    if selected_model:
        secrets.update({"last_ollama_model": selected_model})
    database.log(task_id, "INFO", "queued", f"任务已加入队列，共 {len(images)} 张图片")
    await manager.enqueue(task_id)
    task = database.get_task(task_id)
    return _task_public(task)


@app.post("/api/system/shutdown")
async def shutdown_system(request: Request, background_tasks: BackgroundTasks):
    _require_local_request(request)
    payload = {
        "ok": True,
        "mode": "graceful",
        "message": "停止请求已受理，服务会在当前这张图片处理完成后关闭。",
    }
    if manager.is_shutting_down:
        return {
            **payload,
            "message": "停止请求已受理，服务正在关闭中。",
        }
    background_tasks.add_task(manager.request_shutdown)
    if manager.active_task_id is None and manager.active_manual_edits == 0:
        payload["message"] = "停止请求已受理，当前没有处理中的图片，服务即将关闭。"
    elif manager.active_task_id is None:
        payload["message"] = "停止请求已受理，当前编辑操作完成后关闭服务。"
    return payload


@app.get("/api/tasks")
async def list_tasks():
    return {"tasks": [_task_public(task) for task in database.list_tasks()]}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _task_public(task)


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str, request: Request):
    if not database.get_task(task_id):
        raise HTTPException(status_code=404, detail="任务不存在")

    async def stream():
        last_log_id = 0
        while not await request.is_disconnected():
            task = database.get_task(task_id)
            logs = database.get_logs(task_id, last_log_id)
            if logs:
                last_log_id = logs[-1]["id"]
            payload = {"task": _task_public(task), "logs": logs}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.6)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/images/{image_id}/file/{kind}")
async def image_file(image_id: str, kind: str):
    image = database.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="图片不存在")
    if kind == "original":
        path = Path(image["input_path"])
    elif kind == "result":
        path = Path(image["output_path"])
    else:
        raise HTTPException(status_code=400, detail="未知文件类型")
    if not path.exists():
        raise HTTPException(status_code=404, detail="文件尚未生成")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.get("/api/images/{image_id}/regions")
async def get_regions(image_id: str):
    image = database.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="图片不存在")
    path = Path(image["regions_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="文本区域尚未生成")
    regions, changed = _normalize_region_json(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if changed:
        path.write_text(
            json.dumps(regions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "image": _image_public(image),
        "regions": regions,
    }


@app.post("/api/images/{image_id}/rerender")
async def rerender_image(image_id: str, request: RerenderRequest):
    _ensure_service_writable()
    image = database.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="图片不存在")
    if not Path(image["context_path"]).exists():
        raise HTTPException(status_code=409, detail="缺少可重新嵌字的处理上下文")
    try:
        regions = await manager.rerender_image(
            image, [region.model_dump() for region in request.regions]
        )
    except Exception as exc:
        database.log(
            image["task_id"],
            "ERROR",
            "rerender",
            f"重新嵌字失败：{exc}",
            image_id=image_id,
        )
        raise HTTPException(status_code=500, detail=f"重新嵌字失败：{exc}") from exc
    database.log(
        image["task_id"],
        "INFO",
        "rerender",
        "人工校正后重新嵌字完成",
        image_id=image_id,
    )
    return {"ok": True, "regions": regions, "result_url": f"/api/images/{image_id}/file/result"}


@app.post("/api/images/{image_id}/reprocess-regions")
async def reprocess_regions(image_id: str, request: ReprocessRegionsRequest):
    _ensure_service_writable()
    image = database.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="图片不存在")
    if not Path(image["context_path"]).exists():
        raise HTTPException(status_code=409, detail="缺少可重新处理的上下文")
    try:
        regions = await manager.reprocess_image(
            image,
            [region.model_dump() for region in request.regions],
            request.changed_indices,
            request.mask_changed_indices,
        )
    except Exception as exc:
        database.log(
            image["task_id"],
            "ERROR",
            "manual-ocr",
            f"人工 OCR 框重处理失败：{exc}",
            image_id=image_id,
        )
        raise HTTPException(status_code=500, detail=f"人工 OCR 框重处理失败：{exc}") from exc
    database.log(
        image["task_id"],
        "INFO",
        "manual-ocr",
        "人工 OCR 框调整后重新处理完成",
        image_id=image_id,
    )
    return {
        "ok": True,
        "regions": regions,
        "result_url": f"/api/images/{image_id}/file/result",
    }


@app.post("/api/images/{image_id}/translate-region")
async def translate_region(image_id: str, request: TranslateRegionRequest):
    _ensure_service_writable()
    image = database.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="图片不存在")
    task = database.get_task(image["task_id"])
    if not task:
        raise HTTPException(status_code=404, detail="图片所属任务不存在")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="OCR 原文为空，无法翻译")

    config = task["config"]
    settings = secrets.get()
    if request.mode == "machine":
        provider_name = config.get("provider")
        if provider_name == "ollama":
            raise HTTPException(
                status_code=400,
                detail="当前任务使用的是 Ollama；机器翻译只复用 Google 或 Bing/Microsoft 在线配置",
            )
        provider = create_provider(provider_name, settings, None)
        label = "机器翻译"
    else:
        model = (
            config.get("ollama_model")
            or config.get("polish_model")
            or settings.get("last_ollama_model")
        )
        if not model:
            raise HTTPException(
                status_code=400,
                detail="没有可用的 Ollama 模型，请先在任务或本机设置中选择模型",
            )
        provider = create_provider("ollama", settings, model)
        label = "Ollama 翻译"

    async def log(level: str, stage: str, message: str, details: dict | None):
        database.log(
            image["task_id"],
            level,
            stage,
            message,
            image_id=image_id,
            details=details,
        )

    provider.set_log_callback(log)
    try:
        translations = await provider.translate(
            [text],
            config["source_language"],
            config["target_language"],
        )
    except TranslationError as exc:
        database.log(
            image["task_id"],
            "ERROR",
            "translation",
            f"{label}失败：{exc}",
            image_id=image_id,
        )
        raise HTTPException(status_code=502, detail=f"{label}失败：{exc}") from exc

    translation = sanitize_translation_text(translations[0] if translations else "")
    if not translation:
        raise HTTPException(status_code=502, detail=f"{label}返回了空译文")
    database.log(
        image["task_id"],
        "INFO",
        "translation",
        f"{label}完成：区域译文已更新到编辑框",
        image_id=image_id,
        details={"pairs": [{"source": text, "translation": translation}]},
    )
    return {"ok": True, "translation": translation}


@app.exception_handler(Exception)
async def unhandled_exception(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
