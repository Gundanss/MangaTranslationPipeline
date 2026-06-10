from __future__ import annotations

import asyncio
import gc
import inspect
import json
import os
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import cv2
import numpy as np
from PIL import Image

from .config import MODEL_DIR, VENDOR_CORE_DIR
from .providers import OllamaProvider, TranslatorProvider, sanitize_translation_text

ProgressCallback = Callable[[str, float], Awaitable[None]]
LogCallback = Callable[[str, str, str, dict[str, Any] | None], Awaitable[None]]

CORE_TARGETS = {
    "zh-CN": "CHS",
    "zh-TW": "CHT",
    "en": "ENG",
    "ja": "JPN",
    "ko": "KOR",
}
STAGE_PROGRESS = {
    "running_pre_translation_hooks": 0.02,
    "mps-fallback": 0.02,
    "detection": 0.12,
    "ocr": 0.30,
    "textline_merge": 0.43,
    "translating": 0.52,
    "after-translating": 0.66,
    "mask-generation": 0.72,
    "inpainting": 0.80,
    "rendering": 0.92,
    "finished": 0.98,
    "saved": 1.0,
    "skip-no-regions": 1.0,
    "skip-no-text": 1.0,
}


class CoreUnavailableError(RuntimeError):
    pass


def _import_core():
    if not (VENDOR_CORE_DIR / "manga_translator" / "__init__.py").exists():
        raise CoreUnavailableError(
            "漫画处理核心不存在，请运行“首次安装.command”初始化 Git 子模块"
        )
    vendor = str(VENDOR_CORE_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    try:
        from manga_translator import Config, MangaTranslator
        from manga_translator.config import Detector, Inpainter, Ocr, Translator
        from manga_translator.detection.ctd import ComicTextDetector
        from manga_translator.detection import unload as unload_detection
        from manga_translator.inpainting import unload as unload_inpainting
        from manga_translator.ocr import unload as unload_ocr
        from manga_translator.utils import (
            ModelWrapper,
            Quadrilateral,
            TextBlock,
            dump_image,
            load_image,
        )
    except Exception as exc:
        raise CoreUnavailableError(
            f"漫画处理核心依赖尚未安装，请运行“首次安装.command”：{exc}"
        ) from exc

    ModelWrapper._MODEL_DIR = str(MODEL_DIR)
    cpu_mapping = ComicTextDetector._MODEL_MAPPING.get("model-cpu")
    if cpu_mapping:
        ComicTextDetector._MODEL_MAPPING = {"model-cpu": cpu_mapping}
    return {
        "Config": Config,
        "MangaTranslator": MangaTranslator,
        "Detector": Detector,
        "Inpainter": Inpainter,
        "Ocr": Ocr,
        "Translator": Translator,
        "Quadrilateral": Quadrilateral,
        "TextBlock": TextBlock,
        "dump_image": dump_image,
        "load_image": load_image,
        "unload_detection": unload_detection,
        "unload_inpainting": unload_inpainting,
        "unload_ocr": unload_ocr,
    }


def _font_path() -> str:
    candidates = [
        VENDOR_CORE_DIR / "fonts" / "NotoSansMonoCJK-VF.ttf.ttc",
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    ]
    return str(next((path for path in candidates if path.exists()), candidates[-1]))


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _enum_value(value: Any, fallback: str = "auto") -> str:
    if value is None:
        return fallback
    return str(getattr(value, "value", value))


def _normalize_optional_font_size(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        normalized = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(6, min(300, normalized))


def _rgb_hex(value: Any) -> str:
    channels = [max(0, min(255, int(channel))) for channel in value]
    return "#" + "".join(f"{channel:02X}" for channel in channels[:3])


def _public_direction(value: Any) -> str:
    return {
        "h": "horizontal",
        "hr": "horizontal",
        "v": "vertical",
        "vr": "vertical",
    }.get(_enum_value(value), _enum_value(value))


def _core_direction(value: str) -> str:
    return {"horizontal": "h", "vertical": "v"}.get(value, value)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _is_mps_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "mps",
            "metal",
            "bfloat16 is not supported",
            "not implemented for",
        )
    )


async def _reset_model_caches_for_cpu() -> None:
    core = _import_core()
    await core["unload_detection"](core["Detector"].ctd)
    await core["unload_ocr"](core["Ocr"].ocr48px)
    await core["unload_inpainting"](core["Inpainter"].lama_large)


def trim_runtime_memory(release_accelerator_cache: bool = True) -> None:
    gc.collect()
    if not release_accelerator_cache:
        return
    try:
        import torch

        if (
            "PYTEST_CURRENT_TEST" not in os.environ
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
        ):
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _copy_image_array(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    return np.ascontiguousarray(np.array(image, copy=True))


def _clip_bbox(bbox: Any, width: int, height: int) -> list[int]:
    if bbox is None or len(bbox) != 4:
        raise ValueError("文本框坐标必须包含 4 个数值")
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("文本框尺寸过小，无法处理")
    return [x1, y1, x2, y2]


def _image_size_from_payload(payload: dict[str, Any]) -> tuple[int, int]:
    image = payload.get("img_rgb")
    if image is None:
        raise RuntimeError("重新处理上下文缺少原图数据")
    height, width = image.shape[:2]
    return int(width), int(height)


def _region_bbox(region: Any) -> list[int]:
    return [int(value) for value in region.xyxy]


def _update_bbox_fields(region: Any, ocr_bbox: list[int], render_bbox: list[int]) -> None:
    region.ocr_bbox = [int(value) for value in ocr_bbox]
    region.render_bbox = [int(value) for value in render_bbox]


def _clear_geometry_cache(region: Any) -> None:
    for name in (
        "xyxy",
        "xywh",
        "center",
        "unrotated_polygons",
        "unrotated_min_rect",
        "min_rect",
        "polygon_aspect_ratio",
        "unrotated_size",
    ):
        getattr(region, "__dict__", {}).pop(name, None)
    lines = getattr(region, "lines", None)
    if lines is None:
        return
    for line in lines:
        for name in ("structure", "valid", "aspect_ratio", "font_size", "xyxy", "aabb"):
            getattr(line, "__dict__", {}).pop(name, None)


def _bbox_to_polygon(bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


def _set_region_geometry(region: Any, bbox: list[int]) -> None:
    region.lines = np.array([_bbox_to_polygon(bbox)], dtype=np.int32)
    region._bounding_rect = None
    _clear_geometry_cache(region)


def _bbox_from_update(update: dict[str, Any], name: str, fallback: list[int]) -> list[int]:
    return update.get(name) or update.get("bbox") or fallback


def _set_region_font_preference(region: Any, value: Any) -> None:
    normalized = _normalize_optional_font_size(value)
    region._font_size_user = normalized
    region._font_size_auto = normalized is None


def _region_font_preference(region: Any) -> int | None:
    if getattr(region, "_font_size_auto", False):
        return None
    preferred = _normalize_optional_font_size(getattr(region, "_font_size_user", None))
    if preferred is not None:
        return preferred
    return _normalize_optional_font_size(getattr(region, "font_size", None))


def _normalize_region_updates(
    updates: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for new_index, update in enumerate(updates):
        fallback = update.get("bbox") or update.get("render_bbox") or update.get("ocr_bbox")
        ocr_bbox = _clip_bbox(_bbox_from_update(update, "ocr_bbox", fallback), width, height)
        render_bbox = _clip_bbox(
            _bbox_from_update(update, "render_bbox", ocr_bbox), width, height
        )
        normalized.append(
            {
                **update,
                "index": new_index,
                "ocr_bbox": ocr_bbox,
                "render_bbox": render_bbox,
                "bbox": render_bbox,
                "enabled": bool(update.get("enabled", True)),
                "translation": sanitize_translation_text(update.get("translation", "")),
                "font_size": _normalize_optional_font_size(update.get("font_size")),
            }
        )
    return normalized


def _fill_update_bbox_defaults(
    updates: list[dict[str, Any]], existing_regions: list[Any]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, update in enumerate(updates):
        item = dict(update)
        if index < len(existing_regions):
            region = existing_regions[index]
            bbox = _region_bbox(region)
            if not item.get("ocr_bbox"):
                item["ocr_bbox"] = getattr(region, "ocr_bbox", bbox)
            if not item.get("render_bbox"):
                item["render_bbox"] = getattr(region, "render_bbox", bbox)
            if not item.get("bbox"):
                item["bbox"] = item["render_bbox"]
        result.append(item)
    return result


def _region_enabled(region: Any) -> bool:
    return bool(getattr(region, "enabled", True))


def _config_section_value(
    config: Any, section: str, name: str, fallback: str
) -> str:
    section_value = getattr(config, section, None)
    return _enum_value(getattr(section_value, name, None), fallback)


def _config_target_lang(config: Any) -> str:
    target = _config_section_value(config, "translator", "target_lang", "CHS")
    return target if target and target != "None" else "CHS"


def _config_render_line_spacing(config: Any) -> float | None:
    render = getattr(config, "render", None)
    value = getattr(render, "line_spacing", None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _config_render_hyphenate(config: Any) -> bool:
    render = getattr(config, "render", None)
    return not bool(getattr(render, "no_hyphenation", False))


def _config_render_direction(config: Any) -> str:
    return _config_section_value(config, "render", "direction", "auto")


def _config_render_alignment(config: Any) -> str:
    return _config_section_value(config, "render", "alignment", "auto")


def _prepare_region_for_render(
    region: Any,
    config: Any,
    direction: str | None = None,
    alignment: str | None = None,
) -> None:
    region.target_lang = _config_target_lang(config)
    selected_direction = (
        direction
        if direction is not None
        else _enum_value(
            getattr(region, "_direction", _config_render_direction(config))
        )
    )
    region._direction = _core_direction(
        selected_direction or _config_render_direction(config)
    )
    region._alignment = (
        alignment
        if alignment is not None
        else _enum_value(
            getattr(region, "_alignment", _config_render_alignment(config))
        )
    )
    region.adjust_bg_color = False


def _load_text_render():
    vendor = str(VENDOR_CORE_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from manga_translator.rendering import text_render

    return text_render


def _fit_region_font_size(region: Any, config: Any) -> int:
    preferred = _region_font_preference(region)
    width, height = getattr(region, "unrotated_size", (0, 0))
    width = max(2, int(round(width)))
    height = max(2, int(round(height)))
    upper = preferred if preferred is not None else max(6, min(width, height))
    upper = max(6, upper)
    text = getattr(region, "get_translation_for_rendering", lambda: getattr(region, "translation", ""))()
    if not text or not str(text).strip():
        return upper

    text_render = _load_text_render()
    if hasattr(text_render, "set_font"):
        try:
            text_render.set_font(_font_path())
        except Exception:
            return upper
    hyphenate = _config_render_hyphenate(config)
    line_spacing = _config_render_line_spacing(config)
    target_lang = getattr(region, "target_lang", "en_US")

    def fits(font_size: int) -> bool:
        if getattr(region, "vertical", False):
            columns, heights = text_render.calc_vertical(font_size, text, height)
            spacing_x = int(font_size * (line_spacing or 0.2))
            used_width = font_size * len(columns) + spacing_x * max(0, len(columns) - 1)
            used_height = max(heights) if heights else 0
            return used_width <= width and used_height <= height

        lines, widths = text_render.calc_horizontal(
            font_size,
            text,
            width,
            height,
            target_lang,
            hyphenate,
        )
        spacing_y = int(font_size * (line_spacing or 0.01))
        used_height = font_size * len(lines) + spacing_y * max(0, len(lines) - 1)
        used_width = max(widths) if widths else 0
        return used_width <= width and used_height <= height

    best = 6
    low, high = 6, upper
    try:
        if not fits(low):
            return low
        while low <= high:
            middle = (low + high) // 2
            if fits(middle):
                best = middle
                low = middle + 1
            else:
                high = middle - 1
    except Exception:
        return low
    return best


def _make_text_region(
    core: dict[str, Any],
    bbox: list[int],
    text: str,
    translation: str = "",
    foreground: tuple[int, int, int] = (0, 0, 0),
    outline: tuple[int, int, int] = (255, 255, 255),
    font_size: int | None = None,
) -> Any:
    block = core["TextBlock"](
        [_bbox_to_polygon(bbox)],
        texts=[text or ""],
        font_size=font_size or max(6, min(bbox[2] - bbox[0], bbox[3] - bbox[1])),
        translation=sanitize_translation_text(translation),
        fg_color=foreground,
        bg_color=outline,
    )
    block.text_raw = block.text
    block.adjust_bg_color = False
    _update_bbox_fields(block, bbox, bbox)
    _set_region_font_preference(block, font_size)
    return block


def _make_mask_from_regions(
    shape: tuple[int, int, int] | tuple[int, int], regions: list[Any]
) -> np.ndarray:
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        if not _region_enabled(region):
            continue
        bbox = getattr(region, "ocr_bbox", _region_bbox(region))
        cv2.fillPoly(mask, [_bbox_to_polygon(bbox)], 255)
    return mask


def _ensure_payload_region_boxes(payload: dict[str, Any]) -> dict[str, Any]:
    width, height = _image_size_from_payload(payload)
    config = payload.get("config")
    for region in payload.get("text_regions", []) or []:
        xyxy = _clip_bbox(_region_bbox(region), width, height)
        ocr_bbox = _clip_bbox(getattr(region, "ocr_bbox", xyxy), width, height)
        render_bbox = _clip_bbox(getattr(region, "render_bbox", xyxy), width, height)
        _update_bbox_fields(region, ocr_bbox, render_bbox)
        _set_region_font_preference(region, _region_font_preference(region))
        if config is not None:
            _prepare_region_for_render(region, config)
    return payload


def _extract_gimp_clean_image(ctx: Any) -> np.ndarray | None:
    gimp_mask = getattr(ctx, "gimp_mask", None)
    if isinstance(gimp_mask, np.ndarray) and gimp_mask.ndim == 3 and gimp_mask.shape[2] >= 3:
        return np.ascontiguousarray(gimp_mask[..., :3][:, :, ::-1].copy())
    return None


def _extract_clean_inpainted_image(ctx: Any) -> np.ndarray:
    gimp_clean = _extract_gimp_clean_image(ctx)
    if gimp_clean is not None:
        return gimp_clean
    if getattr(ctx, "img_inpainted", None) is None:
        raise RuntimeError("缺少可重新嵌字的去字底图")
    return _copy_image_array(ctx.img_inpainted)


def _clean_sidecar_path(context_path: Path) -> Path:
    return context_path.with_suffix(".clean.png")


def _image_array_for_png(image: Any) -> np.ndarray | None:
    array = _copy_image_array(image)
    if array is None:
        return None
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        return None
    if array.shape[2] > 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _save_clean_sidecar(context_path: Path, image: Any) -> None:
    clean = _image_array_for_png(image)
    if clean is None:
        return
    path = _clean_sidecar_path(context_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(clean).save(path)


def _load_clean_sidecar(context_path: Path, expected_shape: tuple[int, ...]) -> np.ndarray | None:
    path = _clean_sidecar_path(context_path)
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            clean = np.array(image.convert("RGB"))
    except Exception:
        return None
    if clean.shape[:2] != expected_shape[:2]:
        return None
    return np.ascontiguousarray(clean)


def _build_minimal_rerender_payload(ctx: Any, config: Any) -> dict[str, Any]:
    payload = {
        "config": config,
        "text_regions": list(getattr(ctx, "text_regions", []) or []),
        "img_rgb": _copy_image_array(ctx.img_rgb),
        "img_inpainted": _copy_image_array(ctx.img_rgb),
        "img_alpha": getattr(ctx, "img_alpha", None),
        "clean_image_trusted": True,
    }
    return _ensure_payload_region_boxes(payload)


def _build_rerender_payload(ctx: Any, config: Any) -> dict[str, Any]:
    text_regions = list(getattr(ctx, "text_regions", []) or [])
    if not text_regions:
        return _build_minimal_rerender_payload(ctx, config)
    gimp_clean = _extract_gimp_clean_image(ctx)
    if gimp_clean is not None:
        img_inpainted = gimp_clean
        clean_image_trusted = True
    elif getattr(ctx, "img_inpainted", None) is not None:
        img_inpainted = _copy_image_array(ctx.img_inpainted)
        clean_image_trusted = False
    else:
        return _build_minimal_rerender_payload(ctx, config)
    payload = {
        "config": config,
        "text_regions": text_regions,
        "img_rgb": _copy_image_array(ctx.img_rgb),
        "img_inpainted": img_inpainted,
        "img_alpha": getattr(ctx, "img_alpha", None),
        "clean_image_trusted": clean_image_trusted,
    }
    return _ensure_payload_region_boxes(payload)


def _normalize_rerender_payload(
    payload: dict[str, Any], context_path: Path | None = None
) -> dict[str, Any]:
    if "ctx" in payload:
        legacy_ctx = payload["ctx"]
        normalized = _build_rerender_payload(legacy_ctx, payload["config"])
    else:
        required = {"config", "text_regions", "img_rgb", "img_inpainted"}
        missing = required - set(payload)
        if missing:
            raise RuntimeError(f"重新嵌字上下文缺少必要字段：{', '.join(sorted(missing))}")
        normalized = _ensure_payload_region_boxes(payload)
    if context_path is not None:
        sidecar = _load_clean_sidecar(context_path, normalized["img_rgb"].shape)
        if sidecar is not None:
            normalized["img_inpainted"] = sidecar
            normalized["clean_image_trusted"] = True
    return normalized


def _save_rerender_payload(context_path: Path, payload: dict[str, Any]) -> None:
    context_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    clean = _image_array_for_png(payload.get("img_inpainted"))
    if clean is not None:
        payload["img_inpainted"] = clean
        payload["clean_image_trusted"] = True
        payload["clean_image_sidecar"] = _clean_sidecar_path(context_path).name
        _save_clean_sidecar(context_path, clean)
    with context_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


async def _rebuild_clean_inpainted_image(
    translator: Any,
    config: Any,
    payload: dict[str, Any],
    regions: list[Any],
) -> np.ndarray:
    enabled_regions = [region for region in regions if _region_enabled(region)]
    if not enabled_regions:
        return _copy_image_array(payload["img_rgb"])

    for region in enabled_regions:
        _set_region_geometry(region, getattr(region, "ocr_bbox", _region_bbox(region)))
        _prepare_region_for_render(region, config)
    ctx = SimpleNamespace(
        text_regions=enabled_regions,
        img_rgb=payload["img_rgb"],
        img_inpainted=None,
        img_alpha=payload.get("img_alpha"),
        mask_raw=_make_mask_from_regions(payload["img_rgb"].shape, enabled_regions),
        mask=None,
        render_mask=None,
    )
    ctx.mask = await translator._run_mask_refinement(config, ctx)
    clean = _image_array_for_png(await translator._run_inpainting(config, ctx))
    if clean is None:
        raise RuntimeError("重新生成干净底图失败")
    return clean


async def _clean_inpainted_for_rerender(
    translator: Any,
    config: Any,
    payload: dict[str, Any],
    regions: list[Any],
) -> np.ndarray:
    if payload.get("clean_image_trusted"):
        clean = _image_array_for_png(payload.get("img_inpainted"))
        if clean is not None:
            return clean
    return await _rebuild_clean_inpainted_image(translator, config, payload, regions)


def serialize_regions(regions: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, region in enumerate(regions or []):
        foreground, outline = region.get_font_colors()
        font_size = _region_font_preference(region)
        xyxy = _region_bbox(region)
        ocr_bbox = [
            int(value) for value in getattr(region, "ocr_bbox", xyxy)
        ]
        render_bbox = [
            int(value) for value in getattr(region, "render_bbox", xyxy)
        ]
        result.append(
            {
                "index": index,
                "bbox": render_bbox,
                "ocr_bbox": ocr_bbox,
                "render_bbox": render_bbox,
                "enabled": _region_enabled(region),
                "text": region.text,
                "translation": sanitize_translation_text(
                    getattr(region, "translation", "")
                ),
                "font_size": font_size,
                "direction": _public_direction(
                    getattr(region, "_direction", "auto")
                ),
                "alignment": _enum_value(getattr(region, "_alignment", "auto")),
                "foreground": _rgb_hex(foreground),
                "outline": _rgb_hex(outline),
            }
        )
    return result


def _save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.convert("RGB").save(path, quality=95, optimize=True)
    else:
        image.save(path)


class CoreEngine:
    def __init__(
        self,
        provider: TranslatorProvider,
        source_language: str,
        target_language: str,
        polish_provider: OllamaProvider | None,
        render_direction: str,
        render_alignment: str,
        font_size: int | None,
        progress_callback: ProgressCallback,
        log_callback: LogCallback,
    ):
        self.provider = provider
        self.source_language = source_language
        self.target_language = target_language
        self.polish_provider = polish_provider
        self.render_direction = render_direction
        self.render_alignment = render_alignment
        self.font_size = font_size
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.translation_cache: dict[tuple[str, ...], list[str]] = {}

    def _build(self, use_gpu: bool):
        core = _import_core()
        provider = self.provider
        source = self.source_language
        target = self.target_language
        polish = self.polish_provider
        log_callback = self.log_callback
        translation_cache = self.translation_cache
        base_class = core["MangaTranslator"]
        provider.set_log_callback(log_callback)
        if polish:
            polish.set_log_callback(log_callback)

        class WebMangaTranslator(base_class):
            def _setup_log_file(self):
                self._log_file_path = None

            async def _detector_cleanup_job(self):
                # Upstream loops forever even when TTL is disabled.
                return

            async def _run_detection(self, config, ctx):
                # The installed CTD ONNX model uses the CPU backend.
                device = self.device
                self.device = "cpu"
                try:
                    return await super()._run_detection(config, ctx)
                finally:
                    self.device = device

            async def _dispatch_with_context(self, config, texts, ctx):
                await log_callback(
                    "INFO",
                    "ocr",
                    f"OCR 提取到 {len(texts)} 个文本区域",
                    {"texts": texts},
                )
                cache_key = tuple(texts)
                if cache_key in translation_cache:
                    translations = translation_cache[cache_key]
                    await log_callback(
                        "INFO",
                        "translation",
                        "CPU 重试复用已完成的译文",
                        None,
                    )
                else:
                    translations = await provider.translate(texts, source, target)
                    if polish:
                        await log_callback(
                            "INFO", "polish", "正在使用 Ollama 润色在线译文", None
                        )
                        translations = await polish.polish(texts, translations, target)
                    translation_cache[cache_key] = translations
                await log_callback(
                    "INFO",
                    "translation",
                    f"完成 {len(translations)} 个文本区域的翻译",
                    {
                        "pairs": [
                            {"source": text, "translation": translation}
                            for text, translation in zip(texts, translations)
                        ]
                    },
                )
                return translations

            async def _run_text_translation(self, config, ctx):
                regions = await super()._run_text_translation(config, ctx)
                direction = _enum_value(config.render.direction)
                if direction != "auto":
                    for region in regions:
                        region._direction = _core_direction(direction)
                return regions

        translator = WebMangaTranslator(
            {
                "kernel_size": 3,
                "use_gpu": use_gpu,
                "model_dir": str(MODEL_DIR),
                "font_path": _font_path(),
                "models_ttl": 0,
                "verbose": False,
                "ignore_errors": False,
            }
        )
        return core, translator

    def _config(self, core, use_gpu: bool):
        return core["Config"](
            detector={
                "detector": core["Detector"].ctd,
                "detection_size": 2048,
                "text_threshold": 0.5,
                "box_threshold": 0.7,
                "unclip_ratio": 2.3,
            },
            ocr={"ocr": core["Ocr"].ocr48px, "min_text_length": 1},
            inpainter={
                "inpainter": core["Inpainter"].lama_large,
                "inpainting_size": 2048,
                "inpainting_precision": "bf16" if use_gpu else "fp32",
            },
            translator={
                "translator": core["Translator"].original,
                "target_lang": CORE_TARGETS[self.target_language],
                "no_text_lang_skip": True,
                "enable_post_translation_check": False,
            },
            render={
                "rtl": self.source_language == "ja",
                "direction": self.render_direction,
                "alignment": self.render_alignment,
                "font_size": self.font_size,
                "font_size_minimum": -1,
                "no_hyphenation": self.target_language.startswith("zh"),
            },
            mask_dilation_offset=20,
            kernel_size=3,
            force_simple_sort=False,
        )

    def _join_manual_ocr_texts(self, texts: list[str]) -> str:
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if self.source_language == "ja":
            return "".join(cleaned)
        return " ".join(cleaned)

    def _manual_ocr_text_value(self, text: str) -> int:
        return len(re.sub(r"[\s　、。,.!?！？…:：;；「」『』（）()【】\[\]<>/\\|_-]", "", text or ""))

    def _manual_ocr_text_units(self, text: str) -> list[str]:
        cleaned = re.sub(r"[\s　、。,.!?！？…:：;；「」『』（）()【】\[\]<>/\\|_-]", "", text or "")
        return [char for char in cleaned if char]

    def _manual_ocr_char_recall(self, new_text: str, old_text: str) -> float:
        old_units = self._manual_ocr_text_units(old_text)
        new_units = self._manual_ocr_text_units(new_text)
        if not old_units or not new_units:
            return 1.0 if old_units == new_units else 0.0
        remaining: dict[str, int] = {}
        for char in new_units:
            remaining[char] = remaining.get(char, 0) + 1
        matched = 0
        for char in old_units:
            count = remaining.get(char, 0)
            if count:
                matched += 1
                remaining[char] = count - 1
        return matched / len(old_units)

    def _manual_ocr_is_low_quality(self, new_text: str, old_text: str | None) -> bool:
        old_clean = (old_text or "").strip()
        if not old_clean:
            return False
        new_clean = (new_text or "").strip()
        if not new_clean:
            return True
        if any(marker in new_clean for marker in ("<", ">", "�")):
            return True
        old_score = self._manual_ocr_text_value(old_clean)
        new_score = self._manual_ocr_text_value(new_clean)
        if old_score >= 4 and new_score < max(2, int(old_score * 0.55)):
            return True
        if self.source_language == "ja" and old_score >= 8:
            recall = self._manual_ocr_char_recall(new_clean, old_clean)
            if recall < 0.68 and new_score <= int(old_score * 1.25):
                return True
        return False

    def _textline_center(self, textline: Any) -> tuple[float, float]:
        if hasattr(textline, "pts"):
            points = np.asarray(textline.pts, dtype=np.float32).reshape(-1, 2)
            return float(points[:, 0].mean()), float(points[:, 1].mean())
        center = getattr(textline, "center", None)
        if center is not None:
            return float(center[0]), float(center[1])
        xyxy = getattr(textline, "xyxy", [0, 0, 0, 0])
        return (float(xyxy[0] + xyxy[2]) / 2, float(xyxy[1] + xyxy[3]) / 2)

    def _textline_area(self, textline: Any) -> float:
        if hasattr(textline, "pts"):
            points = np.asarray(textline.pts, dtype=np.float32).reshape(-1, 2)
            return float(max(1.0, cv2.contourArea(points)))
        xyxy = getattr(textline, "xyxy", [0, 0, 0, 0])
        return float(max(1.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])))

    def _sort_manual_ocr_items(self, items: list[Any]) -> list[Any]:
        if self.source_language == "ja":
            return sorted(items, key=lambda item: (-self._textline_center(item)[0], self._textline_center(item)[1]))
        return sorted(items, key=lambda item: (self._textline_center(item)[1], self._textline_center(item)[0]))

    def _filter_detected_textlines_in_user_bbox(
        self,
        textlines: list[Any],
        user_bbox: list[int],
        crop_origin: tuple[int, int],
        image_shape: tuple[int, ...],
    ) -> list[Any]:
        if not textlines:
            return []
        ux1, uy1, ux2, uy2 = user_bbox
        crop_x, crop_y = crop_origin
        image_area = max(1, image_shape[0] * image_shape[1])
        filtered: list[Any] = []
        for line in textlines:
            cx, cy = self._textline_center(line)
            gx = cx + crop_x
            gy = cy + crop_y
            if not (ux1 <= gx <= ux2 and uy1 <= gy <= uy2):
                continue
            if self._textline_area(line) / image_area > 0.85:
                continue
            filtered.append(line)
        return self._sort_manual_ocr_items(filtered)

    async def _log_manual_ocr_source(
        self, source: str, text: str, old_text: str | None = None
    ) -> None:
        details: dict[str, Any] = {"texts": [text]} if text else {}
        if old_text:
            details["old_text"] = old_text
        await self.log_callback(
            "INFO",
            "ocr",
            f"人工 OCR 使用{source}",
            details or None,
        )

    def _offset_textline(self, core: dict[str, Any], textline: Any, x: int, y: int) -> Any:
        if not hasattr(textline, "pts"):
            return textline
        offset = np.array([x, y], dtype=np.int32)
        points = np.asarray(textline.pts, dtype=np.int32) + offset
        text = getattr(textline, "text", "")
        prob = getattr(textline, "prob", 1)
        foreground = [
            int(value) for value in getattr(textline, "fg_colors", (0, 0, 0))[:3]
        ]
        background = [
            int(value)
            for value in getattr(textline, "bg_colors", (255, 255, 255))[:3]
        ]
        try:
            return core["Quadrilateral"](
                points,
                text,
                prob,
                *foreground,
                *background,
            )
        except TypeError:
            return core["Quadrilateral"](points, text, prob)

    async def _ocr_user_bbox(
        self,
        core: dict[str, Any],
        translator: Any,
        config: Any,
        image: np.ndarray,
        bbox: list[int],
        old_text: str | None = None,
    ) -> tuple[str, tuple[Any, Any] | None]:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = _clip_bbox(bbox, width, height)
        pad = max(4, int(round(max(x2 - x1, y2 - y1) * 0.04)))
        crop_x1, crop_y1, crop_x2, crop_y2 = _clip_bbox(
            [x1 - pad, y1 - pad, x2 + pad, y2 + pad], width, height
        )
        crop = np.ascontiguousarray(image[crop_y1:crop_y2, crop_x1:crop_x2])
        crop_height, crop_width = crop.shape[:2]
        crop_ctx = SimpleNamespace(
            img_rgb=crop,
            textlines=[],
            mask_raw=None,
            mask=None,
        )
        detected_textlines: list[Any] = []
        try:
            detected_textlines, _, _ = await translator._run_detection(config, crop_ctx)
        except Exception as exc:
            await self.log_callback(
                "WARNING",
                "ocr",
                f"框内文字检测失败，改用单框 OCR：{exc}",
                None,
            )
        detected_textlines = detected_textlines or []
        detected_textlines = self._filter_detected_textlines_in_user_bbox(
            detected_textlines,
            [x1, y1, x2, y2],
            (crop_x1, crop_y1),
            crop.shape,
        )

        crop_ctx.textlines = detected_textlines
        ocr_textlines = await translator._run_ocr(config, crop_ctx) if detected_textlines else []
        single_box_used = False
        if not ocr_textlines:
            crop_ctx.textlines = [
                core["Quadrilateral"](
                    _bbox_to_polygon([x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1]),
                    "",
                    1,
                )
            ]
            ocr_textlines = await translator._run_ocr(config, crop_ctx)
            single_box_used = True

        ocr_textlines = [line for line in ocr_textlines if getattr(line, "text", "").strip()]
        if not ocr_textlines:
            if old_text and old_text.strip():
                await self._log_manual_ocr_source("复用旧文本", old_text, old_text)
                return old_text.strip(), None
            return "", None

        global_textlines = [
            self._offset_textline(core, line, crop_x1, crop_y1) for line in ocr_textlines
        ]
        merge_textlines = [line for line in global_textlines if hasattr(line, "pts")]
        if not merge_textlines:
            color_source = ocr_textlines[0]
            text = self._join_manual_ocr_texts(
                [getattr(line, "text", "") for line in self._sort_manual_ocr_items(ocr_textlines)]
            )
            colors = (
                getattr(color_source, "fg_colors", (0, 0, 0)),
                getattr(color_source, "bg_colors", (255, 255, 255)),
            )
            if self._manual_ocr_is_low_quality(text, old_text):
                await self._log_manual_ocr_source("复用旧文本", old_text or "", old_text)
                return (old_text or "").strip(), colors
            await self._log_manual_ocr_source("单框兜底" if single_box_used else "框内检测", text, old_text)
            return text, colors

        merge_textlines = self._sort_manual_ocr_items(merge_textlines)
        merge_ctx = SimpleNamespace(textlines=merge_textlines, img_rgb=image)
        try:
            merged_regions = await translator._run_textline_merge(config, merge_ctx)
        except Exception as exc:
            await self.log_callback(
                "WARNING",
                "textline_merge",
                f"框内文字合并失败，使用 OCR 行顺序：{exc}",
                None,
            )
            merged_regions = []

        if merged_regions:
            merged_regions = self._sort_manual_ocr_items(merged_regions)
            color_source = merged_regions[0]
            text = self._join_manual_ocr_texts(
                [getattr(region, "text", "") for region in merged_regions]
            )
            colors = color_source.get_font_colors()
            if self._manual_ocr_is_low_quality(text, old_text):
                await self._log_manual_ocr_source("复用旧文本", old_text or "", old_text)
                return (old_text or "").strip(), colors
            await self._log_manual_ocr_source("框内检测", text, old_text)
            return text, colors

        color_source = merge_textlines[0]
        text = self._join_manual_ocr_texts(
            [getattr(line, "text", "") for line in merge_textlines]
        )
        colors = (
            getattr(color_source, "fg_colors", (0, 0, 0)),
            getattr(color_source, "bg_colors", (255, 255, 255)),
        )
        if self._manual_ocr_is_low_quality(text, old_text):
            await self._log_manual_ocr_source("复用旧文本", old_text or "", old_text)
            return (old_text or "").strip(), colors
        await self._log_manual_ocr_source("单框兜底" if single_box_used else "框内检测", text, old_text)
        return text, colors

    async def _translate_manual_texts(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        self.provider.set_log_callback(self.log_callback)
        if self.polish_provider:
            self.polish_provider.set_log_callback(self.log_callback)
        cache_key = tuple(texts)
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        translations = await self.provider.translate(
            texts, self.source_language, self.target_language
        )
        if self.polish_provider:
            await self.log_callback(
                "INFO", "polish", "正在使用 Ollama 润色人工 OCR 译文", None
            )
            translations = await self.polish_provider.polish(
                texts, translations, self.target_language
            )
        translations = [sanitize_translation_text(value) for value in translations]
        self.translation_cache[cache_key] = translations
        await self.log_callback(
            "INFO",
            "translation",
            f"人工框重处理完成 {len(translations)} 个文本区域的翻译",
            {
                "pairs": [
                    {"source": text, "translation": translation}
                    for text, translation in zip(texts, translations)
                ]
            },
        )
        return translations

    async def process(
        self,
        input_path: Path,
        output_path: Path,
        context_path: Path,
        regions_path: Path,
    ) -> list[dict[str, Any]]:
        use_gpu = _mps_available()
        try:
            return await self._process_once(
                input_path,
                output_path,
                context_path,
                regions_path,
                use_gpu=use_gpu,
            )
        except Exception as exc:
            if not use_gpu or not _is_mps_error(exc):
                raise
            await self.log_callback(
                "WARNING",
                "mps-fallback",
                f"MPS 处理失败，正在使用 CPU 重新处理：{exc}",
                None,
            )
            await self.progress_callback("mps-fallback", STAGE_PROGRESS["mps-fallback"])
            await _reset_model_caches_for_cpu()
            return await self._process_once(
                input_path,
                output_path,
                context_path,
                regions_path,
                use_gpu=False,
            )

    async def reprocess_regions(
        self,
        input_path: Path,
        output_path: Path,
        context_path: Path,
        regions_path: Path,
        updates: list[dict[str, Any]],
        changed_indices: list[int],
    ) -> list[dict[str, Any]]:
        use_gpu = _mps_available()
        try:
            return await self._reprocess_regions_once(
                input_path,
                output_path,
                context_path,
                regions_path,
                updates,
                changed_indices,
                use_gpu=use_gpu,
            )
        except Exception as exc:
            if not use_gpu or not _is_mps_error(exc):
                raise
            await self.log_callback(
                "WARNING",
                "mps-fallback",
                f"MPS 人工重处理失败，正在使用 CPU 重试：{exc}",
                None,
            )
            await _reset_model_caches_for_cpu()
            return await self._reprocess_regions_once(
                input_path,
                output_path,
                context_path,
                regions_path,
                updates,
                changed_indices,
                use_gpu=False,
            )

    async def _process_once(
        self,
        input_path: Path,
        output_path: Path,
        context_path: Path,
        regions_path: Path,
        use_gpu: bool,
    ) -> list[dict[str, Any]]:
        core, translator = self._build(use_gpu)
        config = self._config(core, use_gpu)
        image = None
        ctx = None

        def hook(state: str, finished: bool = False):
            stage = state.split(":", 1)[0]
            callback = (
                self.progress_callback(stage, STAGE_PROGRESS[stage])
                if stage in STAGE_PROGRESS
                else _maybe_await(None)
            )
            return asyncio.create_task(callback)

        translator.add_progress_hook(hook)
        try:
            with Image.open(input_path) as source:
                image = source.convert("RGBA") if source.mode == "RGBA" else source.convert("RGB")
            ctx = await translator.translate(image, config)
            if not ctx.result:
                raise RuntimeError("核心处理完成但未生成输出图片")
            rerender_payload = _build_rerender_payload(ctx, config)
            _save_image(ctx.result, output_path)
            regions = serialize_regions(ctx.text_regions)
            _save_rerender_payload(context_path, rerender_payload)
            regions_path.parent.mkdir(parents=True, exist_ok=True)
            regions_path.write_text(
                json.dumps(regions, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            await self.progress_callback("saved", 1.0)
            return regions
        finally:
            if ctx is not None:
                for field in (
                    "gimp_mask",
                    "img_colorized",
                    "img_inpainted",
                    "img_rendered",
                    "mask",
                    "mask_raw",
                    "result",
                    "textlines",
                    "upscaled",
                ):
                    if field in ctx:
                        ctx[field] = None
            trim_runtime_memory()

    async def _reprocess_regions_once(
        self,
        input_path: Path,
        output_path: Path,
        context_path: Path,
        regions_path: Path,
        updates: list[dict[str, Any]],
        changed_indices: list[int],
        use_gpu: bool,
    ) -> list[dict[str, Any]]:
        core, translator = self._build(use_gpu)
        config = self._config(core, use_gpu)
        with context_path.open("rb") as file:
            payload = _normalize_rerender_payload(pickle.load(file), context_path)
        width, height = _image_size_from_payload(payload)
        changed = set(changed_indices)
        existing_regions = payload.get("text_regions", []) or []
        updates = _normalize_region_updates(
            _fill_update_bbox_defaults(updates, existing_regions), width, height
        )

        all_regions: list[Any] = []
        reocr_slots: list[int] = []
        for update in updates:
            old_region = (
                existing_regions[update["index"]]
                if update["index"] < len(existing_regions)
                else None
            )
            if old_region is None:
                region = _make_text_region(
                    core,
                    update["ocr_bbox"],
                    update.get("text", ""),
                    update.get("translation", ""),
                    _hex_rgb(update["foreground"]),
                    _hex_rgb(update["outline"]),
                    update.get("font_size"),
                )
                changed.add(update["index"])
            else:
                region = old_region
                region.text = update.get("text", "")
                region.translation = sanitize_translation_text(
                    update.get("translation", "")
                )
                region._direction = _core_direction(update["direction"])
                region._alignment = update["alignment"]
                region.set_font_colors(
                    _hex_rgb(update["foreground"]),
                    _hex_rgb(update["outline"]),
                )
            _set_region_font_preference(region, update.get("font_size"))
            _prepare_region_for_render(
                region,
                config,
                update["direction"],
                update["alignment"],
            )
            region.enabled = update["enabled"]
            _update_bbox_fields(region, update["ocr_bbox"], update["render_bbox"])
            _set_region_geometry(region, update["ocr_bbox"])
            all_regions.append(region)
            if update["enabled"] and update["index"] in changed:
                reocr_slots.append(update["index"])

        ctx = SimpleNamespace(
            textlines=[],
            text_regions=[],
            img_rgb=payload["img_rgb"],
            img_inpainted=None,
            img_alpha=payload.get("img_alpha"),
            mask_raw=None,
            mask=None,
            render_mask=None,
        )
        if reocr_slots:
            await self.log_callback(
                "INFO",
                "ocr",
                f"正在重新识别 {len(reocr_slots)} 个人工 OCR 框",
                None,
            )
            texts_to_translate: list[str] = []
            translated_slots: list[int] = []
            for slot in reocr_slots:
                region = all_regions[slot]
                old_text = getattr(region, "text", "")
                text, colors = await self._ocr_user_bbox(
                    core,
                    translator,
                    config,
                    payload["img_rgb"],
                    getattr(region, "ocr_bbox", _region_bbox(region)),
                    old_text,
                )
                if not text:
                    region.text = ""
                    region.translation = ""
                    continue
                region.text = text
                region.text_raw = text
                if colors is not None:
                    region.set_font_colors(colors[0], colors[1])
                if text.strip():
                    texts_to_translate.append(text)
                    translated_slots.append(slot)
            translations = await self._translate_manual_texts(texts_to_translate)
            for slot, translation in zip(translated_slots, translations):
                all_regions[slot].translation = translation
            await self.log_callback(
                "INFO",
                "ocr",
                f"人工 OCR 重识别完成：{len(texts_to_translate)} 个有效文本区域",
                {"texts": texts_to_translate},
            )

        enabled_regions = [region for region in all_regions if _region_enabled(region)]
        for region in enabled_regions:
            _prepare_region_for_render(region, config)
            _set_region_geometry(region, getattr(region, "ocr_bbox", _region_bbox(region)))
        ctx.text_regions = enabled_regions
        if enabled_regions:
            ctx.mask_raw = _make_mask_from_regions(payload["img_rgb"].shape, enabled_regions)
            ctx.mask = await translator._run_mask_refinement(config, ctx)
            ctx.img_inpainted = await translator._run_inpainting(config, ctx)
            clean_inpainted = _image_array_for_png(ctx.img_inpainted)
            if clean_inpainted is None:
                raise RuntimeError("重新生成干净底图失败")
            ctx.img_inpainted = _copy_image_array(clean_inpainted)
            for region in enabled_regions:
                _set_region_geometry(
                    region, getattr(region, "render_bbox", _region_bbox(region))
                )
                _prepare_region_for_render(region, config)
                region.font_size = _fit_region_font_size(region, config)
            ctx.text_regions = enabled_regions
            rendered = await translator._run_text_rendering(config, ctx)
        else:
            clean_inpainted = _copy_image_array(payload["img_rgb"])
            ctx.img_inpainted = _copy_image_array(clean_inpainted)
            rendered = ctx.img_inpainted
        with Image.open(input_path) as source:
            base_image = source.copy()
        result = core["dump_image"](base_image, rendered, ctx.img_alpha)
        _save_image(result, output_path)
        regions = serialize_regions(all_regions)
        regions_path.write_text(
            json.dumps(regions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _save_rerender_payload(
            context_path,
            {
                "config": config,
                "text_regions": all_regions,
                "img_rgb": payload["img_rgb"],
                "img_inpainted": clean_inpainted,
                "img_alpha": payload.get("img_alpha"),
            },
        )
        trim_runtime_memory()
        return regions


async def rerender(
    context_path: Path,
    regions_path: Path,
    output_path: Path,
    input_path: Path,
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    core = _import_core()
    with context_path.open("rb") as file:
        payload = _normalize_rerender_payload(pickle.load(file), context_path)
    config = payload["config"]
    width, height = _image_size_from_payload(payload)
    all_regions = payload["text_regions"]
    updates = _normalize_region_updates(
        _fill_update_bbox_defaults(updates, all_regions), width, height
    )
    if len(updates) != len(all_regions):
        raise ValueError("提交的文本区域数量与已保存上下文不一致")

    for update in updates:
        region = all_regions[update["index"]]
        region.text = update["text"]
        region.translation = sanitize_translation_text(update["translation"])
        _set_region_font_preference(region, update.get("font_size"))
        region.set_font_colors(
            _hex_rgb(update["foreground"]),
            _hex_rgb(update["outline"]),
        )
        region.enabled = update["enabled"]
        _update_bbox_fields(region, update["ocr_bbox"], update["render_bbox"])
        _set_region_geometry(region, update["render_bbox"])
        _prepare_region_for_render(
            region,
            config,
            update["direction"],
            update["alignment"],
        )

    render_regions = [region for region in all_regions if _region_enabled(region)]

    class RenderOnlyTranslator(core["MangaTranslator"]):
        def _setup_log_file(self):
            self._log_file_path = None

    translator = RenderOnlyTranslator(
        {
            "kernel_size": 3,
            "use_gpu": True,
            "model_dir": str(MODEL_DIR),
            "font_path": _font_path(),
            "models_ttl": 0,
            "verbose": False,
            "ignore_errors": False,
        }
    )
    try:
        clean_inpainted = await _clean_inpainted_for_rerender(
            translator, config, payload, all_regions
        )
        for region in render_regions:
            _set_region_geometry(region, getattr(region, "render_bbox", _region_bbox(region)))
            _prepare_region_for_render(region, config)
            region.font_size = _fit_region_font_size(region, config)
        ctx = SimpleNamespace(
            text_regions=render_regions,
            img_rgb=payload["img_rgb"],
            img_inpainted=_copy_image_array(clean_inpainted),
            img_alpha=payload.get("img_alpha"),
            render_mask=None,
        )
        rendered = await translator._run_text_rendering(config, ctx)
        with Image.open(input_path) as source:
            base_image = source.copy()
        result = core["dump_image"](base_image, rendered, ctx.img_alpha)
        _save_image(result, output_path)
        regions = serialize_regions(all_regions)
        regions_path.write_text(
            json.dumps(regions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _save_rerender_payload(
            context_path,
            {
                "config": config,
                "text_regions": all_regions,
                "img_rgb": payload["img_rgb"],
                "img_inpainted": clean_inpainted,
                "img_alpha": payload.get("img_alpha"),
            },
        )
        return regions
    finally:
        trim_runtime_memory()
