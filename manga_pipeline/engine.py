from __future__ import annotations

import asyncio
import gc
import inspect
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

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
        from manga_translator.utils import ModelWrapper, dump_image, load_image
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

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _copy_image_array(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    return np.ascontiguousarray(np.array(image, copy=True))


def _extract_clean_inpainted_image(ctx: Any) -> np.ndarray:
    gimp_mask = getattr(ctx, "gimp_mask", None)
    if isinstance(gimp_mask, np.ndarray) and gimp_mask.ndim == 3 and gimp_mask.shape[2] >= 3:
        return np.ascontiguousarray(gimp_mask[..., :3][:, :, ::-1].copy())
    if getattr(ctx, "img_inpainted", None) is None:
        raise RuntimeError("缺少可重新嵌字的去字底图")
    return _copy_image_array(ctx.img_inpainted)


def _build_rerender_payload(ctx: Any, config: Any) -> dict[str, Any]:
    return {
        "config": config,
        "text_regions": ctx.text_regions,
        "img_rgb": _copy_image_array(ctx.img_rgb),
        "img_inpainted": _extract_clean_inpainted_image(ctx),
        "img_alpha": getattr(ctx, "img_alpha", None),
    }


def _normalize_rerender_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "ctx" in payload:
        legacy_ctx = payload["ctx"]
        return {
            "config": payload["config"],
            "text_regions": legacy_ctx.text_regions,
            "img_rgb": _copy_image_array(legacy_ctx.img_rgb),
            "img_inpainted": _extract_clean_inpainted_image(legacy_ctx),
            "img_alpha": getattr(legacy_ctx, "img_alpha", None),
        }
    required = {"config", "text_regions", "img_rgb", "img_inpainted"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"重新嵌字上下文缺少必要字段：{', '.join(sorted(missing))}")
    return payload


def _save_rerender_payload(context_path: Path, payload: dict[str, Any]) -> None:
    context_path.parent.mkdir(parents=True, exist_ok=True)
    with context_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def serialize_regions(regions: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, region in enumerate(regions or []):
        foreground, outline = region.get_font_colors()
        xyxy = [int(value) for value in region.xyxy]
        result.append(
            {
                "index": index,
                "bbox": xyxy,
                "text": region.text,
                "translation": sanitize_translation_text(
                    getattr(region, "translation", "")
                ),
                "font_size": max(6, int(region.font_size)),
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


async def rerender(
    context_path: Path,
    regions_path: Path,
    output_path: Path,
    input_path: Path,
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    core = _import_core()
    with context_path.open("rb") as file:
        payload = _normalize_rerender_payload(pickle.load(file))
    config = payload["config"]
    ctx = SimpleNamespace(
        text_regions=payload["text_regions"],
        img_rgb=payload["img_rgb"],
        img_inpainted=_copy_image_array(payload["img_inpainted"]),
        img_alpha=payload.get("img_alpha"),
        render_mask=None,
    )
    if len(updates) != len(ctx.text_regions):
        raise ValueError("提交的文本区域数量与已保存上下文不一致")
    if {update["index"] for update in updates} != set(range(len(ctx.text_regions))):
        raise ValueError("提交的文本区域编号无效")

    for update in updates:
        region = ctx.text_regions[update["index"]]
        region.text = update["text"]
        region.translation = sanitize_translation_text(update["translation"])
        region.font_size = update["font_size"]
        region._direction = _core_direction(update["direction"])
        region._alignment = update["alignment"]
        region.set_font_colors(
            _hex_rgb(update["foreground"]),
            _hex_rgb(update["outline"]),
        )
        region.adjust_bg_color = False

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
        rendered = await translator._run_text_rendering(config, ctx)
        with Image.open(input_path) as source:
            base_image = source.copy()
        result = core["dump_image"](base_image, rendered, ctx.img_alpha)
        _save_image(result, output_path)
        regions = serialize_regions(ctx.text_regions)
        regions_path.write_text(
            json.dumps(regions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _save_rerender_payload(
            context_path,
            {
                "config": config,
                "text_regions": ctx.text_regions,
                "img_rgb": payload["img_rgb"],
                "img_inpainted": _copy_image_array(payload["img_inpainted"]),
                "img_alpha": payload.get("img_alpha"),
            },
        )
        return regions
    finally:
        trim_runtime_memory()
