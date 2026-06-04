#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "manga-image-translator"
MODELS = ROOT / "models"


async def main() -> None:
    if not (VENDOR / "manga_translator" / "__init__.py").exists():
        raise SystemExit("漫画处理核心不存在，请先初始化 Git 子模块。")
    sys.path.insert(0, str(VENDOR))

    from manga_translator.detection.ctd import ComicTextDetector
    from manga_translator.inpainting.inpainting_lama_mpe import LamaLargeInpainter
    from manga_translator.ocr.model_48px import Model48pxOCR
    from manga_translator.utils import ModelWrapper

    MODELS.mkdir(parents=True, exist_ok=True)
    ModelWrapper._MODEL_DIR = str(MODELS)

    cpu_model = ComicTextDetector._MODEL_MAPPING["model-cpu"]
    ComicTextDetector._MODEL_MAPPING = {"model-cpu": cpu_model}

    downloads = [
        ("CTD 漫画文字检测模型", ComicTextDetector()),
        ("48px 多语言漫画 OCR 模型", Model48pxOCR()),
        ("LaMa Large 漫画去字模型", LamaLargeInpainter()),
    ]
    for label, model in downloads:
        print(f"\n=== {label} ===")
        await model.download()
    print("\n全部图像处理模型已准备完成。不会下载任何 Ollama 模型。")


if __name__ == "__main__":
    asyncio.run(main())
