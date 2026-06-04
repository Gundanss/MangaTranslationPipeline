#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
VENDOR = ROOT / "vendor" / "manga-image-translator"

checks = {
    "core": (VENDOR / "manga_translator" / "__init__.py").exists(),
    "ctd": (MODELS / "detection" / "comictextdetector.pt.onnx").exists(),
    "ocr": (MODELS / "ocr" / "ocr_ar_48px.ckpt").exists(),
    "ocr_dictionary": (MODELS / "ocr" / "alphabet-all-v7.txt").exists(),
    "inpainting": (MODELS / "inpainting" / "lama_large_512px.ckpt").exists(),
}
print(json.dumps(checks, ensure_ascii=False))
if not all(checks.values()):
    sys.exit(1)
