from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath

# 运行期路径全部锚定在项目根目录下，这样双击脚本、测试和 uvicorn
# 启动时都会共用同一套本地数据目录。
ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT_DIR / "static"
VENDOR_CORE_DIR = Path(
    os.getenv("MANGA_CORE_DIR", ROOT_DIR / "vendor" / "manga-image-translator")
)
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = ROOT_DIR / "uploads"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = ROOT_DIR / "models"
LOCAL_DIR = ROOT_DIR / ".local"
DATABASE_PATH = DATA_DIR / "pipeline.db"
SETTINGS_PATH = LOCAL_DIR / "settings.json"

CORE_COMMIT = "d5a3eee4a7b7b7754b71baa2ee82309dfff468bc"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def ensure_runtime_dirs() -> None:
    """创建上传、输出、模型和本地设置所需的目录。"""
    for path in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR, LOCAL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def safe_name(value: str, fallback: str = "漫画翻译") -> str:
    """清洗出适合文件系统使用的名称，同时尽量保留中文可读性。"""
    value = re.sub(r"[\x00-\x1f/\\:*?\"<>|]+", "-", value.strip())
    value = re.sub(r"\s+", " ", value).strip(" .-")
    return value[:80] or fallback


def safe_relative_path(value: str, fallback: str) -> Path:
    """规范化上传相对路径，并拦截不支持的图片格式。"""
    normalized = value.replace("\\", "/").lstrip("/")
    pure = PurePosixPath(normalized)
    if not normalized or pure.is_absolute() or ".." in pure.parts:
        pure = PurePosixPath(fallback)
    clean_parts = [safe_name(part, "file") for part in pure.parts]
    result = Path(*clean_parts)
    if result.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图片格式：{result.suffix or '无扩展名'}")
    return result
