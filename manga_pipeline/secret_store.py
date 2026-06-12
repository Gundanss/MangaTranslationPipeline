from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .config import SETTINGS_PATH, ensure_runtime_dirs
from .schemas import SettingsUpdate

DEFAULTS = {
    "ollama_base_url": "http://localhost:11434",
    "google_api_key": "",
    "microsoft_api_key": "",
    "microsoft_region": "",
    "microsoft_endpoint": "https://api.cognitive.microsofttranslator.com",
    "last_ollama_model": "",
}


class SecretStore:
    """本地私有设置文件，保存密钥和最近使用的模型。"""

    def __init__(self, path: Path = SETTINGS_PATH):
        self.path = path
        self._lock = threading.Lock()
        ensure_runtime_dirs()
        if not self.path.exists():
            self._write(DEFAULTS)

    def _read(self) -> dict[str, str]:
        """尽量稳妥地读取设置；JSON 损坏时回退到默认值。"""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        return {**DEFAULTS, **data}

    def _write(self, data: dict[str, str]) -> None:
        """以仅当前用户可读写的权限写回设置文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.chmod(self.path, 0o600)

    def get(self) -> dict[str, str]:
        with self._lock:
            return self._read()

    def update(self, update: SettingsUpdate | dict[str, str]) -> dict[str, str | bool]:
        """合并局部更新，不会清掉这次未传入的旧字段。"""
        values = (
            update.model_dump(exclude_none=True)
            if isinstance(update, SettingsUpdate)
            else update
        )
        with self._lock:
            data = self._read()
            data.update(values)
            self._write(data)
        return self.public()

    def public(self) -> dict[str, str | bool]:
        """只暴露非敏感设置和“是否已配置”的状态标记。"""
        data = self.get()
        return {
            "ollama_base_url": data["ollama_base_url"],
            "google_configured": bool(data["google_api_key"]),
            "microsoft_configured": bool(
                data["microsoft_api_key"] and data["microsoft_region"]
            ),
            "microsoft_region": data["microsoft_region"],
            "microsoft_endpoint": data["microsoft_endpoint"],
            "last_ollama_model": data["last_ollama_model"],
        }
