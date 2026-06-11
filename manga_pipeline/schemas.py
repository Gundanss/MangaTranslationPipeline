from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

SourceLanguage = Literal["ja", "en"]
TargetLanguage = Literal["zh-CN", "zh-TW", "en", "ja", "ko"]
ProviderName = Literal["ollama", "google", "microsoft"]
TextDirection = Literal["auto", "horizontal", "vertical"]
TextAlignment = Literal["auto", "left", "center", "right"]
TranslateRegionMode = Literal["machine", "ollama"]
BBox = Annotated[list[int], Field(min_length=4, max_length=4)]


# API 请求体刻意保持收敛：这里负责校验，main.py/tasks.py 负责调度，
# 图像核心对象不直接穿过 HTTP 边界。
class TaskConfig(BaseModel):
    """创建翻译任务时由用户选择的一组配置。"""

    name: str = Field(default="漫画翻译", max_length=80)
    source_language: SourceLanguage
    target_language: TargetLanguage = "zh-CN"
    provider: ProviderName = "ollama"
    ollama_model: str | None = None
    polish_with_ollama: bool = False
    polish_model: str | None = None
    render_direction: TextDirection = "auto"
    render_alignment: TextAlignment = "auto"
    font_size: int | None = Field(default=None, ge=6, le=300)
    mask_dilation_offset: int = Field(default=20, ge=0, le=40)

    @field_validator("ollama_model")
    @classmethod
    def require_ollama_model(cls, value: str | None, info):
        provider = info.data.get("provider")
        if provider == "ollama" and not value:
            raise ValueError("使用 Ollama 翻译时必须选择本地模型")
        return value


class SettingsUpdate(BaseModel):
    """本地翻译设置的局部更新请求。"""

    ollama_base_url: str | None = None
    google_api_key: str | None = None
    microsoft_api_key: str | None = None
    microsoft_region: str | None = None
    microsoft_endpoint: str | None = None


class RegionUpdate(BaseModel):
    """校正界面提交的 OCR 框与渲染框编辑状态。"""

    index: int = Field(ge=0)
    bbox: BBox | None = None
    ocr_bbox: BBox | None = None
    render_bbox: BBox | None = None
    enabled: bool = True
    text: str
    translation: str
    angle: float = Field(default=0, ge=-180, le=180, allow_inf_nan=False)
    font_size: int | None = Field(default=None, ge=6, le=300)
    direction: TextDirection = "auto"
    alignment: TextAlignment = "left"
    foreground: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    outline: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    mask_dilation_offset: int = Field(default=20, ge=0, le=40)


class RerenderRequest(BaseModel):
    """仅保存文本与排版修改，并重新嵌字。"""

    regions: list[RegionUpdate]


class ReprocessRegionsRequest(BaseModel):
    """针对选中的人工 OCR 框重新执行 OCR、翻译和去字。"""

    regions: list[RegionUpdate]
    changed_indices: list[int] = Field(default_factory=list)
    mask_changed_indices: list[int] = Field(default_factory=list)


class TranslateRegionRequest(BaseModel):
    """在编辑面板里单独翻译一个文本区域。"""

    mode: TranslateRegionMode
    text: str = Field(min_length=1)


class ResumeFromImageRequest(BaseModel):
    """从用户选中的图片卡片处恢复失败或已停止的任务。"""

    image_id: str = Field(min_length=1)
