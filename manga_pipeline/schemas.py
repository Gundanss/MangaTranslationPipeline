from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

SourceLanguage = Literal["ja", "en"]
TargetLanguage = Literal["zh-CN", "zh-TW", "en", "ja", "ko"]
ProviderName = Literal["ollama", "google", "microsoft"]
TextDirection = Literal["auto", "horizontal", "vertical"]
TextAlignment = Literal["auto", "left", "center", "right"]
BBox = Annotated[list[int], Field(min_length=4, max_length=4)]


class TaskConfig(BaseModel):
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

    @field_validator("ollama_model")
    @classmethod
    def require_ollama_model(cls, value: str | None, info):
        provider = info.data.get("provider")
        if provider == "ollama" and not value:
            raise ValueError("使用 Ollama 翻译时必须选择本地模型")
        return value


class SettingsUpdate(BaseModel):
    ollama_base_url: str | None = None
    google_api_key: str | None = None
    microsoft_api_key: str | None = None
    microsoft_region: str | None = None
    microsoft_endpoint: str | None = None


class RegionUpdate(BaseModel):
    index: int = Field(ge=0)
    bbox: BBox | None = None
    ocr_bbox: BBox | None = None
    render_bbox: BBox | None = None
    enabled: bool = True
    text: str
    translation: str
    font_size: int | None = Field(default=None, ge=6, le=300)
    direction: TextDirection = "auto"
    alignment: TextAlignment = "auto"
    foreground: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    outline: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")


class RerenderRequest(BaseModel):
    regions: list[RegionUpdate]


class ReprocessRegionsRequest(BaseModel):
    regions: list[RegionUpdate]
    changed_indices: list[int] = Field(default_factory=list)
