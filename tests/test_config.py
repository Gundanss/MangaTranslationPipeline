from pathlib import Path

import pytest

from manga_pipeline.config import safe_name, safe_relative_path
from manga_pipeline.schemas import RegionUpdate, TaskConfig


def test_safe_relative_path_preserves_folder_structure():
    assert safe_relative_path("chapter-1/001.webp", "fallback.png") == Path(
        "chapter-1/001.webp"
    )


def test_safe_relative_path_rejects_unsupported_extension():
    with pytest.raises(ValueError):
        safe_relative_path("chapter-1/page.txt", "fallback.png")


def test_safe_relative_path_does_not_allow_parent_escape():
    assert safe_relative_path("../../secret.png", "fallback.png") == Path("fallback.png")


def test_safe_name_removes_path_characters():
    assert safe_name(" 第1话/测试 ") == "第1话-测试"


def test_task_config_contains_initial_typesetting_options():
    config = TaskConfig(
        source_language="ja",
        ollama_model="model",
        render_direction="vertical",
        render_alignment="center",
        font_size=28,
    )
    assert config.render_direction == "vertical"
    assert config.render_alignment == "center"
    assert config.font_size == 28


def test_mask_dilation_defaults_and_bounds():
    config = TaskConfig(source_language="ja", ollama_model="model")
    assert config.mask_dilation_offset == 20

    assert (
        TaskConfig(
            source_language="ja",
            ollama_model="model",
            mask_dilation_offset=0,
        ).mask_dilation_offset
        == 0
    )
    assert (
        TaskConfig(
            source_language="ja",
            ollama_model="model",
            mask_dilation_offset=40,
        ).mask_dilation_offset
        == 40
    )
    with pytest.raises(ValueError):
        TaskConfig(source_language="ja", ollama_model="model", mask_dilation_offset=-1)
    with pytest.raises(ValueError):
        TaskConfig(source_language="ja", ollama_model="model", mask_dilation_offset=41)


def test_region_update_mask_dilation_defaults_and_bounds():
    base = {
        "index": 0,
        "bbox": [0, 0, 10, 10],
        "enabled": True,
        "text": "原文",
        "translation": "译文",
        "foreground": "#000000",
        "outline": "#FFFFFF",
    }
    assert RegionUpdate(**base).mask_dilation_offset == 20
    assert RegionUpdate(**base).angle == 0
    assert RegionUpdate(**{**base, "angle": -12.5}).angle == -12.5
    assert RegionUpdate(**{**base, "mask_dilation_offset": 0}).mask_dilation_offset == 0
    assert RegionUpdate(**{**base, "mask_dilation_offset": 40}).mask_dilation_offset == 40
    with pytest.raises(ValueError):
        RegionUpdate(**{**base, "mask_dilation_offset": -1})
    with pytest.raises(ValueError):
        RegionUpdate(**{**base, "mask_dilation_offset": 41})
    with pytest.raises(ValueError):
        RegionUpdate(**{**base, "angle": -181})
    with pytest.raises(ValueError):
        RegionUpdate(**{**base, "angle": 181})
