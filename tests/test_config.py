from pathlib import Path

import pytest

from manga_pipeline.config import safe_name, safe_relative_path
from manga_pipeline.schemas import TaskConfig


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
