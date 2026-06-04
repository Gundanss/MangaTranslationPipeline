from manga_pipeline.engine import (
    _core_direction,
    _hex_rgb,
    _is_mps_error,
    _public_direction,
)


def test_direction_mapping_matches_upstream_text_blocks():
    assert _core_direction("horizontal") == "h"
    assert _core_direction("vertical") == "v"
    assert _core_direction("auto") == "auto"
    assert _public_direction("hr") == "horizontal"
    assert _public_direction("vr") == "vertical"


def test_color_and_mps_helpers():
    assert _hex_rgb("#12A0ff") == (18, 160, 255)
    assert _is_mps_error(RuntimeError("MPS backend out of memory"))
    assert not _is_mps_error(RuntimeError("Google translation failed"))
