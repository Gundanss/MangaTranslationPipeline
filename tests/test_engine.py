import asyncio
import pickle
from types import SimpleNamespace

import numpy as np
from PIL import Image

from manga_pipeline import engine
from manga_pipeline.engine import (
    _core_direction,
    _extract_clean_inpainted_image,
    _hex_rgb,
    _is_mps_error,
    _public_direction,
    serialize_regions,
)


class FakeRegion:
    def __init__(self, translation: str):
        self.lines = np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]])
        self.text = "original"
        self.translation = translation
        self.font_size = 12
        self._direction = "auto"
        self._alignment = "auto"
        self.adjust_bg_color = True
        self._fg = np.array([0, 0, 0])
        self._bg = np.array([255, 255, 255])

    @property
    def xyxy(self):
        points = self.lines.reshape(-1, 2)
        return np.array(
            [
                points[:, 0].min(),
                points[:, 1].min(),
                points[:, 0].max(),
                points[:, 1].max(),
            ]
        )

    def set_font_colors(self, fg_colors, bg_colors):
        self._fg = np.array(fg_colors)
        self._bg = np.array(bg_colors)

    def get_font_colors(self):
        return self._fg, self._bg


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


def test_extract_clean_inpainted_image_prefers_legacy_gimp_mask():
    clean = np.zeros((2, 2, 3), dtype=np.uint8)
    rendered = np.full((2, 2, 3), 50, dtype=np.uint8)
    ctx = SimpleNamespace(
        img_inpainted=rendered,
        gimp_mask=np.dstack((clean[:, :, ::-1], np.zeros((2, 2), dtype=np.uint8))),
    )

    actual = _extract_clean_inpainted_image(ctx)

    assert np.array_equal(actual, clean)


def test_serialize_regions_includes_ocr_and_render_boxes():
    region = FakeRegion("translated")
    region.ocr_bbox = [1, 2, 3, 4]
    region.render_bbox = [5, 6, 7, 8]
    region.enabled = False

    data = serialize_regions([region])

    assert data[0]["bbox"] == [5, 6, 7, 8]
    assert data[0]["ocr_bbox"] == [1, 2, 3, 4]
    assert data[0]["render_bbox"] == [5, 6, 7, 8]
    assert data[0]["enabled"] is False


def test_rerender_uses_clean_background_and_latest_translation(tmp_path, monkeypatch):
    clean = np.zeros((4, 4, 3), dtype=np.uint8)
    rendered_old = np.full((4, 4, 3), 50, dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    context_path = tmp_path / "context.pkl"
    regions_path = tmp_path / "regions.json"
    Image.fromarray(clean).save(input_path)

    legacy_ctx = SimpleNamespace(
        text_regions=[FakeRegion("old text")],
        img_rgb=clean.copy(),
        img_inpainted=rendered_old,
        img_rendered=rendered_old,
        gimp_mask=np.dstack((clean[:, :, ::-1], np.zeros((4, 4), dtype=np.uint8))),
        img_alpha=None,
    )
    with context_path.open("wb") as file:
        pickle.dump({"ctx": legacy_ctx, "config": SimpleNamespace()}, file)

    class FakeTranslator:
        def __init__(self, *_args, **_kwargs):
            pass

        async def _run_text_rendering(self, config, ctx):
            assert ctx.text_regions[0].translation == "new text"
            assert ctx.text_regions[0].xyxy.tolist() == [1, 0, 3, 4]
            return ctx.img_inpainted + 5

    monkeypatch.setattr(
        engine,
        "_import_core",
        lambda: {
            "MangaTranslator": FakeTranslator,
            "dump_image": lambda base_image, rendered, alpha: Image.fromarray(rendered),
        },
    )
    monkeypatch.setattr(engine, "_font_path", lambda: "")

    regions = asyncio.run(
        engine.rerender(
            context_path,
            regions_path,
            output_path,
            input_path,
            [
                {
                    "index": 0,
                    "ocr_bbox": [0, 0, 4, 4],
                    "render_bbox": [1, 0, 3, 4],
                    "text": "edited source",
                    "translation": "new text",
                    "font_size": 18,
                    "direction": "auto",
                    "alignment": "center",
                    "foreground": "#000000",
                    "outline": "#FFFFFF",
                }
            ],
        )
    )

    assert regions[0]["translation"] == "new text"
    assert np.array_equal(np.array(Image.open(output_path)), clean + 5)

    with context_path.open("rb") as file:
        payload = pickle.load(file)
    assert "ctx" not in payload
    assert np.array_equal(payload["img_inpainted"], clean)
