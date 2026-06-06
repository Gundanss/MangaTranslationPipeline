import asyncio
import json
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


class DummyProvider:
    def __init__(self, translations: dict[str, str] | None = None):
        self.translations = translations or {}
        self.log_callback = None

    def set_log_callback(self, callback):
        self.log_callback = callback

    async def translate(self, texts, source, target):
        return [self.translations.get(text, f"译文:{text}") for text in texts]


class AttrDict(dict):
    __getattr__ = dict.get

    def __setattr__(self, key, value):
        self[key] = value


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
            assert ctx.text_regions[0].target_lang == "CHS"
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


def test_process_keeps_image_and_saves_empty_regions_when_no_text(
    tmp_path, monkeypatch
):
    source = np.full((4, 4, 3), 180, dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    context_path = tmp_path / "context.pkl"
    regions_path = tmp_path / "regions.json"
    Image.fromarray(source).save(input_path)

    class FakeTranslator:
        def add_progress_hook(self, _hook):
            return None

        async def translate(self, image, config):
            return AttrDict(
                result=Image.fromarray(source.copy()),
                text_regions=[],
                img_rgb=source.copy(),
                img_alpha=None,
            )

    monkeypatch.setattr(engine, "_mps_available", lambda: False)
    monkeypatch.setattr(
        engine.CoreEngine,
        "_build",
        lambda self, use_gpu: ({}, FakeTranslator()),
    )
    monkeypatch.setattr(
        engine.CoreEngine,
        "_config",
        lambda self, core, use_gpu: SimpleNamespace(),
    )

    runner = engine.CoreEngine(
        DummyProvider(),
        "ja",
        "zh-CN",
        None,
        "auto",
        "auto",
        None,
        lambda *_args, **_kwargs: asyncio.sleep(0),
        lambda *_args, **_kwargs: asyncio.sleep(0),
    )

    regions = asyncio.run(
        runner.process(input_path, output_path, context_path, regions_path)
    )

    assert regions == []
    assert np.array_equal(np.array(Image.open(output_path)), source)
    assert json.loads(regions_path.read_text(encoding="utf-8")) == []
    with context_path.open("rb") as file:
        payload = pickle.load(file)
    assert payload["text_regions"] == []
    assert np.array_equal(payload["img_rgb"], source)
    assert np.array_equal(payload["img_inpainted"], source)


def test_reprocess_regions_returns_new_ocr_and_translation(tmp_path, monkeypatch):
    clean = np.zeros((4, 4, 3), dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    context_path = tmp_path / "context.pkl"
    regions_path = tmp_path / "regions.json"
    Image.fromarray(clean).save(input_path)

    region = FakeRegion("old text")
    region.text = "old text"
    region.ocr_bbox = [0, 0, 4, 4]
    region.render_bbox = [0, 0, 4, 4]
    region.enabled = True
    with context_path.open("wb") as file:
        pickle.dump(
            {
                "config": SimpleNamespace(),
                "text_regions": [region],
                "img_rgb": clean.copy(),
                "img_inpainted": clean.copy(),
                "img_alpha": None,
            },
            file,
        )

    class FakeQuadrilateral:
        def __init__(self, polygon, text, prob):
            self.polygon = polygon
            self.text = text
            self.prob = prob

    class FakeTranslator:
        def add_progress_hook(self, _hook):
            return None

        async def _run_ocr(self, config, ctx):
            return [
                SimpleNamespace(
                    text="new ocr",
                    prob=0.96,
                    fg_colors=(10, 20, 30),
                    bg_colors=(240, 240, 240),
                )
            ]

        async def _run_mask_refinement(self, config, ctx):
            return np.zeros((4, 4), dtype=np.uint8)

        async def _run_inpainting(self, config, ctx):
            return clean.copy()

        async def _run_text_rendering(self, config, ctx):
            assert ctx.text_regions[0].text == "new ocr"
            assert ctx.text_regions[0].translation == "new translation"
            return clean + 7

    monkeypatch.setattr(engine, "_mps_available", lambda: False)
    monkeypatch.setattr(
        engine.CoreEngine,
        "_build",
        lambda self, use_gpu: (
            {
                "Quadrilateral": FakeQuadrilateral,
                "dump_image": lambda base_image, rendered, alpha: Image.fromarray(
                    rendered
                ),
            },
            FakeTranslator(),
        ),
    )
    monkeypatch.setattr(
        engine.CoreEngine,
        "_config",
        lambda self, core, use_gpu: SimpleNamespace(),
    )

    runner = engine.CoreEngine(
        DummyProvider({"new ocr": "new translation"}),
        "ja",
        "zh-CN",
        None,
        "auto",
        "auto",
        None,
        lambda *_args, **_kwargs: asyncio.sleep(0),
        lambda *_args, **_kwargs: asyncio.sleep(0),
    )

    updates = serialize_regions([region])
    updates[0]["translation"] = "stale translation"

    regions = asyncio.run(
        runner.reprocess_regions(
            input_path,
            output_path,
            context_path,
            regions_path,
            updates,
            [0],
        )
    )

    assert regions[0]["text"] == "new ocr"
    assert regions[0]["translation"] == "new translation"
    assert np.array_equal(np.array(Image.open(output_path)), clean + 7)
    with context_path.open("rb") as file:
        payload = pickle.load(file)
    assert payload["text_regions"][0].text == "new ocr"
    assert payload["text_regions"][0].translation == "new translation"


def test_reprocess_manual_box_detects_and_merges_inner_textlines(
    tmp_path, monkeypatch
):
    clean = np.zeros((12, 12, 3), dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    context_path = tmp_path / "context.pkl"
    regions_path = tmp_path / "regions.json"
    Image.fromarray(clean).save(input_path)

    with context_path.open("wb") as file:
        pickle.dump(
            {
                "config": SimpleNamespace(),
                "text_regions": [],
                "img_rgb": clean.copy(),
                "img_inpainted": clean.copy(),
                "img_alpha": None,
            },
            file,
        )

    class FakeQuadrilateral:
        def __init__(self, pts, text, prob, *colors):
            self.pts = np.array(pts)
            self.text = text
            self.prob = prob
            self._fg = np.array(colors[:3] or (20, 20, 20))
            self._bg = np.array(colors[3:6] or (240, 240, 240))

        @property
        def fg_colors(self):
            return self._fg

        @property
        def bg_colors(self):
            return self._bg

    class FakeTextBlock(FakeRegion):
        def __init__(
            self,
            lines,
            texts=None,
            font_size=12,
            translation="",
            fg_color=(0, 0, 0),
            bg_color=(255, 255, 255),
            **_kwargs,
        ):
            super().__init__(translation)
            self.lines = np.array(lines)
            self.text = (texts or [""])[0]
            self.font_size = font_size
            self._fg = np.array(fg_color)
            self._bg = np.array(bg_color)

    class FakeMergedRegion(FakeRegion):
        def __init__(self, text):
            super().__init__("")
            self.text = text

    class FakeTranslator:
        def add_progress_hook(self, _hook):
            return None

        async def _run_detection(self, config, ctx):
            assert ctx.img_rgb.shape == (8, 8, 3)
            return (
                [
                    FakeQuadrilateral(
                        [[4, 0], [7, 0], [7, 7], [4, 7]], "", 1
                    ),
                    FakeQuadrilateral(
                        [[1, 0], [3, 0], [3, 7], [1, 7]], "", 1
                    ),
                ],
                None,
                None,
            )

        async def _run_ocr(self, config, ctx):
            ctx.textlines[0].text = "右"
            ctx.textlines[1].text = "左"
            return ctx.textlines

        async def _run_textline_merge(self, config, ctx):
            assert ctx.textlines[0].pts[:, 0].min() >= 2
            assert ctx.textlines[0].pts[:, 1].min() >= 2
            return [FakeMergedRegion("右"), FakeMergedRegion("左")]

        async def _run_mask_refinement(self, config, ctx):
            return np.zeros((12, 12), dtype=np.uint8)

        async def _run_inpainting(self, config, ctx):
            return clean.copy()

        async def _run_text_rendering(self, config, ctx):
            assert ctx.text_regions[0].text == "右左"
            assert ctx.text_regions[0].translation == "合并译文"
            assert ctx.text_regions[0].target_lang == "CHS"
            return clean + 9

    monkeypatch.setattr(engine, "_mps_available", lambda: False)
    monkeypatch.setattr(
        engine.CoreEngine,
        "_build",
        lambda self, use_gpu: (
            {
                "Quadrilateral": FakeQuadrilateral,
                "TextBlock": FakeTextBlock,
                "dump_image": lambda base_image, rendered, alpha: Image.fromarray(
                    rendered
                ),
            },
            FakeTranslator(),
        ),
    )
    monkeypatch.setattr(
        engine.CoreEngine,
        "_config",
        lambda self, core, use_gpu: SimpleNamespace(),
    )
    monkeypatch.setattr(engine, "_save_rerender_payload", lambda *_args: None)

    runner = engine.CoreEngine(
        DummyProvider({"右左": "合并译文"}),
        "ja",
        "zh-CN",
        None,
        "auto",
        "auto",
        None,
        lambda *_args, **_kwargs: asyncio.sleep(0),
        lambda *_args, **_kwargs: asyncio.sleep(0),
    )

    regions = asyncio.run(
        runner.reprocess_regions(
            input_path,
            output_path,
            context_path,
            regions_path,
            [
                {
                    "index": 0,
                    "bbox": [2, 2, 10, 10],
                    "ocr_bbox": [2, 2, 10, 10],
                    "render_bbox": [2, 2, 10, 10],
                    "enabled": True,
                    "text": "",
                    "translation": "",
                    "font_size": 24,
                    "direction": "auto",
                    "alignment": "auto",
                    "foreground": "#000000",
                    "outline": "#FFFFFF",
                }
            ],
            [0],
        )
    )

    assert regions[0]["text"] == "右左"
    assert regions[0]["translation"] == "合并译文"
    assert np.array_equal(np.array(Image.open(output_path)), clean + 9)
