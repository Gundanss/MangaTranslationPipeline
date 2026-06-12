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
        self.angle = 0
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

    @property
    def unrotated_size(self):
        x1, y1, x2, y2 = self.xyxy.tolist()
        return max(1, x2 - x1), max(1, y2 - y1)

    @property
    def direction(self):
        return self._direction

    @property
    def vertical(self):
        return str(self.direction).startswith("v")

    @property
    def horizontal(self):
        return not self.vertical

    def get_translation_for_rendering(self):
        return self.translation


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
    region.mask_dilation_offset = 13
    region.angle = 17.5

    data = serialize_regions([region])

    assert data[0]["bbox"] == [5, 6, 7, 8]
    assert data[0]["ocr_bbox"] == [1, 2, 3, 4]
    assert data[0]["render_bbox"] == [5, 6, 7, 8]
    assert data[0]["enabled"] is False
    assert data[0]["mask_dilation_offset"] == 13
    assert data[0]["angle"] == 17.5


def test_mask_refinement_groups_regions_by_dilation_offset():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    config = SimpleNamespace(mask_dilation_offset=20)
    first = FakeRegion("a")
    first.mask_dilation_offset = 13
    second = FakeRegion("b")
    second.mask_dilation_offset = 15
    third = FakeRegion("c")
    third.mask_dilation_offset = 13
    calls = []

    class FakeTranslator:
        async def _run_mask_refinement(self, config, ctx):
            calls.append(
                (
                    config.mask_dilation_offset,
                    [region.translation for region in ctx.text_regions],
                )
            )
            return np.full((4, 4), config.mask_dilation_offset, dtype=np.uint8)

    mask = asyncio.run(
        engine._run_mask_refinement_by_region_offsets(
            FakeTranslator(), config, image, [first, second, third]
        )
    )

    assert calls == [(13, ["a", "c"]), (15, ["b"])]
    assert config.mask_dilation_offset == 20
    assert np.all(mask == 255)


def test_reprocess_mask_only_change_does_not_run_ocr(tmp_path, monkeypatch):
    clean = np.zeros((4, 4, 3), dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    context_path = tmp_path / "context.pkl"
    regions_path = tmp_path / "regions.json"
    Image.fromarray(clean).save(input_path)

    config = SimpleNamespace(
        mask_dilation_offset=20,
        translator=SimpleNamespace(target_lang="CHS"),
        render=SimpleNamespace(no_hyphenation=True, line_spacing=None),
    )
    region = FakeRegion("old translation")
    region.text = "old text"
    region.ocr_bbox = [0, 0, 3, 3]
    region.render_bbox = [0, 0, 3, 3]
    region.mask_dilation_offset = 20
    with context_path.open("wb") as file:
        pickle.dump(
            {
                "config": config,
                "text_regions": [region],
                "img_rgb": clean.copy(),
                "img_inpainted": clean.copy(),
                "img_alpha": None,
                "clean_image_trusted": True,
            },
            file,
        )

    mask_offsets = []

    class FakeTranslator:
        def add_progress_hook(self, _hook):
            return None

        async def _run_ocr(self, config, ctx):
            raise AssertionError("OCR should not run for mask-only changes")

        async def _run_mask_refinement(self, config, ctx):
            mask_offsets.append(config.mask_dilation_offset)
            return np.ones((4, 4), dtype=np.uint8) * 255

        async def _run_inpainting(self, config, ctx):
            return clean + 1

        async def _run_text_rendering(self, config, ctx):
            assert ctx.text_regions[0].text == "old text"
            assert ctx.text_regions[0].translation == "old translation"
            return clean + 2

    monkeypatch.setattr(engine, "_mps_available", lambda: False)
    monkeypatch.setattr(
        engine,
        "_load_text_render",
        lambda: SimpleNamespace(
            calc_horizontal=lambda font_size, text, width, height, lang, hyphenate: (
                [text],
                [min(width, font_size)],
            ),
            calc_vertical=lambda font_size, text, height: ([text], [min(height, font_size)]),
        ),
    )
    monkeypatch.setattr(
        engine.CoreEngine,
        "_build",
        lambda self, use_gpu: (
            {"dump_image": lambda base_image, rendered, alpha: Image.fromarray(rendered)},
            FakeTranslator(),
        ),
    )
    monkeypatch.setattr(engine.CoreEngine, "_config", lambda self, core, use_gpu: config)

    runner = engine.CoreEngine(
        DummyProvider({"old text": "new translation"}),
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
    updates[0]["mask_dilation_offset"] = 13

    regions = asyncio.run(
        runner.reprocess_regions(
            input_path,
            output_path,
            context_path,
            regions_path,
            updates,
            [],
            [0],
        )
    )

    assert mask_offsets == [13]
    assert regions[0]["text"] == "old text"
    assert regions[0]["translation"] == "old translation"
    assert regions[0]["mask_dilation_offset"] == 13
    assert np.array_equal(np.array(Image.open(output_path)), clean + 2)


def test_fit_region_font_size_shrinks_to_render_box(monkeypatch):
    class FakeTextRender:
        @staticmethod
        def calc_horizontal(font_size, text, width, height, language, hyphenate):
            return [text], [font_size * 4]

        @staticmethod
        def calc_vertical(font_size, text, height):
            return [text], [font_size * max(len(text), 1)]

    monkeypatch.setattr(engine, "_load_text_render", lambda: FakeTextRender())

    region = SimpleNamespace(
        translation="测试内容",
        target_lang="CHS",
        vertical=False,
        unrotated_size=(40, 20),
        get_translation_for_rendering=lambda: "测试内容",
    )
    config = SimpleNamespace(render=SimpleNamespace(no_hyphenation=True, line_spacing=None))

    assert engine._fit_region_font_size(region, config) == 10


def test_fit_region_font_size_initializes_text_render_font(monkeypatch):
    class FakeTextRender:
        initialized = False

        @classmethod
        def set_font(cls, font_path):
            cls.initialized = bool(font_path)

        @classmethod
        def calc_horizontal(cls, font_size, text, width, height, language, hyphenate):
            if not cls.initialized:
                raise AttributeError("'NoneType' object has no attribute 'bitmap'")
            return [text], [font_size * 2]

        @classmethod
        def calc_vertical(cls, font_size, text, height):
            if not cls.initialized:
                raise AttributeError("'NoneType' object has no attribute 'bitmap'")
            return [text], [font_size * max(len(text), 1)]

    monkeypatch.setattr(engine, "_load_text_render", lambda: FakeTextRender)
    monkeypatch.setattr(engine, "_font_path", lambda: "/tmp/fake-font.ttf")

    region = SimpleNamespace(
        translation="到宫中",
        target_lang="CHS",
        vertical=True,
        unrotated_size=(120, 160),
        get_translation_for_rendering=lambda: "到宫中",
    )
    config = SimpleNamespace(render=SimpleNamespace(no_hyphenation=True, line_spacing=None))

    assert engine._fit_region_font_size(region, config) > 6
    assert FakeTextRender.initialized is True


def test_rerender_uses_clean_background_and_latest_translation(tmp_path, monkeypatch):
    clean = np.zeros((4, 4, 3), dtype=np.uint8)
    rendered_old = np.full((4, 4, 3), 50, dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    context_path = tmp_path / "context.pkl"
    regions_path = tmp_path / "regions.json"
    Image.fromarray(clean).save(input_path)

    legacy_ctx = SimpleNamespace(
        text_regions=[FakeRegion("old text"), FakeRegion("second old text")],
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
            assert ctx.text_regions[1].translation == "second new text"
            assert ctx.text_regions[0].xyxy.tolist() == [1, 0, 3, 4]
            assert ctx.text_regions[1].xyxy.tolist() == [0, 1, 4, 3]
            assert ctx.text_regions[0].target_lang == "CHS"
            assert ctx.text_regions[1].target_lang == "CHS"
            assert [region.angle for region in ctx.text_regions] == [14, -22]
            return ctx.img_inpainted + 5

    monkeypatch.setattr(
        engine,
        "_import_core",
        lambda: {
            "MangaTranslator": FakeTranslator,
            "dump_image": lambda base_image, rendered, alpha: Image.fromarray(rendered),
        },
    )
    monkeypatch.setattr(
        engine,
        "_load_text_render",
        lambda: SimpleNamespace(
            calc_horizontal=lambda font_size, text, width, height, lang, hyphenate: (
                [text],
                [min(width, max(1, font_size * max(len(text), 1) // 2))],
            ),
            calc_vertical=lambda font_size, text, height: ([text], [min(height, font_size)]),
        ),
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
                    "angle": 14,
                    "font_size": 18,
                    "direction": "auto",
                    "alignment": "center",
                    "foreground": "#000000",
                    "outline": "#FFFFFF",
                },
                {
                    "index": 1,
                    "ocr_bbox": [0, 0, 4, 4],
                    "render_bbox": [0, 1, 4, 3],
                    "text": "second edited source",
                    "translation": "second new text",
                    "angle": -22,
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
    assert regions[1]["translation"] == "second new text"
    assert [region["angle"] for region in regions] == [14, -22]
    assert np.array_equal(np.array(Image.open(output_path)), clean + 5)

    with context_path.open("rb") as file:
        payload = pickle.load(file)
    assert "ctx" not in payload
    assert np.array_equal(payload["img_inpainted"], clean)
    assert context_path.with_suffix(".clean.png").exists()


def test_rerender_prefers_clean_sidecar_over_dirty_payload(tmp_path, monkeypatch):
    clean = np.zeros((4, 4, 3), dtype=np.uint8)
    dirty = np.full((4, 4, 3), 80, dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    context_path = tmp_path / "context.pkl"
    regions_path = tmp_path / "regions.json"
    Image.fromarray(clean).save(input_path)
    Image.fromarray(clean).save(context_path.with_suffix(".clean.png"))

    region = FakeRegion("old text")
    region.ocr_bbox = [0, 0, 4, 4]
    region.render_bbox = [0, 0, 4, 4]
    with context_path.open("wb") as file:
        pickle.dump(
            {
                "config": SimpleNamespace(),
                "text_regions": [region],
                "img_rgb": clean.copy(),
                "img_inpainted": dirty.copy(),
                "img_alpha": None,
            },
            file,
        )

    class FakeTranslator:
        def __init__(self, *_args, **_kwargs):
            pass

        async def _run_text_rendering(self, config, ctx):
            assert np.array_equal(ctx.img_inpainted, clean)
            return ctx.img_inpainted + 6

    monkeypatch.setattr(
        engine,
        "_import_core",
        lambda: {
            "MangaTranslator": FakeTranslator,
            "dump_image": lambda base_image, rendered, alpha: Image.fromarray(rendered),
        },
    )
    monkeypatch.setattr(
        engine,
        "_load_text_render",
        lambda: SimpleNamespace(
            calc_horizontal=lambda font_size, text, width, height, lang, hyphenate: (
                [text],
                [min(width, max(1, font_size * max(len(text), 1) // 2))],
            ),
            calc_vertical=lambda font_size, text, height: ([text], [min(height, font_size)]),
        ),
    )
    monkeypatch.setattr(engine, "_font_path", lambda: "")

    asyncio.run(
        engine.rerender(
            context_path,
            regions_path,
            output_path,
            input_path,
            [
                {
                    "index": 0,
                    "ocr_bbox": [0, 0, 4, 4],
                    "render_bbox": [0, 0, 4, 4],
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

    assert np.array_equal(np.array(Image.open(output_path)), clean + 6)


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
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    trusted_clean = np.ones((4, 4, 3), dtype=np.uint8) * 5
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    context_path = tmp_path / "context.pkl"
    regions_path = tmp_path / "regions.json"
    Image.fromarray(source).save(input_path)

    region = FakeRegion("old text")
    region.text = "old text"
    region.ocr_bbox = [0, 0, 4, 4]
    region.render_bbox = [0, 0, 4, 4]
    region.enabled = True
    region.angle = 33
    with context_path.open("wb") as file:
        pickle.dump(
            {
                "config": SimpleNamespace(),
                "text_regions": [region],
                "img_rgb": source.copy(),
                "img_inpainted": trusted_clean.copy(),
                "img_alpha": None,
                "clean_image_trusted": True,
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
            raise AssertionError("旧 OCR 框纯重识别不应重新生成 mask")

        async def _run_inpainting(self, config, ctx):
            raise AssertionError("旧 OCR 框纯重识别不应重新去字")

        async def _run_text_rendering(self, config, ctx):
            assert ctx.text_regions[0].text == "new ocr"
            assert ctx.text_regions[0].translation == "new translation"
            assert ctx.text_regions[0].angle == 33
            assert np.array_equal(ctx.img_inpainted, trusted_clean)
            ctx.img_inpainted[:] = trusted_clean + 7
            return ctx.img_inpainted

    monkeypatch.setattr(engine, "_mps_available", lambda: False)
    monkeypatch.setattr(
        engine,
        "_load_text_render",
        lambda: SimpleNamespace(
            calc_horizontal=lambda font_size, text, width, height, lang, hyphenate: (
                [text],
                [min(width, max(1, font_size * max(len(text), 1) // 2))],
            ),
            calc_vertical=lambda font_size, text, height: ([text], [min(height, font_size)]),
        ),
    )
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
    assert regions[0]["angle"] == 33
    assert np.array_equal(np.array(Image.open(output_path)), trusted_clean + 7)
    with context_path.open("rb") as file:
        payload = pickle.load(file)
    assert payload["text_regions"][0].text == "new ocr"
    assert payload["text_regions"][0].translation == "new translation"
    assert payload["text_regions"][0].angle == 33
    assert np.array_equal(payload["img_inpainted"], trusted_clean)


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
                "clean_image_trusted": True,
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

    repaint_calls = []

    class FakeTranslator:
        def add_progress_hook(self, _hook):
            return None

        async def _run_detection(self, config, ctx):
            assert ctx.img_rgb.shape == (12, 12, 3)
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
            repaint_calls.append("mask")
            return np.zeros((12, 12), dtype=np.uint8)

        async def _run_inpainting(self, config, ctx):
            repaint_calls.append("inpaint")
            return clean.copy()

        async def _run_text_rendering(self, config, ctx):
            assert ctx.text_regions[0].text == "右左"
            assert ctx.text_regions[0].translation == "合并译文"
            assert ctx.text_regions[0].target_lang == "CHS"
            assert ctx.text_regions[0].angle == -18
            return clean + 9

    monkeypatch.setattr(engine, "_mps_available", lambda: False)
    monkeypatch.setattr(
        engine,
        "_load_text_render",
        lambda: SimpleNamespace(
            calc_horizontal=lambda font_size, text, width, height, lang, hyphenate: (
                text.split("\n"),
                [min(width, max(1, font_size * max(len(line), 1) // 2)) for line in text.split("\n")],
            ),
            calc_vertical=lambda font_size, text, height: ([text], [min(height, font_size * max(len(text), 1))]),
        ),
    )
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
                    "angle": -18,
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
    assert regions[0]["angle"] == -18
    assert repaint_calls == ["mask", "inpaint"]
    assert np.array_equal(np.array(Image.open(output_path)), clean + 9)


def test_reprocess_manual_box_keeps_old_text_when_new_ocr_is_poor(
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

    class FakeTranslator:
        def add_progress_hook(self, _hook):
            return None

        async def _run_detection(self, config, ctx):
            return [], None, None

        async def _run_ocr(self, config, ctx):
            ctx.textlines[0].text = "あ"
            return ctx.textlines

        async def _run_mask_refinement(self, config, ctx):
            return np.zeros((12, 12), dtype=np.uint8)

        async def _run_inpainting(self, config, ctx):
            return clean.copy()

        async def _run_text_rendering(self, config, ctx):
            assert ctx.text_regions[0].text == "ちゃんと勉強しなさい"
            assert ctx.text_regions[0].translation == "保留旧译文"
            return clean + 10

    monkeypatch.setattr(engine, "_mps_available", lambda: False)
    monkeypatch.setattr(
        engine,
        "_load_text_render",
        lambda: SimpleNamespace(
            calc_horizontal=lambda font_size, text, width, height, lang, hyphenate: (
                [text],
                [min(width, font_size)],
            ),
            calc_vertical=lambda font_size, text, height: ([text], [min(height, font_size)]),
        ),
    )
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
        DummyProvider({"ちゃんと勉強しなさい": "保留旧译文"}),
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
                    "text": "ちゃんと勉強しなさい",
                    "translation": "旧译文",
                    "font_size": 24,
                    "direction": "auto",
                    "alignment": "left",
                    "foreground": "#000000",
                    "outline": "#FFFFFF",
                }
            ],
            [0],
        )
    )

    assert regions[0]["text"] == "ちゃんと勉強しなさい"
    assert regions[0]["translation"] == "保留旧译文"
    assert np.array_equal(np.array(Image.open(output_path)), clean + 10)


def test_reprocess_multiple_manual_boxes_fallback_uses_each_old_text(
    tmp_path, monkeypatch
):
    clean = np.zeros((16, 16, 3), dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    context_path = tmp_path / "context.pkl"
    regions_path = tmp_path / "regions.json"
    Image.fromarray(clean).save(input_path)

    first = FakeRegion("旧译文一")
    first.text = "ちゃんと勉強しなさい"
    first.ocr_bbox = [0, 0, 8, 8]
    first.render_bbox = [0, 0, 8, 8]
    first.enabled = True
    second = FakeRegion("旧译文二")
    second.text = "実はママの体しか見てないんだよね"
    second.ocr_bbox = [8, 0, 16, 8]
    second.render_bbox = [8, 0, 16, 8]
    second.enabled = True

    with context_path.open("wb") as file:
        pickle.dump(
            {
                "config": SimpleNamespace(),
                "text_regions": [first, second],
                "img_rgb": clean.copy(),
                "img_inpainted": clean.copy(),
                "img_alpha": None,
                "clean_image_trusted": True,
            },
            file,
        )

    class FakeTranslator:
        def add_progress_hook(self, _hook):
            return None

        async def _run_mask_refinement(self, config, ctx):
            return np.zeros((16, 16), dtype=np.uint8)

        async def _run_inpainting(self, config, ctx):
            return clean.copy()

        async def _run_text_rendering(self, config, ctx):
            assert [region.text for region in ctx.text_regions] == [
                "ちゃんと勉強しなさい",
                "実はママの体しか見てないんだよね",
            ]
            return clean + 11

    async def fake_ocr_user_bbox(
        self, core, translator, config, image, bbox, old_text=None
    ):
        return (old_text or "").strip(), None

    monkeypatch.setattr(engine, "_mps_available", lambda: False)
    monkeypatch.setattr(
        engine,
        "_load_text_render",
        lambda: SimpleNamespace(
            calc_horizontal=lambda font_size, text, width, height, lang, hyphenate: (
                [text],
                [min(width, font_size)],
            ),
            calc_vertical=lambda font_size, text, height: ([text], [min(height, font_size)]),
        ),
    )
    monkeypatch.setattr(
        engine.CoreEngine,
        "_build",
        lambda self, use_gpu: (
            {
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
    monkeypatch.setattr(engine.CoreEngine, "_ocr_user_bbox", fake_ocr_user_bbox)

    runner = engine.CoreEngine(
        DummyProvider(
            {
                "ちゃんと勉強しなさい": "保留旧译文一",
                "実はママの体しか見てないんだよね": "保留旧译文二",
            }
        ),
        "ja",
        "zh-CN",
        None,
        "auto",
        "auto",
        None,
        lambda *_args, **_kwargs: asyncio.sleep(0),
        lambda *_args, **_kwargs: asyncio.sleep(0),
    )

    updates = serialize_regions([first, second])
    regions = asyncio.run(
        runner.reprocess_regions(
            input_path,
            output_path,
            context_path,
            regions_path,
            updates,
            [0, 1],
        )
    )

    assert [region["text"] for region in regions] == [
        "ちゃんと勉強しなさい",
        "実はママの体しか見てないんだよね",
    ]
    assert [region["translation"] for region in regions] == [
        "保留旧译文一",
        "保留旧译文二",
    ]
    assert np.array_equal(np.array(Image.open(output_path)), clean + 11)


def test_manual_ocr_tightens_vertical_box_before_single_box_fallback():
    clean = np.full((40, 30, 3), 255, dtype=np.uint8)
    for y in range(6, 30, 5):
        clean[y : y + 3, 10:14] = [230, 20, 20]
    full_local_bbox = [4, 2, 20, 34]
    seen_bboxes = []

    class FakeQuadrilateral:
        def __init__(self, polygon, text, prob, *colors):
            self.pts = np.array(polygon)
            points = self.pts.reshape(-1, 2)
            self.xyxy = [
                int(points[:, 0].min()),
                int(points[:, 1].min()),
                int(points[:, 0].max()),
                int(points[:, 1].max()),
            ]
            self.text = text
            self.prob = prob
            self.fg_colors = np.array(colors[:3] or (10, 20, 30))
            self.bg_colors = np.array(colors[3:6] or (240, 240, 240))

    class FakeTranslator:
        async def _run_detection(self, config, ctx):
            return [], None, None

        async def _run_ocr(self, config, ctx):
            bbox = list(ctx.textlines[0].xyxy)
            seen_bboxes.append(bbox)
            ctx.textlines[0].text = (
                "ルな点数また"
                if bbox == full_local_bbox
                else "またこんな点数とったわねーっ?"
            )
            return ctx.textlines

    runner = engine.CoreEngine(
        DummyProvider({}),
        "ja",
        "zh-CN",
        None,
        "auto",
        "auto",
        None,
        lambda *_args, **_kwargs: asyncio.sleep(0),
        lambda *_args, **_kwargs: asyncio.sleep(0),
    )

    text, colors = asyncio.run(
        runner._ocr_user_bbox(
            {"Quadrilateral": FakeQuadrilateral},
            FakeTranslator(),
            SimpleNamespace(),
            clean,
            [4, 2, 20, 34],
        )
    )

    assert text == "またこんな点数とったわねーっ?"
    assert len(seen_bboxes) == 1
    assert seen_bboxes[0] != full_local_bbox
    assert (
        seen_bboxes[0][2] - seen_bboxes[0][0]
        < full_local_bbox[2] - full_local_bbox[0]
    )
    assert [value.tolist() for value in colors] == [[10, 20, 30], [240, 240, 240]]


def test_manual_ocr_rejects_partial_score_text_from_adjusted_box():
    clean = np.zeros((16, 16, 3), dtype=np.uint8)

    class FakeQuadrilateral:
        def __init__(self, polygon, text, prob):
            points = np.array(polygon).reshape(-1, 2)
            self.xyxy = [
                int(points[:, 0].min()),
                int(points[:, 1].min()),
                int(points[:, 0].max()),
                int(points[:, 1].max()),
            ]
            self.text = text
            self.prob = prob
            self.fg_colors = (10, 20, 30)
            self.bg_colors = (240, 240, 240)

    class FakeTranslator:
        async def _run_detection(self, config, ctx):
            return [], None, None

        async def _run_ocr(self, config, ctx):
            ctx.textlines[0].text = "ルな点数また"
            return ctx.textlines

    runner = engine.CoreEngine(
        DummyProvider({}),
        "ja",
        "zh-CN",
        None,
        "auto",
        "auto",
        None,
        lambda *_args, **_kwargs: asyncio.sleep(0),
        lambda *_args, **_kwargs: asyncio.sleep(0),
    )

    text, colors = asyncio.run(
        runner._ocr_user_bbox(
            {"Quadrilateral": FakeQuadrilateral},
            FakeTranslator(),
            SimpleNamespace(),
            clean,
            [2, 2, 10, 14],
            "またこんな点数とったわねーっ?",
        )
    )

    assert text == "またこんな点数とったわねーっ?"
    assert colors == ((10, 20, 30), (240, 240, 240))


def test_manual_ocr_quality_rejects_japanese_text_with_many_missing_chars():
    runner = engine.CoreEngine(
        DummyProvider({}),
        "ja",
        "zh-CN",
        None,
        "auto",
        "auto",
        None,
        lambda *_args, **_kwargs: asyncio.sleep(0),
        lambda *_args, **_kwargs: asyncio.sleep(0),
    )

    assert runner._manual_ocr_is_low_quality(
        "夫は…ママのだよ見て",
        "実は…ママの体しか見てないんだよね",
    )
    assert not runner._manual_ocr_is_low_quality(
        "実は…ママの体しか見てないんだよね",
        "実は…ママの体しか見てないんだよね",
    )


def test_rerender_preserves_multiline_translation_and_auto_font_size(
    tmp_path, monkeypatch
):
    clean = np.zeros((8, 8, 3), dtype=np.uint8)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    context_path = tmp_path / "context.pkl"
    regions_path = tmp_path / "regions.json"
    Image.fromarray(clean).save(input_path)

    region = FakeRegion("old text")
    region.text = "原文"
    region.lines = np.array([[[0, 0], [24, 0], [24, 48], [0, 48]]])
    region.ocr_bbox = [0, 0, 24, 48]
    region.render_bbox = [0, 0, 24, 48]
    with context_path.open("wb") as file:
        pickle.dump(
            {
                "config": SimpleNamespace(render=SimpleNamespace(no_hyphenation=True, line_spacing=None)),
                "text_regions": [region],
                "img_rgb": clean.copy(),
                "img_inpainted": clean.copy(),
                "img_alpha": None,
                "clean_image_trusted": True,
            },
            file,
        )

    class FakeTranslator:
        def __init__(self, *_args, **_kwargs):
            pass

        async def _run_text_rendering(self, config, ctx):
            assert ctx.text_regions[0].translation == "第一行\n第二行"
            assert ctx.text_regions[0]._alignment == "left"
            assert ctx.text_regions[0].font_size >= 6
            return ctx.img_inpainted + 3

    monkeypatch.setattr(
        engine,
        "_import_core",
        lambda: {
            "MangaTranslator": FakeTranslator,
            "dump_image": lambda base_image, rendered, alpha: Image.fromarray(rendered),
        },
    )
    monkeypatch.setattr(
        engine,
        "_load_text_render",
        lambda: SimpleNamespace(
            calc_horizontal=lambda font_size, text, width, height, lang, hyphenate: (
                text.split("\n"),
                [min(width, max(1, font_size * max(len(line), 1) // 2)) for line in text.split("\n")],
            ),
            calc_vertical=lambda font_size, text, height: ([text], [min(height, font_size * max(len(text), 1))]),
        ),
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
                    "ocr_bbox": [0, 0, 24, 48],
                    "render_bbox": [0, 0, 24, 48],
                    "text": "原文",
                    "translation": "第一行\n第二行",
                    "font_size": None,
                    "direction": "horizontal",
                    "alignment": "left",
                    "foreground": "#000000",
                    "outline": "#FFFFFF",
                }
            ],
        )
    )

    assert regions[0]["font_size"] is None
    assert regions[0]["alignment"] == "left"
