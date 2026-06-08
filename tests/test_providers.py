import asyncio

import pytest

from manga_pipeline.providers import (
    GoogleProvider,
    OllamaProvider,
    TranslationError,
    TranslatorProvider,
    _parse_single_response,
    _parse_tagged_response,
    sanitize_translation_text,
)


def test_parse_tagged_response_keeps_region_order():
    response = "<|2|>第二句\n<|1|>第一句"
    assert _parse_tagged_response(response, 2) == ["第一句", "第二句"]


def test_parse_tagged_response_rejects_missing_region():
    with pytest.raises(TranslationError):
        _parse_tagged_response("<|1|>只有一句", 2)


def test_parse_tagged_response_rejects_empty_translation():
    with pytest.raises(TranslationError):
        _parse_tagged_response("<|1|>\n<|2|>第二句", 2)


def test_parse_single_response_accepts_untagged_translation():
    assert _parse_single_response("译文：可以直接使用") == "可以直接使用"


def test_sanitize_translation_text_strips_model_tags():
    assert (
        sanitize_translation_text("翻译：可以直接使用</|1|></|2|>")
        == "可以直接使用"
    )


def test_parse_tagged_response_strips_malformed_closing_tags():
    response = "<|1|>第一句</|1|>\n<|2|>第二句</|2|>"
    assert _parse_tagged_response(response, 2) == ["第一句", "第二句"]


def test_google_web_translation_response_is_joined():
    provider = GoogleProvider()
    payload = [[["你好", "hello", None, None, 10], ["世界", "world", None, None, 10]]]

    assert provider._parse_web_translation(payload) == "你好世界"


def test_google_provider_works_without_api_key(monkeypatch):
    captured = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [[["你好", "hello", None, None, 10]]]

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None, headers=None):
            captured.append((url, params, headers))
            return FakeResponse()

    monkeypatch.setattr("manga_pipeline.providers.httpx.AsyncClient", FakeClient)

    provider = GoogleProvider()
    result = asyncio.run(provider.translate(["hello"], "en", "zh-CN"))

    assert result == ["你好"]
    assert captured[0][0].endswith("/translate_a/single")
    assert captured[0][1]["client"] == "gtx"
    assert captured[0][1]["sl"] == "en"
    assert captured[0][1]["tl"] == "zh-CN"


def test_ollama_falls_back_to_single_regions_when_batch_format_is_invalid():
    events = []
    provider = OllamaProvider("http://localhost:11434", "test-model")

    async def request_tagged(*args):
        raise TranslationError("模型返回的区域编号与 OCR 区域数量不一致")

    async def request_single(system, text, label, temperature):
        return f"翻译：{text}"

    async def log(*args):
        events.append(args)

    provider._request_tagged = request_tagged
    provider._request_single = request_single
    provider.set_log_callback(log)

    result = asyncio.run(provider.translate(["第一句", "第二句"], "ja", "zh-CN"))

    assert result == ["翻译：第一句", "翻译：第二句"]
    assert events[0][1] == "translation-fallback"


def test_provider_retry_is_reported():
    events = []
    attempts = 0

    class Provider(TranslatorProvider):
        async def translate(self, texts, source, target):
            return texts

    async def log(*args):
        events.append(args)

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary")
        return "ok"

    provider = Provider()
    provider.set_log_callback(log)
    assert asyncio.run(provider._run_with_retry("测试翻译", operation)) == "ok"
    assert events[0][1] == "retry"
