import asyncio

import pytest

from manga_pipeline.providers import (
    BingWebProvider,
    GoogleProvider,
    MicrosoftProvider,
    OllamaProvider,
    TranslationError,
    TranslatorProvider,
    _parse_single_response,
    _parse_tagged_response,
    create_provider,
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


def test_sanitize_translation_text_prefers_current_translation_label():
    assert (
        sanitize_translation_text(
            "原文>你作弊了吧？明明说了要好好学习的啊\n"
            "当前译文>你作弊了吧？明明说过要认真学习啊<td>"
        )
        == "你作弊了吧？明明说过要认真学习啊"
    )


def test_parse_tagged_response_cleans_polish_markup():
    response = "<|1|><原文>勉強しなさい</原文><td>译文：好好学习</td>"
    assert _parse_tagged_response(response, 1) == ["好好学习"]


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


def test_bing_bootstrap_parser_extracts_session_fields():
    provider = BingWebProvider()
    html_doc = """
    <script>
    var _G={IG:"ABC123XYZ"};
    fbpkgiid.page = 'translator.5076';
    var params_AbusePreventionHelper = [1780906451889,"token-value",3600000];
    </script>
    """

    session = provider._parse_bootstrap_html(html_doc, "MUID=test-cookie")

    assert session.ig == "ABC123XYZ"
    assert session.iid == "translator.5076"
    assert session.key == "1780906451889"
    assert session.token == "token-value"
    assert session.cookie_header == "MUID=test-cookie"
    assert session.translate_endpoint == "https://cn.bing.com/ttranslatev3"
    assert session.expires_at > 0


def test_create_provider_uses_free_bing_without_official_credentials():
    provider = create_provider(
        "microsoft",
        {
            "ollama_base_url": "http://localhost:11434",
            "google_api_key": "",
            "microsoft_api_key": "",
            "microsoft_region": "",
            "microsoft_endpoint": "https://api.cognitive.microsofttranslator.com",
        },
        None,
    )

    assert isinstance(provider, BingWebProvider)


def test_create_provider_uses_official_microsoft_with_credentials():
    provider = create_provider(
        "microsoft",
        {
            "ollama_base_url": "http://localhost:11434",
            "google_api_key": "",
            "microsoft_api_key": "secret",
            "microsoft_region": "eastasia",
            "microsoft_endpoint": "https://api.cognitive.microsofttranslator.com",
        },
        None,
    )

    assert isinstance(provider, MicrosoftProvider)


def test_bing_provider_works_without_official_key(monkeypatch):
    captured = {"get": [], "post": []}

    class FakeResponse:
        def __init__(self, *, text="", json_data=None, status_code=200, url="https://www.bing.com/translator?mkt=zh-CN"):
            self.text = text
            self._json_data = json_data
            self.status_code = status_code
            self.url = httpx.URL(url)

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "https://cn.bing.com/ttranslatev3")
                raise httpx.HTTPStatusError("bad request", request=request, response=self)
            return None

        def json(self):
            return self._json_data

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.headers = {}
            self.cookies = {"MUID": "cookie-value"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None):
            captured["get"].append((url, headers))
            return FakeResponse(
                text="""
                <script>
                var _G={IG:"ABC123XYZ"};
                fbpkgiid.page = 'translator.5076';
                var params_AbusePreventionHelper = [123456,"token-value",3600000];
                </script>
                """
            )

        async def post(self, url, data=None, headers=None):
            captured["post"].append((url, data, headers))
            return FakeResponse(
                json_data=[{"translations": [{"text": "你好"}]}],
            )

    import httpx

    monkeypatch.setattr("manga_pipeline.providers.httpx.AsyncClient", FakeClient)

    provider = BingWebProvider()
    result = asyncio.run(provider.translate(["hello"], "en", "zh-CN"))

    assert result == ["你好"]
    assert captured["get"][0][0] == "https://cn.bing.com/translator"
    assert captured["post"][0][0].startswith("https://www.bing.com/ttranslatev3?isVertical=1&IG=ABC123XYZ")
    assert captured["post"][0][1]["fromLang"] == "en"
    assert captured["post"][0][1]["to"] == "zh-Hans"
    assert captured["post"][0][1]["token"] == "token-value"
    assert captured["post"][0][1]["key"] == "123456"


def test_bing_provider_rebootstraps_after_400(monkeypatch):
    calls = {"bootstrap": 0, "post": 0}

    class FakeResponse:
        def __init__(self, *, text="", json_data=None, status_code=200, url="https://www.bing.com/translator?mkt=zh-CN"):
            self.text = text
            self._json_data = json_data
            self.status_code = status_code
            self.url = httpx.URL(url)

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "https://cn.bing.com/ttranslatev3")
                raise httpx.HTTPStatusError("bad request", request=request, response=self)
            return None

        def json(self):
            return self._json_data

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.headers = {}
            self.cookies = {"MUID": "cookie-value"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None):
            calls["bootstrap"] += 1
            return FakeResponse(
                text=f"""
                <script>
                var _G={{IG:"ABC123XYZ{calls["bootstrap"]}"}};
                fbpkgiid.page = 'translator.5076';
                var params_AbusePreventionHelper = [123456,"token-{calls["bootstrap"]}",3600000];
                </script>
                """
            )

        async def post(self, url, data=None, headers=None):
            calls["post"] += 1
            if calls["post"] == 1:
                return FakeResponse(status_code=400, json_data={"statusCode": 400})
            return FakeResponse(json_data=[{"translations": [{"text": "你好"}]}])

    import httpx

    monkeypatch.setattr("manga_pipeline.providers.httpx.AsyncClient", FakeClient)

    provider = BingWebProvider()
    result = asyncio.run(provider.translate(["hello"], "en", "zh-CN"))

    assert result == ["你好"]
    assert calls["bootstrap"] == 2
    assert calls["post"] == 2


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


def test_ollama_polish_prompt_uses_structured_inputs():
    captured = {}
    provider = OllamaProvider("http://localhost:11434", "test-model")

    async def request_tagged(system, texts, label, temperature):
        captured["system"] = system
        captured["texts"] = texts
        return ["自然译文"]

    provider._request_tagged = request_tagged

    result = asyncio.run(
        provider.polish(["勉強しなさい"], ["你要学习"], "zh-CN")
    )

    assert result == ["自然译文"]
    assert "禁止输出" in captured["system"]
    assert "源文本=" in captured["texts"][0]
    assert "待润色译文=" in captured["texts"][0]


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
