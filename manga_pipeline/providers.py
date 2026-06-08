from __future__ import annotations

import asyncio
from dataclasses import dataclass
import html
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

import httpx

SOURCE_NAMES = {"ja": "日语", "en": "英语"}
TARGET_NAMES = {
    "zh-CN": "简体中文",
    "zh-TW": "繁体中文",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
}
MICROSOFT_LANG = {"zh-CN": "zh-Hans", "zh-TW": "zh-Hant"}
GOOGLE_WEB_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
BING_TRANSLATOR_URL = "https://cn.bing.com/translator"
BING_TRANSLATE_ENDPOINT = "https://cn.bing.com/ttranslatev3"
BING_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/137.0.0.0 Safari/537.36"
)
_BING_BOOTSTRAP_HEADERS = {
    "User-Agent": BING_USER_AGENT,
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
_BING_IG_PATTERN = re.compile(r'IG:"([^"]+)"')
_BING_IID_PATTERN = re.compile(r"fbpkgiid\.page\s*=\s*['\"]([^'\"]+)['\"]")
_BING_TOKEN_PATTERN = re.compile(
    r"params_AbusePreventionHelper\s*=\s*\[(\d+),\s*\"([^\"]+)\",\s*(\d+)\]"
)


class TranslationError(RuntimeError):
    pass


class TranslatorProvider(ABC):
    log_callback = None

    def set_log_callback(self, callback) -> None:
        self.log_callback = callback

    async def _run_with_retry(self, label: str, operation, attempts: int = 2):
        error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await operation()
            except Exception as exc:
                error = exc
                if attempt == attempts:
                    break
                if self.log_callback:
                    await self.log_callback(
                        "WARNING",
                        "retry",
                        f"{label}失败，正在进行第 {attempt + 1}/{attempts} 次尝试",
                        {"error": str(exc), "attempt": attempt + 1, "attempts": attempts},
                    )
                await asyncio.sleep(0.8 * attempt)
        raise error or TranslationError(f"{label}失败")

    @abstractmethod
    async def translate(
        self, texts: list[str], source: str, target: str
    ) -> list[str]:
        raise NotImplementedError


@dataclass
class BingSession:
    ig: str
    iid: str
    key: str
    token: str
    expires_at: float
    cookie_header: str = ""
    translator_url: str = BING_TRANSLATOR_URL
    origin: str = "https://cn.bing.com"
    translate_endpoint: str = BING_TRANSLATE_ENDPOINT


def _tagged_prompt(texts: list[str]) -> str:
    return "\n".join(f"<|{index + 1}|>{text}" for index, text in enumerate(texts))


_MODEL_TAG_PATTERN = re.compile(r"(?:<\|\d+\|>|</\|\d+\|>|<\|/\d+\|>)")


def sanitize_translation_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    cleaned = _MODEL_TAG_PATTERN.sub("", cleaned)
    cleaned = re.sub(
        r"^(?:译文|翻译|translation)\s*[:：]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _parse_tagged_response(response: str, expected: int) -> list[str]:
    matches = re.findall(
        r"<\|(\d+)\|>\s*(.*?)(?=<\|\d+\|>|$)", response, flags=re.DOTALL
    )
    parsed = {int(index): sanitize_translation_text(value) for index, value in matches}
    if len(parsed) != expected or any(index not in parsed for index in range(1, expected + 1)):
        raise TranslationError("模型返回的区域编号与 OCR 区域数量不一致")
    results = [parsed[index] for index in range(1, expected + 1)]
    if any(not result for result in results):
        raise TranslationError("模型返回了空翻译")
    return results


def _parse_single_response(response: str) -> str:
    try:
        return _parse_tagged_response(response, 1)[0]
    except TranslationError:
        cleaned = sanitize_translation_text(response)
        if not cleaned:
            raise TranslationError("模型返回了空翻译")
        return cleaned


class OllamaProvider(TranslatorProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def translate(
        self, texts: list[str], source: str, target: str
    ) -> list[str]:
        if not texts:
            return []
        system = (
            "你是专业漫画翻译。"
            f"把{SOURCE_NAMES[source]}漫画台词翻译成{TARGET_NAMES[target]}。"
            "译文自然简洁，适合放回原气泡；保留语气、称谓、标点与每个 <|n|> 编号。"
            "不要解释，不要输出原文，只输出按原顺序逐行排列的编号译文。"
        )

        try:
            return await self._translate_with_format_fallback(
                system, texts, "Ollama 翻译", temperature=0.2
            )
        except Exception as exc:
            raise TranslationError(f"Ollama 翻译失败：{exc}") from exc

    async def polish(
        self, source_texts: list[str], translations: list[str], target: str
    ) -> list[str]:
        inputs = [
            f"原文：{source}\n当前译文：{translation}"
            for source, translation in zip(source_texts, translations)
        ]
        system = (
            f"你是漫画译文校对。将每条当前译文润色为自然简洁的{TARGET_NAMES[target]}，"
            "不改变含义，不添加解释，保留每个 <|n|> 编号，只输出编号译文。"
        )
        try:
            return await self._translate_with_format_fallback(
                system, inputs, "Ollama 润色", temperature=0.15
            )
        except Exception as exc:
            raise TranslationError(f"Ollama 润色失败：{exc}") from exc

    def _chat_payload(
        self, system: str, texts: list[str], temperature: float
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": _tagged_prompt(texts)},
            ],
        }

    async def _request_tagged(
        self, system: str, texts: list[str], label: str, temperature: float
    ) -> list[str]:
        payload = self._chat_payload(system, texts, temperature)

        async def operation():
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
            return _parse_tagged_response(
                response.json()["message"]["content"], len(texts)
            )

        return await self._run_with_retry(label, operation)

    async def _request_single(
        self, system: str, text: str, label: str, temperature: float
    ) -> str:
        payload = self._chat_payload(system, [text], temperature)

        async def operation():
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
            return _parse_single_response(response.json()["message"]["content"])

        return await self._run_with_retry(label, operation)

    async def _translate_with_format_fallback(
        self, system: str, texts: list[str], label: str, temperature: float
    ) -> list[str]:
        try:
            return await self._request_tagged(system, texts, label, temperature)
        except TranslationError:
            if len(texts) == 1:
                return [
                    await self._request_single(
                        system, texts[0], f"{label}（单区域）", temperature
                    )
                ]
            if self.log_callback:
                await self.log_callback(
                    "WARNING",
                    "translation-fallback",
                    f"{label}批量格式不完整，自动切换为逐区域处理",
                    {"regions": len(texts)},
                )
            return [
                await self._request_single(
                    system,
                    text,
                    f"{label}（逐区域 {index}/{len(texts)}）",
                    temperature,
                )
                for index, text in enumerate(texts, start=1)
            ]


class GoogleProvider(TranslatorProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = (api_key or "").strip()

    def _parse_web_translation(self, payload: Any) -> str:
        if not isinstance(payload, list) or not payload:
            raise TranslationError("Google 网页翻译返回格式异常")
        segments = payload[0]
        if not isinstance(segments, list):
            raise TranslationError("Google 网页翻译缺少文本片段")
        parts: list[str] = []
        for segment in segments:
            if isinstance(segment, list) and segment and isinstance(segment[0], str):
                parts.append(segment[0])
        result = html.unescape("".join(parts)).strip()
        if not result:
            raise TranslationError("Google 网页翻译返回了空译文")
        return result

    async def _translate_official(
        self, texts: list[str], source: str, target: str
    ) -> list[str]:
        url = "https://translation.googleapis.com/language/translate/v2"
        payload = {"q": texts, "source": source, "target": target, "format": "text"}

        async def operation():
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(
                    url, params={"key": self.api_key}, json=payload
                )
                response.raise_for_status()
            items = response.json()["data"]["translations"]
            results = [html.unescape(item["translatedText"]).strip() for item in items]
            if len(results) != len(texts):
                raise TranslationError("Google 返回的翻译数量不一致")
            return results

        return await self._run_with_retry("Google 翻译", operation)

    async def _translate_web(
        self, texts: list[str], source: str, target: str
    ) -> list[str]:
        async with httpx.AsyncClient(timeout=90) as client:
            results: list[str] = []
            for text in texts:
                response = await client.get(
                    GOOGLE_WEB_ENDPOINT,
                    params={
                        "client": "gtx",
                        "sl": source,
                        "tl": target,
                        "dt": "t",
                        "q": text,
                    },
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                response.raise_for_status()
                try:
                    payload = response.json()
                except json.JSONDecodeError as exc:
                    raise TranslationError("Google 网页翻译返回了非 JSON 响应") from exc
                results.append(self._parse_web_translation(payload))
            if len(results) != len(texts):
                raise TranslationError("Google 网页翻译返回的翻译数量不一致")
            return results

    async def translate(
        self, texts: list[str], source: str, target: str
    ) -> list[str]:
        try:
            if self.api_key:
                return await self._translate_official(texts, source, target)

            async def operation():
                return await self._translate_web(texts, source, target)

            return await self._run_with_retry("Google 网页翻译", operation)
        except Exception as exc:
            raise TranslationError(f"Google 翻译失败：{exc}") from exc


class BingWebProvider(TranslatorProvider):
    def __init__(self, translator_url: str = BING_TRANSLATOR_URL):
        self.translator_url = translator_url.rstrip("/")
        self.translate_endpoint = f"{self.translator_url.rsplit('/', 1)[0]}/ttranslatev3"
        self._session: BingSession | None = None
        self._request_counter = 0

    def _map_lang(self, lang: str) -> str:
        return MICROSOFT_LANG.get(lang, lang)

    def _parse_bootstrap_html(
        self,
        content: str,
        cookie_header: str = "",
        translator_url: str = BING_TRANSLATOR_URL,
        origin: str = "https://cn.bing.com",
    ) -> BingSession:
        ig_match = _BING_IG_PATTERN.search(content)
        iid_match = _BING_IID_PATTERN.search(content)
        token_match = _BING_TOKEN_PATTERN.search(content)
        if not ig_match or not iid_match or not token_match:
            raise TranslationError("免费 Bing 网页翻译页面结构已变化，无法提取鉴权参数")
        ttl_ms = max(int(token_match.group(3)), 1000)
        return BingSession(
            ig=ig_match.group(1),
            iid=iid_match.group(1),
            key=token_match.group(1),
            token=token_match.group(2),
            expires_at=time.monotonic() + ttl_ms / 1000,
            cookie_header=cookie_header,
            translator_url=translator_url,
            origin=origin,
            translate_endpoint=f"{origin}/ttranslatev3",
        )

    def _cookie_header(self, client: httpx.AsyncClient) -> str:
        return "; ".join(f"{key}={value}" for key, value in client.cookies.items())

    def _is_session_valid(self, session: BingSession | None) -> bool:
        return bool(session and session.expires_at - time.monotonic() > 5)

    async def _bootstrap(self, client: httpx.AsyncClient) -> BingSession:
        response = await client.get(self.translator_url, headers=_BING_BOOTSTRAP_HEADERS)
        response.raise_for_status()
        final_origin = f"{response.url.scheme}://{response.url.host}"
        if response.url.port:
            final_origin = f"{final_origin}:{response.url.port}"
        session = self._parse_bootstrap_html(
            response.text,
            self._cookie_header(client),
            str(response.url),
            final_origin,
        )
        self._session = session
        return session

    async def _ensure_session(
        self, client: httpx.AsyncClient, force_refresh: bool = False
    ) -> BingSession:
        if force_refresh or not self._is_session_valid(self._session):
            return await self._bootstrap(client)
        if self._session and self._session.cookie_header:
            client.headers["Cookie"] = self._session.cookie_header
        return self._session

    def _translate_url(self, session: BingSession) -> str:
        self._request_counter += 1
        return (
            f"{session.translate_endpoint}?isVertical=1&IG={session.ig}"
            f"&IID={session.iid}&SFX={self._request_counter}"
        )

    def _parse_translation_payload(self, payload: Any) -> str:
        items = payload if isinstance(payload, list) else [payload]
        if not items or not isinstance(items[0], dict):
            raise TranslationError("免费 Bing 网页翻译返回格式异常")
        translations = items[0].get("translations")
        if not isinstance(translations, list) or not translations:
            raise TranslationError("免费 Bing 网页翻译缺少译文内容")
        result = html.unescape(str(translations[0].get("text", ""))).strip()
        if not result:
            raise TranslationError("免费 Bing 网页翻译返回了空译文")
        return result

    async def _translate_one(
        self,
        client: httpx.AsyncClient,
        text: str,
        source: str,
        target: str,
    ) -> str:
        for attempt in range(2):
            session = await self._ensure_session(client, force_refresh=attempt > 0)
            body = {
                "fromLang": self._map_lang(source),
                "to": self._map_lang(target),
                "text": text,
                "token": session.token,
                "key": session.key,
            }
            headers = {
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Cache-Control": "no-cache",
                "Origin": session.origin,
                "Pragma": "no-cache",
                "Priority": "u=1, i",
                "Referer": session.translator_url,
                "Sec-CH-UA": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
                "Sec-CH-UA-Mobile": "?0",
                "Sec-CH-UA-Platform": '"macOS"',
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "User-Agent": BING_USER_AGENT,
                "X-Requested-With": "XMLHttpRequest",
            }
            if session.cookie_header:
                headers["Cookie"] = session.cookie_header
            response = await client.post(
                self._translate_url(session),
                data=body,
                headers=headers,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if response.status_code == 400 and attempt == 0:
                    continue
                raise TranslationError(f"免费 Bing 网页翻译请求失败：{exc}") from exc
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise TranslationError("免费 Bing 网页翻译返回了非 JSON 响应") from exc
            return self._parse_translation_payload(payload)
        raise TranslationError("免费 Bing 网页翻译鉴权失效")

    async def translate(
        self, texts: list[str], source: str, target: str
    ) -> list[str]:
        if not texts:
            return []

        async def operation():
            async with httpx.AsyncClient(
                timeout=90,
                follow_redirects=True,
                headers=_BING_BOOTSTRAP_HEADERS,
            ) as client:
                results: list[str] = []
                for text in texts:
                    results.append(await self._translate_one(client, text, source, target))
                if len(results) != len(texts):
                    raise TranslationError("免费 Bing 网页翻译返回的翻译数量不一致")
                return results

        try:
            return await self._run_with_retry("免费 Bing 网页翻译", operation)
        except Exception as exc:
            raise TranslationError(f"免费 Bing 网页翻译失败：{exc}") from exc


class MicrosoftProvider(TranslatorProvider):
    def __init__(self, api_key: str, region: str, endpoint: str):
        if not api_key or not region:
            raise TranslationError("尚未配置 Microsoft Translator API Key 与区域")
        self.api_key = api_key
        self.region = region
        self.endpoint = endpoint.rstrip("/")

    async def translate(
        self, texts: list[str], source: str, target: str
    ) -> list[str]:
        params = {
            "api-version": "3.0",
            "from": source,
            "to": MICROSOFT_LANG.get(target, target),
        }
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Ocp-Apim-Subscription-Region": self.region,
            "X-ClientTraceId": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }
        async def operation():
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(
                    f"{self.endpoint}/translate",
                    params=params,
                    headers=headers,
                    json=[{"text": text} for text in texts],
                )
                response.raise_for_status()
            results = [
                item["translations"][0]["text"].strip() for item in response.json()
            ]
            if len(results) != len(texts):
                raise TranslationError("Microsoft/Bing 返回的翻译数量不一致")
            return results

        try:
            return await self._run_with_retry("Microsoft/Bing 翻译", operation)
        except Exception as exc:
            raise TranslationError(f"Microsoft/Bing 翻译失败：{exc}") from exc


async def list_ollama_models(base_url: str) -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
    except Exception as exc:
        raise TranslationError(f"无法连接 Ollama：{exc}") from exc
    return [
        {
            "name": item["name"],
            "size": item.get("size", 0),
            "modified_at": item.get("modified_at"),
            "parameter_size": item.get("details", {}).get("parameter_size", ""),
            "quantization": item.get("details", {}).get("quantization_level", ""),
            "family": item.get("details", {}).get("family", ""),
        }
        for item in response.json().get("models", [])
    ]


def create_provider(name: str, settings: dict[str, str], ollama_model: str | None):
    if name == "ollama":
        return OllamaProvider(settings["ollama_base_url"], ollama_model or "")
    if name == "google":
        return GoogleProvider(settings["google_api_key"])
    if name == "microsoft":
        if settings["microsoft_api_key"] and settings["microsoft_region"]:
            return MicrosoftProvider(
                settings["microsoft_api_key"],
                settings["microsoft_region"],
                settings["microsoft_endpoint"],
            )
        return BingWebProvider()
    raise TranslationError(f"未知翻译提供方：{name}")
