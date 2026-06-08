from __future__ import annotations

import asyncio
import html
import json
import re
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
        return MicrosoftProvider(
            settings["microsoft_api_key"],
            settings["microsoft_region"],
            settings["microsoft_endpoint"],
        )
    raise TranslationError(f"未知翻译提供方：{name}")
