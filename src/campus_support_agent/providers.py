from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from .config import Settings
from .logging_utils import get_logger


logger = get_logger("providers")


class LLMProvider(Protocol):
    name: str

    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


class STTProvider(Protocol):
    name: str

    def transcribe(self, *, file_bytes: bytes, filename: str, content_type: str | None) -> str: ...


def _join_url(base_url: str, path: str) -> str:
    normalized_base = base_url.rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{normalized_base}{normalized_path}"


def _post_json(url: str, payload: dict, api_key: str, timeout_seconds: int) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM request failed: HTTP {exc.code} - {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc


def _build_multipart_body(
    *,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    boundary = f"----CampusSupportAgent{uuid4().hex}"
    chunks: list[bytes] = []

    # 手动拼 multipart，避免再引入额外 HTTP 客户端依赖。
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )

    return b"".join(chunks), boundary


def _post_multipart(
    *,
    url: str,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
    api_key: str,
    timeout_seconds: int,
) -> dict | str:
    body, boundary = _build_multipart_body(
        fields=fields,
        file_field=file_field,
        filename=filename,
        file_bytes=file_bytes,
        content_type=content_type,
    )
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return payload
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"STT request failed: HTTP {exc.code} - {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"STT request failed: {exc.reason}") from exc


@dataclass(slots=True)
class MockLLMProvider:
    name: str = "mock"

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        payload = json.loads(user_prompt)
        text = payload.get("student_text", "")
        logger.debug("Mock LLM provider handling text with length=%s", len(text))

        emotions = ["焦虑", "疲惫"] if any(term in text for term in ["考试", "挂科", "论文"]) else ["低落", "紧张"]
        stressors = []
        if "考试" in text or "挂科" in text:
            stressors.append("学业评估压力")
        if "室友" in text or "人际" in text:
            stressors.append("宿舍或同伴关系压力")
        if "睡" in text:
            stressors.append("睡眠紊乱")
        if not stressors:
            stressors.append("近期综合压力负荷")

        result = {
            "primary_emotions": emotions,
            "stressors": stressors,
            "protective_factors": ["愿意主动求助", "能够描述自己的状态"],
            "entropy_level": 3 if "睡" in text or "考试" in text else 2,
            "balance_state": "strained",
            "summary": "你正在承受一段持续性的校园压力，当前最重要的是先稳住节律，再拆解问题。",
            "immediate_support": [
                "先把今天最担心的事情写成 1 句话，避免在脑中反复放大。",
                "今晚只给自己安排 1 个最小任务，完成后就停下。",
                "如果已经连续多天睡不好，今晚优先处理睡眠而不是继续硬撑。",
            ],
            "campus_actions": [
                "如果压力持续超过 1-2 周，预约学校心理中心的支持服务。",
                "把本周课程或作业压力和可信任同学或辅导员同步一次。",
            ],
            "self_regulation": [
                "做 3 轮缓慢呼吸：吸气 4 秒，呼气 6 秒。",
                "把待办拆成 15 分钟一段，先开始第一段。",
            ],
            "follow_up": [
                "明天记录睡眠时长和醒来后的精力评分。",
                "48 小时后再评估情绪和压力是否下降。",
            ],
        }
        return json.dumps(result, ensure_ascii=False)


@dataclass(slots=True)
class OpenAICompatibleLLMProvider:
    base_url: str
    model: str
    api_key: str
    timeout_seconds: int
    max_tokens: int
    name: str = "openai_compatible"

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        logger.info("Calling OpenAI-compatible LLM model=%s base_url=%s", self.model, self.base_url)
        url = _join_url(self.base_url, "/chat/completions")
        payload = {
            "model": self.model,
            "temperature": 0.3,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = _post_json(url, payload, self.api_key, self.timeout_seconds)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("LLM request failed: missing choices in response.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            logger.info("LLM response received successfully.")
            return content
        if isinstance(content, list):
            texts = [item.get("text", "") for item in content if isinstance(item, dict)]
            if texts:
                logger.info("LLM response received successfully.")
                return "".join(texts)
        raise RuntimeError("LLM request failed: missing message content.")


@dataclass(slots=True)
class DisabledSTTProvider:
    name: str = "disabled"

    def transcribe(self, *, file_bytes: bytes, filename: str, content_type: str | None) -> str:
        del file_bytes, filename, content_type
        raise RuntimeError("当前没有启用语音转写服务，请把 STT_PROVIDER 改成 mock 或 openai_compatible。")


@dataclass(slots=True)
class MockSTTProvider:
    name: str = "mock"

    def transcribe(self, *, file_bytes: bytes, filename: str, content_type: str | None) -> str:
        del content_type
        byte_size = len(file_bytes)
        logger.debug("Mock STT provider handling file=%s size=%s", filename, byte_size)
        return f"模拟语音转写结果：用户上传了音频 {filename}，大小约 {byte_size} 字节，当前表达出学习与情绪压力。"


@dataclass(slots=True)
class OpenAICompatibleSTTProvider:
    base_url: str
    model: str
    api_key: str
    timeout_seconds: int
    language: str
    name: str = "openai_compatible"

    def transcribe(self, *, file_bytes: bytes, filename: str, content_type: str | None) -> str:
        logger.info("Calling OpenAI-compatible STT model=%s base_url=%s", self.model, self.base_url)
        guessed_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        url = _join_url(self.base_url, "/audio/transcriptions")
        fields = {
            "model": self.model,
            "language": self.language,
            "response_format": "json",
        }
        response = _post_multipart(
            url=url,
            fields=fields,
            file_field="file",
            filename=filename,
            file_bytes=file_bytes,
            content_type=guessed_type,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        if isinstance(response, str):
            logger.info("STT response received successfully.")
            return response.strip()
        text = response.get("text")
        if not text:
            raise RuntimeError("STT request failed: missing text in response.")
        logger.info("STT response received successfully.")
        return text.strip()


def build_llm_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.strip().lower()
    if provider == "openai_compatible":
        return OpenAICompatibleLLMProvider(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            max_tokens=settings.llm_max_tokens,
        )
    return MockLLMProvider()


def build_stt_provider(settings: Settings) -> STTProvider:
    provider = settings.stt_provider.strip().lower()
    if provider == "openai_compatible":
        return OpenAICompatibleSTTProvider(
            base_url=settings.stt_base_url,
            model=settings.stt_model,
            api_key=settings.stt_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            language=settings.stt_language,
        )
    if provider == "disabled":
        return DisabledSTTProvider()
    return MockSTTProvider()
