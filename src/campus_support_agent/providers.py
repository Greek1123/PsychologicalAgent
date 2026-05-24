from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
class LocalCheckpointLLMProvider:
    checkpoint_path: str
    base_model_path: str
    cache_root: str
    max_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    name: str = "local_checkpoint"
    _model: Any | None = None
    _tokenizer: Any | None = None

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        user_text = str(payload.get("student_text", "")).strip()
        history = payload.get("conversation_history") or []
        if not user_text:
            raise RuntimeError("Local checkpoint request failed: missing student_text.")

        model, tokenizer = self._load_model_and_tokenizer()
        messages = self._build_messages(system_prompt=system_prompt, user_text=user_text, history=history)
        reply = self._generate_reply(model, tokenizer, messages)
        logger.info("Local checkpoint response generated successfully.")
        return json.dumps(self._wrap_reply_as_plan(reply, payload), ensure_ascii=False)

    def _load_model_and_tokenizer(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Local checkpoint dependencies are missing. Use the same Anaconda environment that can run ms-swift."
            ) from exc

        checkpoint_dir = Path(self.checkpoint_path).expanduser().resolve()
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Local checkpoint directory not found: {checkpoint_dir}")

        self._configure_cache_root()
        checkpoint_args = self._load_checkpoint_args(checkpoint_dir)
        base_model, local_only = self._resolve_base_model(checkpoint_args)
        compute_dtype = self._resolve_dtype(torch, checkpoint_args.get("bnb_4bit_compute_dtype"))

        logger.info("Loading local checkpoint base=%s adapter=%s", base_model, checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=local_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": local_only,
        }
        if checkpoint_args.get("quant_method") == "bnb" and checkpoint_args.get("quant_bits") == 4:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=checkpoint_args.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=bool(checkpoint_args.get("bnb_4bit_use_double_quant", True)),
            )
        else:
            model_kwargs["torch_dtype"] = self._resolve_dtype(torch, checkpoint_args.get("torch_dtype"))

        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model = PeftModel.from_pretrained(model, str(checkpoint_dir))
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def _configure_cache_root(self) -> None:
        cache_root = Path(self.cache_root).expanduser()
        huggingface_root = cache_root / "huggingface"
        os.environ.setdefault("MODELSCOPE_CACHE", str(cache_root / "modelscope"))
        os.environ.setdefault("HF_HOME", str(huggingface_root))
        os.environ.setdefault("HF_HUB_CACHE", str(huggingface_root / "hub"))
        os.environ.setdefault("HF_XET_CACHE", str(huggingface_root / "xet"))
        for env_name in ("MODELSCOPE_CACHE", "HF_HUB_CACHE", "HF_XET_CACHE"):
            Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_checkpoint_args(checkpoint_dir: Path) -> dict[str, Any]:
        args_path = checkpoint_dir / "args.json"
        if not args_path.exists():
            raise FileNotFoundError(f"Could not find args.json under {checkpoint_dir}")
        return json.loads(args_path.read_text(encoding="utf-8"))

    def _resolve_base_model(self, checkpoint_args: dict[str, Any]) -> tuple[str, bool]:
        if self.base_model_path:
            base_path = Path(self.base_model_path).expanduser()
            return str(base_path), base_path.exists()
        model_dir = checkpoint_args.get("model_dir")
        if isinstance(model_dir, str) and model_dir.strip():
            expanded = Path(model_dir).expanduser()
            if expanded.exists():
                return str(expanded), True
        return str(checkpoint_args["model"]), False

    @staticmethod
    def _resolve_dtype(torch_module: Any, dtype_name: str | None) -> Any:
        if dtype_name == "bfloat16":
            return torch_module.bfloat16
        if dtype_name == "float32":
            return torch_module.float32
        return torch_module.float16

    @staticmethod
    def _build_messages(*, system_prompt: str, user_text: str, history: list[Any]) -> list[dict[str, str]]:
        del system_prompt
        natural_system = (
            "你是一个面向中国大学生的校园心理支持助手。"
            "请用自然中文回复，不要输出 JSON，不要暴露心理熵或内部分析。"
            "先接住用户当下的感受，再给一个很小、现实可做的下一步。"
            "如果用户不想细说，尊重边界，不要追问。"
            "涉及隐私时，明确说明用户不需要透露姓名、班级、宿舍号或具体对象，"
            "并说明不会主动告诉别人；只有出现明确人身安全风险时，才建议联系现实支持。"
            "涉及可能伤害自己或他人时，优先要求用户离开危险现场、不要见对方、移开危险物品，"
            "并联系可信同学、辅导员、家人、校园安保或当地紧急电话。"
            "涉及胸痛、胸闷、用药或诊断时，不下诊断或用药决定，建议联系校医院、医生或急诊。"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": natural_system}]
        for item in history[-8:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        return messages

    def _generate_reply(self, model: Any, tokenizer: Any, messages: list[dict[str, str]]) -> str:
        import torch

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": self.temperature, "top_p": self.top_p})
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _wrap_reply_as_plan(reply: str, payload: dict[str, Any]) -> dict[str, Any]:
        entropy = payload.get("psychological_entropy") or {}
        return {
            "primary_emotions": ["压力", "低落"],
            "stressors": entropy.get("dominant_drivers") or ["当前表达的校园压力"],
            "protective_factors": ["愿意表达当前状态", "仍在尝试寻求支持"],
            "entropy_level": entropy.get("level", 2),
            "balance_state": entropy.get("balance_state", "stable"),
            "summary": reply,
            "immediate_support": [reply],
            "campus_actions": [],
            "self_regulation": [],
            "follow_up": [reply],
        }


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
    if provider == "local_checkpoint":
        return LocalCheckpointLLMProvider(
            checkpoint_path=settings.local_checkpoint_path,
            base_model_path=settings.local_base_model_path,
            cache_root=settings.local_model_cache_root,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.local_generation_temperature,
            top_p=settings.local_generation_top_p,
            repetition_penalty=settings.local_generation_repetition_penalty,
        )
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
