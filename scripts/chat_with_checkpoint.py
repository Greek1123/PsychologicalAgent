from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.response_guardrails import sanitize_user_visible_reply


LOGGER = logging.getLogger("chat_with_checkpoint")
DEFAULT_CACHE_ROOT = Path("D:/llm_cache")
DEFAULT_SYSTEM_PROMPT = (
    "You are a warm campus support companion. Reply in natural Chinese by default. "
    "Most of the time, answer in 2 to 4 sentences instead of only one very short line. "
    "First respond to the user's current feeling or situation, then offer one small follow-up, clarification, "
    "or gentle suggestion. If the user does not want to elaborate, respect that boundary and keep the conversation light "
    "instead of pushing for details. Avoid abrupt topic changes and avoid repetitive template wording."
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_checkpoint_args(checkpoint_dir: Path) -> dict[str, Any]:
    args_path = checkpoint_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Could not find args.json under {checkpoint_dir}")
    return json.loads(args_path.read_text(encoding="utf-8"))


def _configure_cache_root(cache_root: Path) -> None:
    """Reuse a dedicated non-system cache directory unless the user overrides it."""
    cache_root = cache_root.expanduser()
    huggingface_root = cache_root / "huggingface"
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache_root / "modelscope"))
    os.environ.setdefault("HF_HOME", str(huggingface_root))
    os.environ.setdefault("HF_HUB_CACHE", str(huggingface_root / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(huggingface_root / "xet"))
    for env_name in ("MODELSCOPE_CACHE", "HF_HUB_CACHE", "HF_XET_CACHE"):
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)


def _resolve_dtype(torch_module: Any, dtype_name: str | None) -> Any:
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float32":
        return torch_module.float32
    return torch_module.float16


def _resolve_base_model_source(checkpoint_args: dict[str, Any], override_base_model: str | None) -> tuple[str, bool]:
    """Prefer an already-downloaded local base model directory to avoid duplicate downloads."""
    if override_base_model:
        expanded = Path(override_base_model).expanduser()
        return str(expanded), expanded.exists()

    model_dir = checkpoint_args.get("model_dir")
    if isinstance(model_dir, str) and model_dir.strip():
        expanded = Path(model_dir).expanduser()
        if expanded.exists():
            return str(expanded), True

    return str(checkpoint_args["model"]), False


def _load_model_and_tokenizer(checkpoint_dir: Path, *, override_base_model: str | None = None) -> tuple[Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Missing inference dependencies. Please use the same environment where ms-swift training works."
        ) from exc

    checkpoint_args = _load_checkpoint_args(checkpoint_dir)
    base_model_id, local_only = _resolve_base_model_source(checkpoint_args, override_base_model)
    quant_method = checkpoint_args.get("quant_method")
    quant_bits = checkpoint_args.get("quant_bits")
    compute_dtype = _resolve_dtype(torch, checkpoint_args.get("bnb_4bit_compute_dtype"))

    LOGGER.info(
        "Loading base model %s with adapter %s (local_only=%s)",
        base_model_id,
        checkpoint_dir,
        local_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": local_only,
    }
    if quant_method == "bnb" and quant_bits == 4:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=checkpoint_args.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(checkpoint_args.get("bnb_4bit_use_double_quant", True)),
        )
    else:
        model_kwargs["torch_dtype"] = _resolve_dtype(torch, checkpoint_args.get("torch_dtype"))

    # Load the base model first, then attach the LoRA adapter from the checkpoint directory.
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model.eval()
    return model, tokenizer


def _generate_reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return reply


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(description="Chat with a local LoRA checkpoint without relying on swift infer.")
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint directory that contains adapter files.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional local base model path. If omitted, the script will reuse model_dir from checkpoint args.json when available.",
    )
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help="Cache root for ModelScope and Hugging Face downloads. Defaults to D:/llm_cache.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of tokens per reply.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature. Set 0 for greedy.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.08,
        help="Penalty to reduce repetitive continuations such as counting loops.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    _configure_cache_root(Path(args.cache_root))
    LOGGER.info(
        "Using cache directories modelscope=%s hf=%s",
        os.environ["MODELSCOPE_CACHE"],
        os.environ["HF_HUB_CACHE"],
    )
    model, tokenizer = _load_model_and_tokenizer(checkpoint_dir, override_base_model=args.base_model)
    LOGGER.info("Checkpoint loaded successfully. Commands: /reset to clear history, /exit to quit.")

    messages: list[dict[str, str]] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if user_text.lower() == "/reset":
            messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": user_text})
        reply = _generate_reply(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        reply = sanitize_user_visible_reply(user_text, reply, conversation_history=messages)
        print(f"Assistant> {reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Local checkpoint chat failed: %s", exc)
        sys.exit(1)
