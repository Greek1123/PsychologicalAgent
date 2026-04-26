from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("ms_swift_recipe_builder")


DEFAULT_WINDOWS_CACHE_ROOT = "D:\\llm_cache"


def _checkpoint_matches_model(checkpoint_dir: Path, expected_model: str | None) -> bool:
    """Skip stale checkpoints when the training base model family changes."""
    if not expected_model:
        return True

    args_path = checkpoint_dir / "args.json"
    if not args_path.exists():
        return False

    try:
        payload = json.loads(args_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return str(payload.get("model", "")).strip() == expected_model


def _latest_checkpoint_hint(output_dir: str, fallback: str, *, expected_model: str | None = None) -> str:
    """Prefer the newest real checkpoint over the placeholder checkpoint-last path."""
    base = Path(output_dir)
    if not base.exists():
        return fallback

    candidates: list[Path] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("checkpoint-"):
            candidates.append(child)
            continue
        nested = [grand for grand in child.iterdir() if grand.is_dir() and grand.name.startswith("checkpoint-")]
        candidates.extend(nested)

    if expected_model:
        candidates = [path for path in candidates if _checkpoint_matches_model(path, expected_model)]

    if not candidates:
        return fallback

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _build_quant_block(config: dict[str, Any], *, shell: str) -> str:
    quant_method = config.get("quant_method")
    quant_bits = config.get("quant_bits")
    if not quant_method or not quant_bits:
        return ""

    continuation = "`" if shell == "powershell" else "\\"
    return (
        f"  --quant_method {quant_method} {continuation}\n"
        f"  --quant_bits {quant_bits} {continuation}\n"
        f"  --bnb_4bit_compute_dtype {config['torch_dtype']} {continuation}\n"
    )


def _powershell_cache_block(cache_root: str) -> str:
    cache_root = cache_root.rstrip("\\/")
    huggingface_root = f"{cache_root}\\huggingface"
    return (
        "# Keep large model downloads off the system drive by default.\n"
        f'$env:MODELSCOPE_CACHE = if ($env:MODELSCOPE_CACHE) {{ $env:MODELSCOPE_CACHE }} else {{ "{cache_root}\\modelscope" }}\n'
        f'$env:HF_HOME = if ($env:HF_HOME) {{ $env:HF_HOME }} else {{ "{huggingface_root}" }}\n'
        f'$env:HF_HUB_CACHE = if ($env:HF_HUB_CACHE) {{ $env:HF_HUB_CACHE }} else {{ "{huggingface_root}\\hub" }}\n'
        f'$env:HF_XET_CACHE = if ($env:HF_XET_CACHE) {{ $env:HF_XET_CACHE }} else {{ "{huggingface_root}\\xet" }}\n'
        "New-Item -ItemType Directory -Force $env:MODELSCOPE_CACHE | Out-Null\n"
        "New-Item -ItemType Directory -Force $env:HF_HUB_CACHE | Out-Null\n"
        "New-Item -ItemType Directory -Force $env:HF_XET_CACHE | Out-Null\n\n"
    )


def _powershell_sft_script(
    config: dict[str, Any],
    dataset_path: str,
    output_dir: str,
    *,
    phase_label: str,
    previous_adapter_hint: str | None = None,
    epoch_override: int | None = None,
    lr_override: str | None = None,
    grad_accum_override: int | None = None,
) -> str:
    quant_block = _build_quant_block(config, shell="powershell")
    cache_block = _powershell_cache_block(config["windows_cache_root"])
    num_train_epochs = epoch_override if epoch_override is not None else config["num_train_epochs"]
    learning_rate = lr_override if lr_override is not None else config["learning_rate"]
    gradient_accumulation_steps = (
        grad_accum_override if grad_accum_override is not None else config["gradient_accumulation_steps"]
    )

    adapter_block = ""
    adapter_args = ""
    if previous_adapter_hint:
        adapter_block = (
            "# Replace the adapter path below with the actual checkpoint from the previous phase.\n"
            "# Important: the adapter must come from the same base model family as --model.\n"
            f'$previous_adapter = "{previous_adapter_hint}"\n\n'
        )
        adapter_args = "  --adapters $previous_adapter `\n"

    return f"""# {phase_label}
# Profile: {config["profile"]}
# This repo defaults to an explicit dtype because some ms-swift versions do not accept `auto`.
{cache_block}{adapter_block}swift sft `
  --model "{config["model"]}" `
  --dataset "{dataset_path}" `
  --train_type lora `
{adapter_args}  --torch_dtype {config["torch_dtype"]} `
{quant_block}  --num_train_epochs {num_train_epochs} `
  --per_device_train_batch_size {config["per_device_train_batch_size"]} `
  --gradient_accumulation_steps {gradient_accumulation_steps} `
  --learning_rate {learning_rate} `
  --lora_rank {config["lora_rank"]} `
  --lora_alpha {config["lora_alpha"]} `
  --target_modules all-linear `
  --max_length {config["max_length"]} `
  --gradient_checkpointing true `
  --logging_steps 10 `
  --save_steps 100 `
  --save_total_limit 2 `
  --output_dir "{output_dir}"
"""


def _bash_sft_script(
    config: dict[str, Any],
    dataset_path: str,
    output_dir: str,
    *,
    phase_label: str,
    previous_adapter_hint: str | None = None,
    epoch_override: int | None = None,
    lr_override: str | None = None,
    grad_accum_override: int | None = None,
) -> str:
    quant_block = _build_quant_block(config, shell="bash")
    num_train_epochs = epoch_override if epoch_override is not None else config["num_train_epochs"]
    learning_rate = lr_override if lr_override is not None else config["learning_rate"]
    gradient_accumulation_steps = (
        grad_accum_override if grad_accum_override is not None else config["gradient_accumulation_steps"]
    )

    adapter_block = ""
    adapter_args = ""
    if previous_adapter_hint:
        adapter_block = (
            "# Replace PHASE0_ADAPTER_PATH with the actual checkpoint from the previous phase.\n"
            "# Important: the adapter must come from the same base model family as --model.\n"
            f'PHASE0_ADAPTER_PATH="{previous_adapter_hint}"\n\n'
        )
        adapter_args = '  --adapters "$PHASE0_ADAPTER_PATH" \\\n'

    return f"""# {phase_label}
# Profile: {config["profile"]}
{adapter_block}swift sft \\
  --model "{config["model"]}" \\
  --dataset "{dataset_path}" \\
  --train_type lora \\
{adapter_args}  --torch_dtype {config["torch_dtype"]} \\
{quant_block}  --num_train_epochs {num_train_epochs} \\
  --per_device_train_batch_size {config["per_device_train_batch_size"]} \\
  --gradient_accumulation_steps {gradient_accumulation_steps} \\
  --learning_rate {learning_rate} \\
  --lora_rank {config["lora_rank"]} \\
  --lora_alpha {config["lora_alpha"]} \\
  --target_modules all-linear \\
  --max_length {config["max_length"]} \\
  --gradient_checkpointing true \\
  --logging_steps 10 \\
  --save_steps 100 \\
  --save_total_limit 2 \\
  --output_dir "{output_dir}"
"""


def _powershell_dpo_script(
    config: dict[str, Any],
    dataset_path: str,
    sft_adapter_hint: str,
    output_dir: str,
) -> str:
    quant_block = _build_quant_block(config, shell="powershell")
    cache_block = _powershell_cache_block(config["windows_cache_root"])
    return f"""# Phase 2: style alignment with DPO
# Profile: {config["profile"]}
# Before running this script, fill in rejected responses and replace the adapter path below.
$phase1_adapter = "{sft_adapter_hint}"

{cache_block}swift rlhf `
  --rlhf_type dpo `
  --model "{config["model"]}" `
  --dataset "{dataset_path}" `
  --train_type lora `
  --adapters $phase1_adapter `
  --ref_adapters $phase1_adapter `
  --torch_dtype {config["torch_dtype"]} `
{quant_block}  --beta 0.1 `
  --learning_rate 5e-5 `
  --per_device_train_batch_size {config["per_device_train_batch_size"]} `
  --gradient_accumulation_steps {config["gradient_accumulation_steps"]} `
  --max_length {config["max_length"]} `
  --gradient_checkpointing true `
  --logging_steps 10 `
  --save_steps 50 `
  --save_total_limit 2 `
  --output_dir "{output_dir}"
"""


def _bash_dpo_script(config: dict[str, Any], dataset_path: str, sft_adapter_hint: str, output_dir: str) -> str:
    quant_block = _build_quant_block(config, shell="bash")
    return f"""# Phase 2: style alignment with DPO
# Profile: {config["profile"]}
# Replace PHASE1_ADAPTER_PATH after phase 1 training.
PHASE1_ADAPTER_PATH="{sft_adapter_hint}"

swift rlhf \\
  --rlhf_type dpo \\
  --model "{config["model"]}" \\
  --dataset "{dataset_path}" \\
  --train_type lora \\
  --adapters "$PHASE1_ADAPTER_PATH" \\
  --ref_adapters "$PHASE1_ADAPTER_PATH" \\
  --torch_dtype {config["torch_dtype"]} \\
{quant_block}  --beta 0.1 \\
  --learning_rate 5e-5 \\
  --per_device_train_batch_size {config["per_device_train_batch_size"]} \\
  --gradient_accumulation_steps {config["gradient_accumulation_steps"]} \\
  --max_length {config["max_length"]} \\
  --gradient_checkpointing true \\
  --logging_steps 10 \\
  --save_steps 50 \\
  --save_total_limit 2 \\
  --output_dir "{output_dir}"
"""


def _profile_config(profile: str, model_override: str | None, torch_dtype: str | None) -> dict[str, Any]:
    configs: dict[str, dict[str, Any]] = {
        "default": {
            "profile": "default",
            # Use the Qwen3 4B instruct model as the mainline training base.
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "windows_cache_root": DEFAULT_WINDOWS_CACHE_ROOT,
            "torch_dtype": "float16",
            "quant_method": None,
            "quant_bits": None,
            "general_num_train_epochs": 1,
            "weak_input_num_train_epochs": 2,
            "style_num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "weak_input_gradient_accumulation_steps": 4,
            "general_learning_rate": "8e-5",
            "weak_input_learning_rate": "6e-5",
            "style_learning_rate": "5e-5",
            "lora_rank": 8,
            "lora_alpha": 32,
            "max_length": 2048,
        },
        "local_8gb": {
            "profile": "local_8gb",
            # Keep the local profile aligned with the mainline base model while
            # relying on 4-bit QLoRA to fit a consumer 8GB GPU.
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "windows_cache_root": DEFAULT_WINDOWS_CACHE_ROOT,
            "torch_dtype": "float16",
            "quant_method": "bnb",
            "quant_bits": 4,
            "general_num_train_epochs": 1,
            "weak_input_num_train_epochs": 2,
            "style_num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "weak_input_gradient_accumulation_steps": 4,
            "general_learning_rate": "8e-5",
            "weak_input_learning_rate": "6e-5",
            "style_learning_rate": "5e-5",
            "lora_rank": 8,
            "lora_alpha": 16,
            "max_length": 1024,
        },
    }
    if profile not in configs:
        raise ValueError(f"Unsupported profile: {profile}")

    config = dict(configs[profile])
    if model_override:
        config["model"] = model_override
    if torch_dtype:
        config["torch_dtype"] = torch_dtype
    return config


def build_ms_swift_recipes(
    dataset_manifest_path: str,
    output_dir: str,
    *,
    model: str | None = None,
    torch_dtype: str | None = None,
    profile: str = "default",
    general_phase0_dataset: str | None = None,
    weak_input_phase0_5_dataset: str | None = None,
) -> dict[str, Any]:
    dataset_manifest = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    config = _profile_config(profile, model, torch_dtype)
    train_path = dataset_manifest["files"]["train"]
    preference_path = dataset_manifest["files"]["preference"]
    phase0_output_dir = str(output / "outputs" / "general_phase0_sft")
    phase0_5_output_dir = str(output / "outputs" / "weak_input_phase0_5_sft")
    phase1_output_dir = str(output / "outputs" / "style_phase1_sft")
    phase2_output_dir = str(output / "outputs" / "style_phase2_dpo")
    phase0_adapter_hint = _latest_checkpoint_hint(
        phase0_output_dir,
        str(Path(phase0_output_dir) / "checkpoint-last"),
        expected_model=config["model"],
    )
    phase0_5_adapter_hint = _latest_checkpoint_hint(
        phase0_5_output_dir,
        str(Path(phase0_5_output_dir) / "checkpoint-last"),
        expected_model=config["model"],
    )
    phase1_adapter_hint = _latest_checkpoint_hint(
        phase1_output_dir,
        str(Path(phase1_output_dir) / "checkpoint-last"),
        expected_model=config["model"],
    )

    phase0_ps1 = output / "run_general_phase0_sft.ps1"
    phase0_sh = output / "run_general_phase0_sft.sh"
    phase0_5_ps1 = output / "run_weak_input_phase0_5_sft.ps1"
    phase0_5_sh = output / "run_weak_input_phase0_5_sft.sh"
    phase1_ps1 = output / "run_style_phase1_sft.ps1"
    phase1_sh = output / "run_style_phase1_sft.sh"
    phase2_ps1 = output / "run_style_phase2_dpo.ps1"
    phase2_sh = output / "run_style_phase2_dpo.sh"
    manifest_path = output / "ms_swift_recipe_manifest.json"

    if general_phase0_dataset:
        phase0_ps1.write_text(
            _powershell_sft_script(
                config,
                general_phase0_dataset,
                phase0_output_dir,
                phase_label="Phase 0: general multi-turn warmup SFT",
                epoch_override=config["general_num_train_epochs"],
                lr_override=config["general_learning_rate"],
            ),
            encoding="utf-8",
        )
        phase0_sh.write_text(
            _bash_sft_script(
                config,
                general_phase0_dataset,
                phase0_output_dir,
                phase_label="Phase 0: general multi-turn warmup SFT",
                epoch_override=config["general_num_train_epochs"],
                lr_override=config["general_learning_rate"],
            ),
            encoding="utf-8",
        )

    if weak_input_phase0_5_dataset:
        phase0_5_ps1.write_text(
            _powershell_sft_script(
                config,
                weak_input_phase0_5_dataset,
                phase0_5_output_dir,
                phase_label="Phase 0.5: weak-input repair SFT",
                previous_adapter_hint=phase0_adapter_hint if general_phase0_dataset else None,
                epoch_override=config["weak_input_num_train_epochs"],
                lr_override=config["weak_input_learning_rate"],
                grad_accum_override=config.get("weak_input_gradient_accumulation_steps"),
            ),
            encoding="utf-8",
        )
        phase0_5_sh.write_text(
            _bash_sft_script(
                config,
                weak_input_phase0_5_dataset,
                phase0_5_output_dir,
                phase_label="Phase 0.5: weak-input repair SFT",
                previous_adapter_hint=phase0_adapter_hint if general_phase0_dataset else None,
                epoch_override=config["weak_input_num_train_epochs"],
                lr_override=config["weak_input_learning_rate"],
                grad_accum_override=config.get("weak_input_gradient_accumulation_steps"),
            ),
            encoding="utf-8",
        )

    phase1_previous_adapter = None
    if weak_input_phase0_5_dataset:
        phase1_previous_adapter = phase0_5_adapter_hint
    elif general_phase0_dataset:
        phase1_previous_adapter = phase0_adapter_hint

    phase1_ps1.write_text(
        _powershell_sft_script(
            config,
            train_path,
            phase1_output_dir,
            phase_label="Phase 1: light support-style SFT",
            previous_adapter_hint=phase1_previous_adapter,
            epoch_override=config["style_num_train_epochs"],
            lr_override=config["style_learning_rate"],
        ),
        encoding="utf-8",
    )
    phase1_sh.write_text(
        _bash_sft_script(
            config,
            train_path,
            phase1_output_dir,
            phase_label="Phase 1: light support-style SFT",
            previous_adapter_hint=phase1_previous_adapter,
            epoch_override=config["style_num_train_epochs"],
            lr_override=config["style_learning_rate"],
        ),
        encoding="utf-8",
    )
    phase2_ps1.write_text(
        _powershell_dpo_script(config, preference_path, phase1_adapter_hint, phase2_output_dir),
        encoding="utf-8",
    )
    phase2_sh.write_text(
        _bash_dpo_script(config, preference_path, phase1_adapter_hint, phase2_output_dir),
        encoding="utf-8",
    )

    manifest = {
        "profile": config["profile"],
        "model": config["model"],
        "torch_dtype": config["torch_dtype"],
        "quant_method": config["quant_method"],
        "quant_bits": config["quant_bits"],
        "general_phase0_dataset": general_phase0_dataset,
        "weak_input_phase0_5_dataset": weak_input_phase0_5_dataset,
        "files": {
            "phase0_powershell": str(phase0_ps1) if general_phase0_dataset else "",
            "phase0_bash": str(phase0_sh) if general_phase0_dataset else "",
            "phase0_5_powershell": str(phase0_5_ps1) if weak_input_phase0_5_dataset else "",
            "phase0_5_bash": str(phase0_5_sh) if weak_input_phase0_5_dataset else "",
            "phase1_powershell": str(phase1_ps1),
            "phase2_powershell": str(phase2_ps1),
            "phase1_bash": str(phase1_sh),
            "phase2_bash": str(phase2_sh),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Built ms-swift recipe scripts at %s for profile=%s model=%s quant=%s/%s general_phase0=%s",
        output,
        config["profile"],
        config["model"],
        config["quant_method"],
        config["quant_bits"],
        bool(general_phase0_dataset),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build phase-0/1/2 ms-swift scripts for dialogue and style training.")
    parser.add_argument("--dataset-manifest", required=True, help="Path to ms-swift dataset manifest JSON.")
    parser.add_argument("--outdir", required=True, help="Output directory for scripts.")
    parser.add_argument("--model", default=None, help="Optional base model override.")
    parser.add_argument("--torch-dtype", default=None, help="Optional explicit torch dtype override.")
    parser.add_argument("--profile", default="default", choices=["default", "local_8gb"], help="Recipe profile.")
    parser.add_argument("--general-phase0-dataset", default=None, help="Optional general warmup dataset JSONL.")
    parser.add_argument("--weak-input-phase0-5-dataset", default=None, help="Optional weak-input repair dataset JSONL.")
    args = parser.parse_args()

    manifest = build_ms_swift_recipes(
        args.dataset_manifest,
        args.outdir,
        model=args.model,
        torch_dtype=args.torch_dtype,
        profile=args.profile,
        general_phase0_dataset=args.general_phase0_dataset,
        weak_input_phase0_5_dataset=args.weak_input_phase0_5_dataset,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
