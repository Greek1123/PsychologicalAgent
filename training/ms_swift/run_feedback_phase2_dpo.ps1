# Phase 2: feedback-driven DPO
# This script trains from real bad-case feedback after reviewers fill chosen replies.

$model_path = if (Test-Path "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507") {
  "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507"
} else {
  "Qwen/Qwen3-4B-Instruct-2507"
}

# Replace this with your current best chat/SFT adapter when needed.
$base_adapter = if ($env:FEEDBACK_BASE_ADAPTER) {
  $env:FEEDBACK_BASE_ADAPTER
} else {
  "D:\psychologicalAgent\training\ms_swift\outputs\public_phase0_sft\v0-20260426-134431\checkpoint-465"
}

$dataset_path = if ($env:FEEDBACK_DPO_DATASET) {
  $env:FEEDBACK_DPO_DATASET
} else {
  "D:\psychologicalAgent\data\training\feedback_bad_cases\feedback_dpo_next_train_ready_ms_swift.jsonl"
}
$output_dir = "D:\psychologicalAgent\training\ms_swift\outputs\feedback_phase2_dpo"
$max_length = if ($env:FEEDBACK_DPO_MAX_LENGTH) { $env:FEEDBACK_DPO_MAX_LENGTH } else { "768" }
$grad_accum = if ($env:FEEDBACK_DPO_GRAD_ACCUM) { $env:FEEDBACK_DPO_GRAD_ACCUM } else { "4" }
$save_steps = if ($env:FEEDBACK_DPO_SAVE_STEPS) { $env:FEEDBACK_DPO_SAVE_STEPS } else { "20" }
$extra_args = @()
if ($env:FEEDBACK_DPO_MAX_STEPS) {
  $extra_args += @("--max_steps", $env:FEEDBACK_DPO_MAX_STEPS)
}

if (-not (Test-Path $dataset_path)) {
  Write-Error "Dataset not found: $dataset_path. Run scripts\build_feedback_dpo_dataset.py first."
  exit 1
}

if ((Get-Item $dataset_path).Length -eq 0) {
  Write-Error "Dataset is empty: $dataset_path. Fill chosen replies in bad_cases.jsonl before DPO."
  exit 1
}

if (-not (Test-Path $base_adapter)) {
  Write-Error "Base adapter not found: $base_adapter. Set FEEDBACK_BASE_ADAPTER to your latest checkpoint."
  exit 1
}

# Keep large model downloads off the system drive by default.
$env:MODELSCOPE_CACHE = if ($env:MODELSCOPE_CACHE) { $env:MODELSCOPE_CACHE } else { "D:\llm_cache\modelscope" }
$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { "D:\llm_cache\huggingface" }
$env:HF_HUB_CACHE = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE } else { "D:\llm_cache\huggingface\hub" }
$env:HF_XET_CACHE = if ($env:HF_XET_CACHE) { $env:HF_XET_CACHE } else { "D:\llm_cache\huggingface\xet" }
New-Item -ItemType Directory -Force $env:MODELSCOPE_CACHE | Out-Null
New-Item -ItemType Directory -Force $env:HF_HUB_CACHE | Out-Null
New-Item -ItemType Directory -Force $env:HF_XET_CACHE | Out-Null

swift rlhf `
  --rlhf_type dpo `
  --model $model_path `
  --dataset $dataset_path `
  --train_type lora `
  --adapters $base_adapter `
  --ref_adapters $base_adapter `
  --torch_dtype float16 `
  --quant_method bnb `
  --quant_bits 4 `
  --bnb_4bit_compute_dtype float16 `
  --beta 0.1 `
  --num_train_epochs 1 `
  --learning_rate 2e-5 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps $grad_accum `
  --max_length $max_length `
  --gradient_checkpointing true `
  --logging_steps 5 `
  --save_steps $save_steps `
  --save_total_limit 2 `
  --output_dir $output_dir `
  @extra_args
