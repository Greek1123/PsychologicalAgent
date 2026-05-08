# Feedback behavior SFT: learn reviewed good replies without replaying bad assistant history.
# Run after:
# python scripts\build_feedback_behavior_sft_dataset.py --input data\training\feedback_bad_cases\feedback_preference_next_clean.jsonl --out data\training\feedback_bad_cases\feedback_behavior_sft_ms_swift.jsonl

$model_path = if (Test-Path "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507") {
  "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507"
} else {
  "Qwen/Qwen3-4B-Instruct-2507"
}

$previous_adapter = if ($env:FEEDBACK_BEHAVIOR_BASE_ADAPTER) {
  $env:FEEDBACK_BEHAVIOR_BASE_ADAPTER
} else {
  "D:\psychologicalAgent\training\ms_swift\outputs\public_phase0_sft\v0-20260426-134431\checkpoint-465"
}

$dataset_path = if ($env:FEEDBACK_BEHAVIOR_DATASET) {
  $env:FEEDBACK_BEHAVIOR_DATASET
} else {
  "D:\psychologicalAgent\data\training\feedback_bad_cases\feedback_behavior_sft_ms_swift.jsonl"
}
$max_steps_args = @()
if ($env:FEEDBACK_BEHAVIOR_MAX_STEPS) {
  $max_steps_args += @("--max_steps", $env:FEEDBACK_BEHAVIOR_MAX_STEPS)
}

if (-not (Test-Path $dataset_path)) {
  Write-Error "Dataset not found: $dataset_path. Run scripts\build_feedback_behavior_sft_dataset.py first."
  exit 1
}

if ((Get-Item $dataset_path).Length -eq 0) {
  Write-Error "Dataset is empty: $dataset_path."
  exit 1
}

if (-not (Test-Path $previous_adapter)) {
  Write-Error "Base adapter not found: $previous_adapter. Set FEEDBACK_BEHAVIOR_BASE_ADAPTER to your latest checkpoint."
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

swift sft `
  --model $model_path `
  --dataset $dataset_path `
  --train_type lora `
  --adapters $previous_adapter `
  --torch_dtype float16 `
  --quant_method bnb `
  --quant_bits 4 `
  --bnb_4bit_compute_dtype float16 `
  --num_train_epochs 1 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 1 `
  --learning_rate 1e-5 `
  --lora_rank 8 `
  --lora_alpha 16 `
  --target_modules all-linear `
  --max_length 512 `
  --gradient_checkpointing true `
  --logging_steps 1 `
  --save_steps 20 `
  --save_total_limit 2 `
  --output_dir "D:\psychologicalAgent\training\ms_swift\outputs\feedback_behavior_sft" `
  @max_steps_args
