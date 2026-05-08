# Noisy distress SFT: typo-heavy crisis/distress input handling.

$model_path = if (Test-Path "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507") {
  "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507"
} else {
  "Qwen/Qwen3-4B-Instruct-2507"
}

$previous_adapter = if ($env:NOISY_DISTRESS_BASE_ADAPTER) {
  $env:NOISY_DISTRESS_BASE_ADAPTER
} else {
  "D:\psychologicalAgent\training\ms_swift\outputs\feedback_behavior_sft\v4-20260507-205302\checkpoint-93"
}

$dataset_path = if ($env:NOISY_DISTRESS_DATASET) {
  $env:NOISY_DISTRESS_DATASET
} else {
  "D:\psychologicalAgent\data\training\noisy_distress\noisy_distress_sft_ms_swift.jsonl"
}

$max_steps_args = @()
if ($env:NOISY_DISTRESS_MAX_STEPS) {
  $max_steps_args += @("--max_steps", $env:NOISY_DISTRESS_MAX_STEPS)
}

if (-not (Test-Path $dataset_path)) {
  Write-Error "Dataset not found: $dataset_path."
  exit 1
}

if (-not (Test-Path $previous_adapter)) {
  Write-Error "Base adapter not found: $previous_adapter. Set NOISY_DISTRESS_BASE_ADAPTER."
  exit 1
}

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
  --num_train_epochs 2 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 1 `
  --learning_rate 8e-6 `
  --lora_rank 8 `
  --lora_alpha 16 `
  --target_modules all-linear `
  --max_length 512 `
  --gradient_checkpointing true `
  --logging_steps 1 `
  --save_steps 15 `
  --save_total_limit 2 `
  --output_dir "D:\psychologicalAgent\training\ms_swift\outputs\noisy_distress_sft" `
  @max_steps_args
