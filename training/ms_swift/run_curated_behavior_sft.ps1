# Clean curated behavior SFT: role boundary, privacy wording, weak-input handling.
# Run after:
# python scripts\build_curated_behavior_dataset.py

# Keep large model downloads off the system drive by default.
$env:MODELSCOPE_CACHE = if ($env:MODELSCOPE_CACHE) { $env:MODELSCOPE_CACHE } else { "D:\llm_cache\modelscope" }
$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { "D:\llm_cache\huggingface" }
$env:HF_HUB_CACHE = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE } else { "D:\llm_cache\huggingface\hub" }
$env:HF_XET_CACHE = if ($env:HF_XET_CACHE) { $env:HF_XET_CACHE } else { "D:\llm_cache\huggingface\xet" }
New-Item -ItemType Directory -Force $env:MODELSCOPE_CACHE | Out-Null
New-Item -ItemType Directory -Force $env:HF_HUB_CACHE | Out-Null
New-Item -ItemType Directory -Force $env:HF_XET_CACHE | Out-Null

$model_path = if (Test-Path "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507") {
  "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507"
} else {
  "Qwen/Qwen3-4B-Instruct-2507"
}

$dataset_path = if ($env:CURATED_BEHAVIOR_DATASET) {
  $env:CURATED_BEHAVIOR_DATASET
} else {
  "D:\psychologicalAgent\data\training\curated_behavior\curated_behavior_messages_only_ms_swift.jsonl"
}

$adapter_args = @()
if ($env:CURATED_BEHAVIOR_BASE_ADAPTER -and $env:CURATED_BEHAVIOR_BASE_ADAPTER -ne "none") {
  if (-not (Test-Path $env:CURATED_BEHAVIOR_BASE_ADAPTER)) {
    Write-Error "Base adapter not found: $env:CURATED_BEHAVIOR_BASE_ADAPTER"
    exit 1
  }
  $adapter_args += @("--adapters", $env:CURATED_BEHAVIOR_BASE_ADAPTER)
}

$max_steps = if ($env:CURATED_BEHAVIOR_MAX_STEPS) { $env:CURATED_BEHAVIOR_MAX_STEPS } else { "352" }

if (-not (Test-Path $dataset_path)) {
  Write-Error "Dataset not found: $dataset_path. Run scripts\build_curated_behavior_dataset.py first."
  exit 1
}

swift sft `
  --model $model_path `
  --dataset $dataset_path `
  --train_type lora `
  @adapter_args `
  --torch_dtype float16 `
  --quant_method bnb `
  --quant_bits 4 `
  --bnb_4bit_compute_dtype float16 `
  --num_train_epochs 1 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 1 `
  --learning_rate 1.5e-5 `
  --lora_rank 8 `
  --lora_alpha 16 `
  --target_modules all-linear `
  --max_length 512 `
  --gradient_checkpointing true `
  --logging_steps 20 `
  --save_steps 100 `
  --save_total_limit 2 `
  --output_dir "D:\psychologicalAgent\training\ms_swift\outputs\curated_behavior_clean_sft" `
  --max_steps $max_steps
