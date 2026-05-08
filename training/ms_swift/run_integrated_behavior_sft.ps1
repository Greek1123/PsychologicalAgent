# Integrated behavior SFT: train one adapter from Qwen3 base with public chat,
# safety, weak-input, reviewed feedback, and noisy distress rows.
# Run first:
# python scripts\build_integrated_behavior_sft_dataset.py

$model_path = if (Test-Path "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507") {
  "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507"
} else {
  "Qwen/Qwen3-4B-Instruct-2507"
}

$dataset_path = if ($env:INTEGRATED_BEHAVIOR_DATASET) {
  $env:INTEGRATED_BEHAVIOR_DATASET
} else {
  "D:\psychologicalAgent\data\training\integrated_behavior\integrated_behavior_train_ms_swift.jsonl"
}

$max_steps = if ($env:INTEGRATED_BEHAVIOR_MAX_STEPS) { $env:INTEGRATED_BEHAVIOR_MAX_STEPS } else { "100" }
$resume_args = @()
if ($env:INTEGRATED_BEHAVIOR_RESUME_FROM_CHECKPOINT) {
  $resume_args += @("--resume_from_checkpoint", $env:INTEGRATED_BEHAVIOR_RESUME_FROM_CHECKPOINT)
}

if (-not (Test-Path $dataset_path)) {
  Write-Error "Dataset not found: $dataset_path. Run scripts\build_integrated_behavior_sft_dataset.py first."
  exit 1
}

if ((Get-Item $dataset_path).Length -eq 0) {
  Write-Error "Dataset is empty: $dataset_path."
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
  --torch_dtype float16 `
  --quant_method bnb `
  --quant_bits 4 `
  --bnb_4bit_compute_dtype float16 `
  --num_train_epochs 1 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 1 `
  --learning_rate 2e-5 `
  --lora_rank 8 `
  --lora_alpha 16 `
  --target_modules all-linear `
  --max_length 512 `
  --gradient_checkpointing true `
  --logging_steps 10 `
  --save_steps 50 `
  --save_total_limit 2 `
  --output_dir "D:\psychologicalAgent\training\ms_swift\outputs\integrated_behavior_sft" `
  --max_steps $max_steps `
  @resume_args
