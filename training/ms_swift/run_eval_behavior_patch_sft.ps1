# Evaluation behavior patch SFT.
# This is a tiny corrective run built from the latest 50-scenario model evaluation.
# It should patch concrete bad cases without replaying large public datasets.

$model_path = if (Test-Path "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507") {
  "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507"
} else {
  "Qwen/Qwen3-4B-Instruct-2507"
}

$previous_adapter = if ($env:EVAL_BEHAVIOR_BASE_ADAPTER) {
  $env:EVAL_BEHAVIOR_BASE_ADAPTER
} else {
  "D:\psychologicalAgent\training\ms_swift\outputs\curated_behavior_clean_sft\v5-20260508-173940\checkpoint-352"
}

$dataset_path = if ($env:EVAL_BEHAVIOR_DATASET) {
  $env:EVAL_BEHAVIOR_DATASET
} else {
  "D:\psychologicalAgent\data\training\feedback_bad_cases\eval_behavior_sft_20260511_ms_swift.jsonl"
}

$max_steps_args = @()
if ($env:EVAL_BEHAVIOR_MAX_STEPS) {
  $max_steps_args += @("--max_steps", $env:EVAL_BEHAVIOR_MAX_STEPS)
}

if (-not (Test-Path $dataset_path)) {
  Write-Error "Dataset not found: $dataset_path. Run scripts\build_evaluation_review_cases.py first."
  exit 1
}

if ((Get-Item $dataset_path).Length -eq 0) {
  Write-Error "Dataset is empty: $dataset_path."
  exit 1
}

if (-not (Test-Path $previous_adapter)) {
  Write-Error "Base adapter not found: $previous_adapter. Set EVAL_BEHAVIOR_BASE_ADAPTER to your latest checkpoint."
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
  --num_train_epochs 1 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 1 `
  --learning_rate 5e-6 `
  --lora_rank 8 `
  --lora_alpha 16 `
  --target_modules all-linear `
  --max_length 512 `
  --gradient_checkpointing true `
  --logging_steps 1 `
  --save_steps 20 `
  --save_total_limit 2 `
  --output_dir "D:\psychologicalAgent\training\ms_swift\outputs\eval_behavior_patch_sft" `
  @max_steps_args
