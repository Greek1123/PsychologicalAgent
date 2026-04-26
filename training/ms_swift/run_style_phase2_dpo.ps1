# Phase 2: style alignment with DPO
# Profile: local_8gb
# Before running this script, fill in rejected responses and replace the adapter path below.
$phase1_adapter = "D:\psychologicalAgent\training\ms_swift\outputs\style_phase1_sft\v12-20260421-195735\checkpoint-37"

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
  --model "Qwen/Qwen3-4B-Instruct-2507" `
  --dataset "D:\psychologicalAgent\data\training\ms_swift\style_phase2_preference_ms_swift.jsonl" `
  --train_type lora `
  --adapters $phase1_adapter `
  --ref_adapters $phase1_adapter `
  --torch_dtype float16 `
  --quant_method bnb `
  --quant_bits 4 `
  --bnb_4bit_compute_dtype float16 `
  --beta 0.1 `
  --learning_rate 5e-5 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 16 `
  --max_length 1024 `
  --gradient_checkpointing true `
  --logging_steps 10 `
  --save_steps 50 `
  --save_total_limit 2 `
  --output_dir "D:\psychologicalAgent\training\ms_swift\outputs\style_phase2_dpo"
