# Phase 0.5b: weak-input repair with public dialogue anchors
# This is safer than training only on a tiny weak-input set.

# Keep large model downloads off the system drive by default.
$env:MODELSCOPE_CACHE = if ($env:MODELSCOPE_CACHE) { $env:MODELSCOPE_CACHE } else { "D:\llm_cache\modelscope" }
$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { "D:\llm_cache\huggingface" }
$env:HF_HUB_CACHE = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE } else { "D:\llm_cache\huggingface\hub" }
$env:HF_XET_CACHE = if ($env:HF_XET_CACHE) { $env:HF_XET_CACHE } else { "D:\llm_cache\huggingface\xet" }
New-Item -ItemType Directory -Force $env:MODELSCOPE_CACHE | Out-Null
New-Item -ItemType Directory -Force $env:HF_HUB_CACHE | Out-Null
New-Item -ItemType Directory -Force $env:HF_XET_CACHE | Out-Null

$previous_adapter = "D:\psychologicalAgent\training\ms_swift\outputs\public_phase0_sft\v0-20260426-134431\checkpoint-465"

swift sft `
  --model "Qwen/Qwen3-4B-Instruct-2507" `
  --dataset "D:\psychologicalAgent\data\training\weak_input\public_weak_input_mixed_train_ms_swift.jsonl" `
  --train_type lora `
  --adapters $previous_adapter `
  --torch_dtype float16 `
  --quant_method bnb `
  --quant_bits 4 `
  --bnb_4bit_compute_dtype float16 `
  --num_train_epochs 1 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --learning_rate 3e-5 `
  --lora_rank 8 `
  --lora_alpha 16 `
  --target_modules all-linear `
  --max_length 1024 `
  --gradient_checkpointing true `
  --logging_steps 10 `
  --save_steps 100 `
  --save_total_limit 2 `
  --output_dir "D:\psychologicalAgent\training\ms_swift\outputs\public_weak_input_mixed_sft"
