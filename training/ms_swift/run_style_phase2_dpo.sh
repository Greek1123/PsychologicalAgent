# Phase 2: style alignment with DPO
# Profile: local_8gb
# Replace PHASE1_ADAPTER_PATH after phase 1 training.
PHASE1_ADAPTER_PATH="D:\psychologicalAgent\training\ms_swift\outputs\style_phase1_sft\v12-20260421-195735\checkpoint-37"

swift rlhf \
  --rlhf_type dpo \
  --model "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset "D:\psychologicalAgent\data\training\ms_swift\style_phase2_preference_ms_swift.jsonl" \
  --train_type lora \
  --adapters "$PHASE1_ADAPTER_PATH" \
  --ref_adapters "$PHASE1_ADAPTER_PATH" \
  --torch_dtype float16 \
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_compute_dtype float16 \
  --beta 0.1 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 1024 \
  --gradient_checkpointing true \
  --logging_steps 10 \
  --save_steps 50 \
  --save_total_limit 2 \
  --output_dir "D:\psychologicalAgent\training\ms_swift\outputs\style_phase2_dpo"
