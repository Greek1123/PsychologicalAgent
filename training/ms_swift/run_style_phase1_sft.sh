# Phase 1: light support-style SFT
# Profile: local_8gb
# Replace PHASE0_ADAPTER_PATH with the actual checkpoint from the previous phase.
# Important: the adapter must come from the same base model family as --model.
PHASE0_ADAPTER_PATH="D:\psychologicalAgent\training\ms_swift\outputs\weak_input_phase0_5_sft\v5-20260421-210028\checkpoint-18"

swift sft \
  --model "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset "D:\psychologicalAgent\data\training\ms_swift\style_phase1_train_ms_swift.jsonl" \
  --train_type lora \
  --adapters "$PHASE0_ADAPTER_PATH" \
  --torch_dtype float16 \
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_compute_dtype float16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules all-linear \
  --max_length 1024 \
  --gradient_checkpointing true \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 2 \
  --output_dir "D:\psychologicalAgent\training\ms_swift\outputs\style_phase1_sft"
