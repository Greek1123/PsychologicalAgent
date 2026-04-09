# Phase 0: general multi-turn warmup SFT
# Profile: local_8gb
swift sft \
  --model "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset "D:\psychologicalAgent\data\training\general_multiturn\general_phase0_train_ms_swift.jsonl" \
  --train_type lora \
  --torch_dtype float16 \
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_compute_dtype float16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 8e-5 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules all-linear \
  --max_length 1024 \
  --gradient_checkpointing true \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 2 \
  --output_dir "D:\psychologicalAgent\training\ms_swift\outputs\general_phase0_sft"
