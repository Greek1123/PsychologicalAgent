# Phase 2: feedback-driven DPO
# This script trains from real bad-case feedback after reviewers fill chosen replies.

MODEL_PATH="D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507"
if [ ! -d "$MODEL_PATH" ]; then
  MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
fi

BASE_ADAPTER="${FEEDBACK_BASE_ADAPTER:-D:\psychologicalAgent\training\ms_swift\outputs\public_phase0_sft\v0-20260426-134431\checkpoint-465}"
DATASET_PATH="D:\psychologicalAgent\data\training\feedback_bad_cases\feedback_dpo_ms_swift.jsonl"
OUTPUT_DIR="D:\psychologicalAgent\training\ms_swift\outputs\feedback_phase2_dpo"

if [ ! -f "$DATASET_PATH" ]; then
  echo "Dataset not found: $DATASET_PATH. Run scripts/build_feedback_dpo_dataset.py first." >&2
  exit 1
fi

if [ ! -s "$DATASET_PATH" ]; then
  echo "Dataset is empty: $DATASET_PATH. Fill chosen replies in bad_cases.jsonl before DPO." >&2
  exit 1
fi

if [ ! -d "$BASE_ADAPTER" ]; then
  echo "Base adapter not found: $BASE_ADAPTER. Set FEEDBACK_BASE_ADAPTER to your latest checkpoint." >&2
  exit 1
fi

swift rlhf \
  --rlhf_type dpo \
  --model "$MODEL_PATH" \
  --dataset "$DATASET_PATH" \
  --train_type lora \
  --adapters "$BASE_ADAPTER" \
  --ref_adapters "$BASE_ADAPTER" \
  --torch_dtype float16 \
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_compute_dtype float16 \
  --beta 0.1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --gradient_checkpointing true \
  --logging_steps 5 \
  --save_steps 50 \
  --save_total_limit 2 \
  --output_dir "$OUTPUT_DIR"
