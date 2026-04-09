# ms-swift Style Training

This document connects the existing style-first pack to runnable `ms-swift` training stages.

## Stage order

1. Build the style-first pack.
2. Export the pack into ms-swift standard datasets.
3. Run phase 1 SFT.
4. Fill in `rejected` responses for the preference file.
5. Re-export the ms-swift DPO dataset.
6. Run phase 2 DPO.
7. Keep human evaluation outside the training loop and score naturalness, empathy, follow-up quality, summary accuracy, and safety.

## Build ms-swift datasets

```bash
python scripts/build_ms_swift_style_datasets.py
```

This creates:

- `data/training/ms_swift/style_phase1_train_ms_swift.jsonl`
- `data/training/ms_swift/style_phase1_dev_ms_swift.jsonl`
- `data/training/ms_swift/style_phase1_test_ms_swift.jsonl`
- `data/training/ms_swift/style_phase2_preference_ms_swift.jsonl`
- `data/training/ms_swift/ms_swift_style_manifest.json`

Notes:

- The SFT exports use the standard `messages` format.
- The DPO export only includes samples that already have a non-empty `rejected` field.
- If `style_phase2_preference_ms_swift.jsonl` is empty, you still need to finish preference annotation first.

## Generate stage scripts

```bash
python scripts/generate_ms_swift_recipes.py
```

This creates:

- `training/ms_swift/run_style_phase1_sft.ps1`
- `training/ms_swift/run_style_phase2_dpo.ps1`
- `training/ms_swift/run_style_phase1_sft.sh`
- `training/ms_swift/run_style_phase2_dpo.sh`

If your installed `ms-swift` version rejects `--torch_dtype auto`, regenerate the scripts with an explicit dtype:

```bash
python scripts/generate_ms_swift_recipes.py --torch-dtype float16
```

If you are training locally on an RTX 4060 8GB class machine, generate the lighter recipe instead:

```bash
python scripts/generate_ms_swift_recipes.py --profile local_8gb
```

The generated PowerShell scripts now default model caches to `D:\llm_cache` so large downloads do not go to the system drive first. You can still override any of these before running the script:

- `MODELSCOPE_CACHE`
- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_XET_CACHE`

## Recommended defaults

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Torch dtype: `float16` on most consumer NVIDIA GPUs, switch to `bfloat16` only if your environment supports it
- Training type: `LoRA`
- Phase 1 objective: supportive dialogue style
- Phase 2 objective: style alignment only
- Analysis layer: train separately later

Important:

- `Qwen3-4B-Instruct-2507` is the preferred main model for this repo's chat-first route.
- Make sure your environment uses a recent `transformers` build. Qwen3 requires `transformers>=4.51.0`.
- If you switch from a `Qwen2.5` training run to `Qwen3`, restart the staged SFT chain from phase 0. Old LoRA checkpoints are not adapter-compatible across different base model families.

## Local 8GB profile

The `local_8gb` profile is intended for a single-consumer GPU setup.

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Quantization: `bnb` 4-bit
- Max length: `1024`
- Goal: keep the main model feasible on an RTX 4060 / 8GB class GPU

This profile keeps the same mainline base model while using QLoRA to fit smaller VRAM.

## Why this matches the project plan

- It keeps style learning and analysis learning separate.
- It uses real multi-turn support data as the backbone.
- It treats single-turn expansion as auxiliary material instead of the main source.
- It leaves risk rules and structured analysis outside the first style model.
