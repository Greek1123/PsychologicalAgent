# Weak Input Repair Warmup

Use this stage after general multi-turn warmup and before support-style SFT.

## Why this stage exists

Small local models often collapse when the user sends:

- `?`
- `嗯`
- `哦`
- `1`
- `...`
- `不知道`

Without dedicated SFT data, the model may:

- loop on generic support templates
- continue counting after numeric input
- mirror filler tokens back to the user
- lose topic grounding

## What this stage teaches

- acknowledge hesitation without getting stuck
- clarify ambiguous input
- re-anchor the dialogue topic
- avoid mechanical counting or filler-token echoing

## Current dataset design

- bilingual curated records
- explicit repairs for `? / 嗯 / 1 / 不知道 / ...`
- discourage counting continuation and filler echoing
- tuned for a small local model before support-style SFT

## Build command

```powershell
python scripts/build_weak_input_dataset.py
```

Default output:

- `data/training/weak_input/weak_input_phase0_5_train_ms_swift.jsonl`

## Recommended order

1. `Phase 0`: general multi-turn dialogue warmup
2. `Phase 0.5`: weak-input repair warmup
3. `Phase 1`: light support-style SFT
