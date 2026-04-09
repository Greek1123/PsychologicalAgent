# General Multi-turn Warmup

Use this stage before style-support SFT if the current model still sounds too rigid, too repetitive, or too weak at ordinary dialogue continuation.

## Why this stage exists

The model should first learn to:

- follow context
- continue a conversation naturally
- avoid collapsing into the same support template
- handle ordinary multi-turn turn-taking

Only after that should the project push it toward campus mental-health support style.

## Recommended order

1. Build the general multi-turn warmup dataset from `dialog_release.json`.
2. Run phase-0 SFT on the general dataset.
3. Run phase-0.5 weak-input repair SFT on short/noisy user inputs.
4. Run phase-1 light support-style SFT on the psychological support dataset.
5. Re-evaluate before deciding whether DPO is needed.

## Build command

```bash
python scripts/build_general_multiturn_dataset.py
```

Default output:

- `data/training/general_multiturn/general_phase0_train_ms_swift.jsonl`

## Important caution

This dataset is for warmup, not for final deployment style. It should improve dialogue flow, but it should not dominate the final campus mental-health model.
