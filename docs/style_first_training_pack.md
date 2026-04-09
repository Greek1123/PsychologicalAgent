# Style-First Training Pack

This workflow follows the user's 8-step plan:

1. Train supportive speaking style first.
2. Use real multi-turn support dialogues for SFT.
3. Add expanded single-turn data only as auxiliary material.
4. Build preference annotation files separately for style alignment.
5. Keep analysis ability as a second layer.
6. Preserve `stage_goal` in every training record.
7. Evaluate with human counselors.
8. Add risk rules and scales after the style model is stable.

## What the pack contains

After running the builder, the output directory contains:

- `style_phase1_train.jsonl`
  Real multi-turn training data plus a capped amount of expanded single-turn data.
- `style_phase1_dev.jsonl`
  Real multi-turn dev data.
- `style_phase1_test.jsonl`
  Real multi-turn test data.
- `style_phase2_preference.jsonl`
  Preference annotation templates for DPO/ORPO style alignment.
- `style_training_manifest.json`
  Counts, file paths, and recommended training order.

## Why the pack is structured this way

- Real multi-turn data stays the core of phase 1.
- Real multi-turn data is normalized before packing so phase 1 does not start from assistant-led mid-session snippets.
- Weak openers such as bare acknowledgements (`嗯`, `好`, `yes`) or digit-only turns are dropped because they often teach template continuation instead of real dialogue.
- Expanded single-turn data is sampled with a ratio cap so it does not dominate style learning.
- Preference alignment is prepared as a separate phase instead of being mixed into SFT.
- Human evaluation can be run on the phase 1 and phase 2 checkpoints with the existing evaluation sheet builder.

## Recommended order

1. Run phase 0 general multi-turn SFT first if the model still feels rigid.
2. Run phase 0.5 weak-input repair SFT so short inputs do not collapse into loops.
3. Run light phase 1 support-style SFT with `style_phase1_train.jsonl`.
4. Validate on `style_phase1_dev.jsonl`.
5. Keep `style_phase1_test.jsonl` for held-out human evaluation.
6. Fill `style_phase2_preference.jsonl` with `chosen/rejected` pairs only after the model already chats normally.
7. Run DPO/ORPO for style alignment when the remaining gap is mostly style.
8. Train the analysis layer later with structured labels such as `risk`, `entropy`, and `entropy_reduction`.
