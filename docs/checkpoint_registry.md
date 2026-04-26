# Checkpoint Registry

This file records the local training checkpoints that matter for the project.
Large model folders should stay local and should not be committed to GitHub.

## Current Recommended Chain

Use this chain for normal testing:

1. Base model
   - `D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507`

2. Stable public SFT adapter
   - `D:\psychologicalAgent\training\ms_swift\outputs\public_phase0_sft\v0-20260426-134431\checkpoint-465`
   - Status: current stable baseline
   - Notes: best checkpoint to share with teammates for now

3. Next behavior adapter
   - Status: not selected yet
   - Notes: `curated_behavior_sft\v0-20260426-171433\checkpoint-20` failed manual testing and should not be used.

## Not Recommended As Final

These are useful experiment records, but should not be shared as the current final model:

- `D:\psychologicalAgent\training\ms_swift\outputs\public_weak_input_phase0_5_sft\v1-20260426-160403\checkpoint-27`
  - Issue: weak-input overfitting, awkward responses around numbers and privacy

- `D:\psychologicalAgent\training\ms_swift\outputs\public_weak_input_mixed_sft\v1-20260426-163126\checkpoint-76`
  - Issue: role boundary still unstable, sometimes gives generic therapy-plan responses

- `D:\psychologicalAgent\training\ms_swift\outputs\curated_behavior_sft\v0-20260426-171433\checkpoint-20`
  - Issue: worse than the public baseline in manual testing; asks directly for details, gives too-short replies, and still mishandles weak inputs such as `1`

- `D:\psychologicalAgent\training\ms_swift\outputs\curated_behavior_sft\v1-20260426-173422\checkpoint-63`
  - Issue: still worse than the public baseline; role boundary unstable, overuses guided breathing, mishandles `1`, and invents assistant states such as feeling dizzy

## Naming Rules

ms-swift creates a new timestamped folder every time a script runs:

- `v0-YYYYMMDD-HHMMSS`
- `v1-YYYYMMDD-HHMMSS`
- `checkpoint-xx`

This is normal. Do not rename these folders while training or resuming. Instead, record the useful checkpoint path in this file.

## What To Share With Teammates

Minimum files/folders for a teammate who wants to run the same local model:

- Project code from GitHub
- Base model cache or instructions to download `Qwen3-4B-Instruct-2507`
- Current adapter folder:
  - `D:\psychologicalAgent\training\ms_swift\outputs\public_phase0_sft\v0-20260426-134431\checkpoint-465`

Do not share `curated_behavior_sft\v0-20260426-171433\checkpoint-20`; it failed manual testing.

## Next Test Command

Current stable checkpoint test command:

```powershell
python scripts\chat_with_checkpoint.py --checkpoint "D:\psychologicalAgent\training\ms_swift\outputs\public_phase0_sft\v0-20260426-134431\checkpoint-465" --max-new-tokens 512 --temperature 0.7
```

If `.venv` does not have `torch`, use Anaconda Python:

```powershell
D:\Anaconda\python.exe scripts\chat_with_checkpoint.py --checkpoint "D:\psychologicalAgent\training\ms_swift\outputs\public_phase0_sft\v0-20260426-134431\checkpoint-465" --max-new-tokens 512 --temperature 0.7
```
