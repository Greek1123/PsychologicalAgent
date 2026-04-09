# Local Checkpoint Chat

Use this workflow when `swift infer` on Windows fails to recognize a local adapter path.

## Purpose

- Load a local LoRA checkpoint directly with `transformers + peft`
- Reuse the base model and quantization settings stored in the checkpoint `args.json`
- Prefer the already-downloaded local `model_dir` from `args.json` so the script does not download another copy of the same base model
- Default ModelScope and Hugging Face caches to `D:\llm_cache` so local testing does not fill the system drive
- Test multi-turn chat behavior without exporting or merging the adapter

## Script

- `scripts/chat_with_checkpoint.py`

## Example

```powershell
python scripts\chat_with_checkpoint.py --checkpoint "D:\psychologicalAgent\training\ms_swift\outputs\style_phase1_sft\v6-20260408-114633\checkpoint-74"
```

If you want a different cache location, override it explicitly:

```powershell
python scripts\chat_with_checkpoint.py --checkpoint "D:\...\checkpoint-74" --cache-root "E:\model_cache"
```

Useful controls during chat:

- `/reset`: clear dialogue history
- `/exit`: quit the session

## What to check

- Whether the model can continue a normal multi-turn conversation
- Whether it still falls into template repetition
- Whether digit prompts like `1` still trigger counting behavior
