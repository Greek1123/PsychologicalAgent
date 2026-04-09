# ms-swift Installation Notes

If PowerShell reports:

`swift : 无法将“swift”项识别为 cmdlet、函数、脚本文件或可运行程序的名称`

it means the `ms-swift` CLI is not installed in the current Python environment, or the current shell has not picked up the installed CLI yet.

## Recommended setup

1. Create and activate a dedicated virtual environment.
2. Install `ms-swift` into that environment.
3. Confirm `swift --help` works.
4. Run the generated training scripts from the same activated environment.

## Windows example

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "ms-swift" -U
swift --help
```

You can also use the helper script:

```powershell
powershell -ExecutionPolicy Bypass -File .\training\ms_swift\install_ms_swift.ps1
```

## Important note

The generated `run_style_phase1_sft.ps1` and `run_style_phase2_dpo.ps1` scripts assume:

- `ms-swift` is already installed
- the current shell is using the same Python environment where `ms-swift` was installed

If `swift --help` still fails after installation, close the terminal, reopen it, reactivate the environment, and try again.

## Qwen3 note

If you switch the recipe to `Qwen/Qwen3-4B-Instruct-2507`, keep your inference and training stack reasonably current.

- `transformers>=4.51.0` is recommended for Qwen3
- if you see errors such as `KeyError: 'qwen3'`, upgrade `transformers` first
