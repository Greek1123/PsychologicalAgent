$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

$python = "D:\Anaconda\python.exe"
$baseModel = "D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507"
$baseAdapter = Join-Path $projectRoot "training\ms_swift\outputs\refinement_pool_v3_peft\v0-20260511-204204\checkpoint-final"
$dataset = Join-Path $projectRoot "data\training\feedback_bad_cases\refinement_pool_v4_20260520_ms_swift.jsonl"
$outDir = Join-Path $projectRoot "training\ms_swift\outputs\refinement_pool_v4_peft\v0-$timestamp"

if (-not (Test-Path $python)) {
    Write-Error "Python not found: $python"
}
if (-not (Test-Path $baseModel)) {
    Write-Error "Base model not found: $baseModel"
}
if (-not (Test-Path $baseAdapter)) {
    Write-Error "Base adapter not found: $baseAdapter"
}
if (-not (Test-Path $dataset)) {
    Write-Error "Dataset not found: $dataset. Run scripts\build_refinement_pool.py first."
}

Write-Host "Training internal-jargon refinement PEFT patch..."
Write-Host "Base model: $baseModel"
Write-Host "Base adapter: $baseAdapter"
Write-Host "Dataset: $dataset"
Write-Host "Output: $outDir"

& $python (Join-Path $projectRoot "scripts\train_eval_behavior_patch_peft.py") `
    --base-model $baseModel `
    --adapter $baseAdapter `
    --dataset $dataset `
    --out-dir $outDir `
    --cache-root "D:\llm_cache" `
    --max-length 512 `
    --learning-rate 3e-7 `
    --epochs 1 `
    --max-steps -1
