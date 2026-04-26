# Safely remove old local training checkpoints.
# This script only deletes the explicit paths listed below after verifying
# they are inside D:\psychologicalAgent\training\ms_swift\outputs.

$base = (Resolve-Path "D:\psychologicalAgent\training\ms_swift\outputs").Path

$targets = @(
  "D:\psychologicalAgent\training\ms_swift\outputs\general_phase0_sft\v3-20260408-191339",
  "D:\psychologicalAgent\training\ms_swift\outputs\general_phase0_sft\v2-20260408-175058",
  "D:\psychologicalAgent\training\ms_swift\outputs\general_phase0_sft\v1-20260408-092818",
  "D:\psychologicalAgent\training\ms_swift\outputs\general_phase0_sft\v0-20260407-232331",
  "D:\psychologicalAgent\training\ms_swift\outputs\style_phase1_sft\v10-20260409-150539",
  "D:\psychologicalAgent\training\ms_swift\outputs\style_phase1_sft\v9-20260408-170105",
  "D:\psychologicalAgent\training\ms_swift\outputs\style_phase1_sft\v8-20260408-165301",
  "D:\psychologicalAgent\training\ms_swift\outputs\style_phase1_sft\v7-20260408-134320",
  "D:\psychologicalAgent\training\ms_swift\outputs\weak_input_phase0_5_sft\v1-20260408-165049",
  "D:\psychologicalAgent\training\ms_swift\outputs\weak_input_phase0_5_sft\v0-20260408-133802"
)

foreach ($target in $targets) {
  if (!(Test-Path -LiteralPath $target)) {
    Write-Output "Skipped missing $target"
    continue
  }

  $resolved = (Resolve-Path -LiteralPath $target).Path
  if (-not $resolved.StartsWith($base, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to delete outside outputs: $resolved"
  }

  Remove-Item -LiteralPath $resolved -Recurse -Force
  Write-Output "Deleted $resolved"
}
