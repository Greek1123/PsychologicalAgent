from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from campus_support_agent.ms_swift_recipe_builder import build_ms_swift_recipes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ms-swift stage scripts for style training.")
    parser.add_argument(
        "--dataset-manifest",
        default=str(ROOT / "data" / "training" / "ms_swift" / "ms_swift_style_manifest.json"),
        help="Path to the ms-swift dataset manifest.",
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "training" / "ms_swift"),
        help="Output directory for recipe scripts.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional base model override.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Optional explicit torch dtype override.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        choices=["default", "local_8gb"],
        help="Recipe profile. Use local_8gb for an RTX 4060/8GB style machine.",
    )
    parser.add_argument(
        "--general-phase0-dataset",
        default=str(ROOT / "data" / "training" / "general_multiturn" / "general_phase0_train_ms_swift.jsonl"),
        help="Optional general multi-turn warmup dataset. If the file exists, phase-0 scripts will be generated.",
    )
    parser.add_argument(
        "--weak-input-phase0-5-dataset",
        default=str(ROOT / "data" / "training" / "weak_input" / "weak_input_phase0_5_train_ms_swift.jsonl"),
        help="Optional weak-input repair dataset. If the file exists, phase-0.5 scripts will be generated.",
    )
    args = parser.parse_args()

    general_phase0_dataset = args.general_phase0_dataset
    if general_phase0_dataset and not Path(general_phase0_dataset).exists():
        general_phase0_dataset = None
    weak_input_phase0_5_dataset = args.weak_input_phase0_5_dataset
    if weak_input_phase0_5_dataset and not Path(weak_input_phase0_5_dataset).exists():
        weak_input_phase0_5_dataset = None

    manifest = build_ms_swift_recipes(
        args.dataset_manifest,
        args.outdir,
        model=args.model,
        torch_dtype=args.torch_dtype,
        profile=args.profile,
        general_phase0_dataset=general_phase0_dataset,
        weak_input_phase0_5_dataset=weak_input_phase0_5_dataset,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
