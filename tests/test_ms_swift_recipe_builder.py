from __future__ import annotations

import json
import shutil
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.ms_swift_recipe_builder import build_ms_swift_recipes


@contextmanager
def _temporary_workspace_dir() -> Path:
    parent = ROOT / "tmp_test_artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    tmpdir = parent / f"ms-swift-recipes-{uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class MsSwiftRecipeBuilderTests(unittest.TestCase):
    def test_local_8gb_profile_defaults_to_qwen3_4b(self) -> None:
        with _temporary_workspace_dir() as root:
            dataset_manifest = root / "ms_swift_style_manifest.json"
            outdir = root / "recipes"
            payload = {
                "files": {
                    "train": str(root / "style_phase1_train_ms_swift.jsonl"),
                    "preference": str(root / "style_phase2_preference_ms_swift.jsonl"),
                }
            }
            dataset_manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            manifest = build_ms_swift_recipes(
                str(dataset_manifest),
                str(outdir),
                profile="local_8gb",
            )

            self.assertEqual(manifest["model"], "Qwen/Qwen3-4B-Instruct-2507")

    def test_mismatched_checkpoint_is_not_reused_after_model_switch(self) -> None:
        with _temporary_workspace_dir() as root:
            dataset_manifest = root / "ms_swift_style_manifest.json"
            outdir = root / "recipes"
            general_phase0 = root / "general_phase0_train_ms_swift.jsonl"
            weak_input_phase0_5 = root / "weak_input_phase0_5_train_ms_swift.jsonl"
            previous_general_ckpt = outdir / "outputs" / "general_phase0_sft" / "v0-test" / "checkpoint-12"
            previous_general_ckpt.mkdir(parents=True, exist_ok=True)
            (previous_general_ckpt / "args.json").write_text(
                json.dumps({"model": "Qwen/Qwen2.5-3B-Instruct"}, ensure_ascii=False),
                encoding="utf-8",
            )

            payload = {
                "files": {
                    "train": str(root / "style_phase1_train_ms_swift.jsonl"),
                    "preference": str(root / "style_phase2_preference_ms_swift.jsonl"),
                }
            }
            dataset_manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            general_phase0.write_text("", encoding="utf-8")
            weak_input_phase0_5.write_text("", encoding="utf-8")

            build_ms_swift_recipes(
                str(dataset_manifest),
                str(outdir),
                profile="local_8gb",
                general_phase0_dataset=str(general_phase0),
                weak_input_phase0_5_dataset=str(weak_input_phase0_5),
            )

            weak_input_script = (outdir / "run_weak_input_phase0_5_sft.ps1").read_text(encoding="utf-8")
            self.assertNotIn(str(previous_general_ckpt), weak_input_script)
            self.assertIn("checkpoint-last", weak_input_script)

    def test_build_ms_swift_recipes_writes_stage_scripts(self) -> None:
        with _temporary_workspace_dir() as root:
            dataset_manifest = root / "ms_swift_style_manifest.json"
            outdir = root / "recipes"
            general_phase0 = root / "general_phase0_train_ms_swift.jsonl"
            weak_input_phase0_5 = root / "weak_input_phase0_5_train_ms_swift.jsonl"
            previous_general_ckpt = outdir / "outputs" / "general_phase0_sft" / "v0-test" / "checkpoint-12"
            previous_general_ckpt.mkdir(parents=True, exist_ok=True)
            (previous_general_ckpt / "args.json").write_text(
                json.dumps({"model": "Qwen/Qwen2.5-7B-Instruct"}, ensure_ascii=False),
                encoding="utf-8",
            )

            payload = {
                "files": {
                    "train": str(root / "style_phase1_train_ms_swift.jsonl"),
                    "preference": str(root / "style_phase2_preference_ms_swift.jsonl"),
                }
            }
            dataset_manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            general_phase0.write_text("", encoding="utf-8")
            weak_input_phase0_5.write_text("", encoding="utf-8")

            manifest = build_ms_swift_recipes(
                str(dataset_manifest),
                str(outdir),
                model="Qwen/Qwen2.5-7B-Instruct",
                general_phase0_dataset=str(general_phase0),
                weak_input_phase0_5_dataset=str(weak_input_phase0_5),
            )

            self.assertTrue((outdir / "run_general_phase0_sft.ps1").exists())
            self.assertTrue((outdir / "run_weak_input_phase0_5_sft.ps1").exists())
            self.assertTrue((outdir / "run_style_phase1_sft.ps1").exists())
            self.assertTrue((outdir / "run_style_phase2_dpo.ps1").exists())
            self.assertIn("Qwen/Qwen2.5-7B-Instruct", manifest["model"])
            self.assertEqual(manifest["general_phase0_dataset"], str(general_phase0))
            self.assertEqual(manifest["weak_input_phase0_5_dataset"], str(weak_input_phase0_5))

            weak_input_script = (outdir / "run_weak_input_phase0_5_sft.ps1").read_text(encoding="utf-8")
            self.assertIn(str(previous_general_ckpt), weak_input_script)
            self.assertIn("$env:MODELSCOPE_CACHE", weak_input_script)
            self.assertIn("D:\\llm_cache\\modelscope", weak_input_script)


if __name__ == "__main__":
    unittest.main()
