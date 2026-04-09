from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .logging_utils import get_logger


logger = get_logger("style_preference_annotation")


FAILURE_MODE_GUIDE: dict[str, dict[str, str]] = {
    "formulaic_opening": {
        "label": "Formulaic opening",
        "description": "The reply starts with overused support phrases and feels interchangeable.",
    },
    "generic_reassurance": {
        "label": "Generic reassurance",
        "description": "The reply offers broad comfort but does not move the conversation forward.",
    },
    "weak_followup": {
        "label": "Weak follow-up",
        "description": "The question is too broad or does not build on the user's specific context.",
    },
    "premature_solution": {
        "label": "Premature solution",
        "description": "The reply jumps into advice before enough emotional containment or clarification.",
    },
    "therapy_heavy": {
        "label": "Therapy-heavy tone",
        "description": "The reply sounds too much like a formal therapy script instead of a natural support chat.",
    },
    "summary_drift": {
        "label": "Summary drift",
        "description": "The reply paraphrases inaccurately or over-interprets the user's concern.",
    },
}


ZH_FORMULAIC_MARKERS = (
    "听起来",
    "感谢你的分享",
    "这是可以理解的",
    "我能理解",
    "别太担心",
    "慢慢来",
)

EN_FORMULAIC_MARKERS = (
    "it sounds like",
    "thank you for sharing",
    "that makes sense",
    "it is understandable",
    "take it one step at a time",
    "do not worry too much",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _flatten_prompt(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = str(message.get("content", "")).strip()
        if content:
            rendered.append(f"{role}: {content}")
    return "\n".join(rendered)


def _infer_failure_modes(record: dict[str, Any]) -> list[str]:
    chosen = str(record.get("chosen", "")).strip()
    review_notes = str(record.get("review_notes", "")).lower()
    lowered = chosen.lower()
    language = str(record.get("language", "")).lower()
    modes: list[str] = []

    # Look for high-frequency support openers that often make the model sound canned.
    markers = EN_FORMULAIC_MARKERS if language == "en" else ZH_FORMULAIC_MARKERS
    if any(marker in lowered for marker in markers) or any(marker in chosen for marker in markers):
        modes.append("formulaic_opening")

    if "therapy_heavy" in review_notes or "softer" in review_notes:
        modes.append("therapy_heavy")
    if "summary" in review_notes:
        modes.append("summary_drift")

    question_count = chosen.count("?") + chosen.count("？")
    if question_count:
        modes.append("weak_followup")

    lowered_generic = lowered.replace("-", " ")
    generic_signals = (
        "everything will be okay",
        "stay positive",
        "慢慢会好起来",
        "积极一点",
        "不要想太多",
    )
    if any(signal in lowered_generic for signal in generic_signals) or any(signal in chosen for signal in generic_signals):
        modes.append("generic_reassurance")

    action_signals = (
        "you should",
        "try to",
        "可以先",
        "你应该",
        "建议你",
    )
    if any(signal in lowered_generic for signal in action_signals) or any(signal in chosen for signal in action_signals):
        modes.append("premature_solution")

    if not modes:
        modes.append("generic_reassurance")

    return list(dict.fromkeys(modes))


def _candidate_rejected(language: str, failure_modes: list[str]) -> str:
    # The candidate is intentionally mediocre so annotators can refine it into a true rejected answer.
    if language.lower() == "en":
        if "premature_solution" in failure_modes:
            return (
                "It sounds like you are going through a lot right now. "
                "You should stay positive, calm down first, and make a plan so things can get better soon."
            )
        return (
            "It sounds like you are under a lot of pressure right now. "
            "Do not worry too much. Things will get better if you take it one step at a time."
        )

    if "premature_solution" in failure_modes:
        return "听起来你现在压力很大。你先调整心态，积极一点，再给自己列个计划，问题会慢慢解决的。"
    return "听起来你最近真的不容易。先别太担心，一切都会慢慢好起来的，你可以先放松一下。"


def _annotation_goal(failure_modes: list[str]) -> str:
    priorities = []
    for mode in failure_modes:
        guide = FAILURE_MODE_GUIDE.get(mode)
        if guide:
            priorities.append(guide["label"])
    return "; ".join(priorities)


def build_style_dpo_annotation_sheet(
    input_path: str,
    output_path: str,
    *,
    include_candidate: bool = True,
) -> int:
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(source)
    fieldnames = [
        "id",
        "language",
        "stage_goal",
        "annotation_goal",
        "failure_modes",
        "review_notes",
        "prompt_text",
        "chosen",
        "candidate_rejected",
        "rejected",
        "annotator_notes",
    ]

    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            prompt = record.get("prompt", [])
            failure_modes = _infer_failure_modes(record)
            row = {
                "id": record.get("id", ""),
                "language": record.get("language", ""),
                "stage_goal": record.get("stage_goal", ""),
                "annotation_goal": _annotation_goal(failure_modes),
                "failure_modes": "|".join(failure_modes),
                "review_notes": record.get("review_notes", ""),
                "prompt_text": _flatten_prompt(prompt if isinstance(prompt, list) else []),
                "chosen": record.get("chosen", ""),
                "candidate_rejected": _candidate_rejected(str(record.get("language", "")), failure_modes)
                if include_candidate
                else "",
                "rejected": record.get("rejected", ""),
                "annotator_notes": "",
            }
            writer.writerow(row)

    logger.info(
        "Built DPO annotation sheet at %s with %s rows (include_candidate=%s)",
        output,
        len(records),
        include_candidate,
    )
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSV annotation sheet for style DPO rejected responses.")
    parser.add_argument("--input", required=True, help="Input style preference JSONL.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument(
        "--no-candidate",
        action="store_true",
        help="Do not prefill candidate rejected responses.",
    )
    args = parser.parse_args()

    count = build_style_dpo_annotation_sheet(
        args.input,
        args.out,
        include_candidate=not args.no_candidate,
    )
    print(f"wrote {count} rows")


if __name__ == "__main__":
    main()
