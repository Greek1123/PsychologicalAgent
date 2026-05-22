from __future__ import annotations

from typing import Any

from .schemas import FeedbackAdaptation


QUESTION_PRESSURE_TAGS = {
    "too_many_questions",
    "pressure_too_high",
    "forced_disclosure",
    "over_questioning",
}
DETAIL_TOO_LOW_TAGS = {"too_short", "too_generic", "not_actionable", "empty_comfort"}
REPETITION_TAGS = {"repetitive", "template_reply", "robotic", "looping"}
MISSED_CONTEXT_TAGS = {"missed_context", "privacy_missed", "wrong_focus", "not_listening"}


def build_feedback_adaptation(
    *,
    feedback_summary: dict[str, Any] | None = None,
    recent_feedback: list[dict[str, Any]] | None = None,
) -> FeedbackAdaptation:
    summary = feedback_summary or {}
    recent = recent_feedback or []
    common_tags = summary.get("common_tags") or {}
    recent_tags = _recent_tags(recent)
    tags = sorted(set(common_tags) | set(recent_tags))
    negative_count = int(summary.get("negative_count") or 0)
    positive_count = int(summary.get("positive_count") or 0)
    average_score = summary.get("average_helpful_score")

    reasons = [f"negative_count:{negative_count}", f"positive_count:{positive_count}"]
    if average_score is not None:
        reasons.append(f"average_helpful_score:{average_score}")
    reasons.extend(f"tag:{tag}" for tag in tags[:8])

    mode = "standard"
    question_pressure = "normal"
    detail_level = "normal"
    should_avoid_repetition = False
    should_collect_bad_case = negative_count > 0
    preferred_moves: list[str] = []

    if negative_count >= 2 or (average_score is not None and float(average_score) < 0):
        mode = "repair_next_turn"
        preferred_moves.append("acknowledge_possible_miss")

    if any(tag in QUESTION_PRESSURE_TAGS for tag in tags):
        question_pressure = "low"
        preferred_moves.append("ask_at_most_one_optional_question")

    if any(tag in DETAIL_TOO_LOW_TAGS for tag in tags):
        detail_level = "more_concrete"
        preferred_moves.append("give_one_specific_small_step")

    if any(tag in REPETITION_TAGS for tag in tags):
        should_avoid_repetition = True
        preferred_moves.append("avoid_reusing_previous_opening")

    if any(tag in MISSED_CONTEXT_TAGS for tag in tags):
        preferred_moves.append("reflect_user_context_first")

    if negative_count == 0 and positive_count >= 2:
        mode = "keep_working_pattern"
        preferred_moves.append("preserve_current_style")

    return FeedbackAdaptation(
        adaptation_id=f"{mode}:{question_pressure}:{detail_level}",
        mode=mode,
        question_pressure=question_pressure,
        detail_level=detail_level,
        should_avoid_repetition=should_avoid_repetition,
        should_collect_bad_case=should_collect_bad_case,
        avoid_tags=tags,
        preferred_moves=_dedupe(preferred_moves),
        reasons=_dedupe(reasons),
    )


def _recent_tags(recent_feedback: list[dict[str, Any]]) -> list[str]:
    tags: list[str] = []
    for item in recent_feedback[-5:]:
        tags.extend(str(tag).strip() for tag in item.get("tags", []) if str(tag).strip())
        if int(item.get("helpful_score") or 0) < 0:
            tags.append("recent_negative_feedback")
    return tags


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
