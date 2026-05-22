# Feedback Adaptation Loop

## Purpose

This layer closes the loop between user feedback and the next intervention. Feedback should not only be stored for future training. It should also influence the next support reply inside the same session.

The target project direction is campus psychological entropy reduction. In that direction, feedback is treated as a signal that the previous intervention may have failed to reduce entropy, failed to match the user's boundary, or created extra pressure.

## Feedback Input

The current feedback endpoint accepts:

```json
{
  "response_id": "support_xxx",
  "helpful_score": -2,
  "mood_after": 35,
  "tags": ["too_many_questions", "missed_context"],
  "user_note": "The answer pushed me to explain too much."
}
```

## Adaptation Output

The next support response can include:

```json
{
  "feedback_adaptation": {
    "adaptation_id": "repair_next_turn:low:normal",
    "mode": "repair_next_turn",
    "question_pressure": "low",
    "detail_level": "normal",
    "should_avoid_repetition": false,
    "should_collect_bad_case": true,
    "avoid_tags": ["missed_context", "too_many_questions"],
    "preferred_moves": [
      "acknowledge_possible_miss",
      "ask_at_most_one_optional_question",
      "reflect_user_context_first"
    ]
  }
}
```

## Current Rules

- Negative feedback switches the next turn into `repair_next_turn`.
- `too_many_questions`, `pressure_too_high`, or `forced_disclosure` lowers question pressure.
- `too_short`, `too_generic`, or `not_actionable` asks the system to provide one more concrete small step.
- `repetitive`, `template_reply`, or `robotic` asks the system to avoid repeating the same opening.
- `missed_context`, `privacy_missed`, or `wrong_focus` asks the system to reflect the user's stated context first.
- Repeated positive feedback switches to `keep_working_pattern`.

## Backend Integration

- `src/campus_support_agent/feedback_adaptation.py` converts stored feedback into adaptation decisions.
- `src/campus_support_agent/main.py` loads recent session feedback before generating the next response.
- `src/campus_support_agent/agent.py` attaches the adaptation object to each response.
- `src/campus_support_agent/strategy_execution.py` uses the adaptation to change the user-visible plan.
- `src/campus_support_agent/storage.py` exposes the latest feedback adaptation in session analysis.

## Training Value

When `should_collect_bad_case` is true, the conversation should be treated as a candidate bad case for later review, SFT repair, or DPO/KTO preference training.

This means the project now has two optimization loops:

- Runtime loop: feedback changes the next reply immediately.
- Training loop: feedback produces bad cases for later dataset construction.
