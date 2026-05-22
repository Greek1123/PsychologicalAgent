# Entropy Reduction Outcome

This layer evaluates whether the system is actually helping the session move toward lower psychological entropy. It is not a user-facing diagnosis. It is a backend evaluation signal for tracking intervention effectiveness.

## Inputs

- Entropy trace across the session.
- Stored support responses and risk levels.
- User feedback summary.
- Current care pathway decision.

## Session Output

`GET /api/v1/sessions/{session_id}/analysis` includes:

```json
{
  "entropy_reduction_outcome": {
    "status": "improving",
    "effectiveness_score": 78,
    "entropy_delta": -13,
    "risk_shift": "flat",
    "feedback_signal": "positive",
    "pathway_signal": "stable_support",
    "next_action": "maintain_working_strategy"
  }
}
```

Common statuses:

- `insufficient_data`: not enough turns or feedback yet.
- `improving`: entropy is falling without increased risk.
- `stable_helpful`: stable entropy with positive user feedback.
- `stable_observe`: stable enough for continued observation.
- `watching`: needs next-turn monitoring.
- `deteriorating`: entropy or risk is rising.
- `needs_strategy_repair`: user feedback suggests the intervention style is not working.
- `needs_human_followup`: human or offline support should be linked.
- `crisis_priority`: safety protocol has priority over normal entropy-reduction evaluation.

## Overview Output

`GET /api/v1/analytics/overview` includes:

```json
{
  "entropy_outcome_statuses": {
    "stable_observe": 18,
    "watching": 4,
    "needs_strategy_repair": 2
  },
  "current_entropy_outcome_statuses": {
    "stable_observe": 9,
    "watching": 1
  }
}
```

The first field counts recent records. The `current_` field counts only the latest known status for each session, which is better for a live dashboard.

## Project Meaning

This makes the project measurable: the system can report not only the current psychological entropy score, but also whether its intervention is associated with improvement, stability, deterioration, or the need for strategy repair.
