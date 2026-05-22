# Longitudinal State Profile

## Purpose

The longitudinal state profile summarizes a student's state across multiple turns in the same session. It is designed for the campus psychological entropy reduction loop: the system should not only answer the current message, but also track whether the student's entropy, risk, and support needs are changing over time.

## Output Location

`GET /api/v1/sessions/{session_id}/analysis` now returns:

```json
{
  "longitudinal_profile": {
    "profile_id": "student-001:longitudinal:6",
    "session_id": "student-001",
    "observation_count": 6,
    "dominant_states": ["academic_sleep_stress", "privacy_boundary"],
    "dominant_stress_domains": ["academic", "sleep"],
    "entropy_course": "rising",
    "risk_course": "repeated_medium",
    "average_entropy": 52.3,
    "latest_entropy_score": 61,
    "peak_entropy_score": 70,
    "volatility_score": 18,
    "engagement_signal": "active",
    "recommended_care_level": "watch_closely",
    "next_review_hours": 24,
    "priority_actions": [
      "reduce_intervention_pressure",
      "track_entropy_next_turn"
    ]
  }
}
```

## Key Fields

- `dominant_states`: Most frequent internal state labels across the session.
- `dominant_stress_domains`: Main stress areas such as academic, sleep, dorm, social, or family.
- `entropy_course`: Current entropy trend: `baseline`, `stable`, `rising`, `falling`, `volatile`, or `sustained_high`.
- `risk_course`: Whether risk has stayed low, repeatedly appeared at medium level, reached high risk, or included crisis content.
- `recommended_care_level`: System-level care route: `observe`, `strategy_repair`, `watch_closely`, `manual_followup`, or `urgent`.
- `priority_actions`: Backend actions that should guide the next response, dashboard, or human review.

## Integration

- `src/campus_support_agent/longitudinal_profile.py` calculates the profile.
- `src/campus_support_agent/storage.py` adds the profile to session analysis.
- `tests/test_longitudinal_profile.py` covers sustained high entropy, negative-feedback repair, and critical-risk override.

## Project Meaning

This layer helps shift the project from a simple chatbot to a dynamic adjustment system. A single reply is not enough for psychological support. The system needs to observe entropy changes, detect repeated stress patterns, and decide whether the next step should be observation, strategy repair, closer monitoring, human follow-up, or urgent support.
