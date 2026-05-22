# Care Pathway Orchestration

This layer turns the backend analysis stack into one executable care route. It is designed for the campus psychological entropy reduction project, where the system should not only generate a reply, but also decide what the next support action should be.

## Position In The Backend

Input layers:

- `state_profile`: current user state and stress domains.
- `dynamic_adjustment`: entropy movement and next entropy intervention.
- `feedback_adaptation`: user feedback about whether prior replies helped.
- `longitudinal_profile`: multi-turn state course, risk course, and care level.

Output layer:

- `care_pathway`: one route that the backend and frontend can use.

## Routes

- `continue_observation`: normal supportive chat, continue storing context and entropy.
- `monitor_next_turn`: entropy is rising, volatile, or needs closer tracking.
- `repair_reply_style`: user feedback suggests the reply style needs repair.
- `human_followup_recommended`: the user may need campus counseling or manual follow-up.
- `urgent_safety`: crisis or urgent safety route. AI-only reply should not be the only support.

## API Location

`GET /api/v1/sessions/{session_id}/analysis` now includes:

```json
{
  "care_pathway": {
    "route": "monitor_next_turn",
    "priority": "medium",
    "user_visible_mode": "low_pressure_stabilization",
    "review_window_hours": 24,
    "should_notify_human": false,
    "should_pause_ai_only_reply": false,
    "backend_actions": [
      "keep_intervention_small",
      "compare_entropy_next_turn",
      "check_support_access_if_score_rises"
    ]
  }
}
```

`GET /api/v1/analytics/overview` now includes aggregate pathway fields:

```json
{
  "care_pathway_routes": {
    "continue_observation": 12,
    "monitor_next_turn": 3,
    "human_followup_recommended": 1
  },
  "care_pathway_priorities": {
    "low": 12,
    "medium": 3,
    "high": 1
  },
  "current_care_pathway_routes": {
    "continue_observation": 8,
    "monitor_next_turn": 2
  }
}
```

The `care_pathway_routes` field counts pathway observations across recent records. The `current_care_pathway_routes` field only counts the latest known route per session, which is more suitable for a dashboard showing the current care workload.

## Project Meaning

This module is the bridge from "chatbot" to "dynamic adjustment system". It makes the psychological entropy result actionable without exposing technical labels to the user. The user still receives a natural support response, while the backend tracks whether the system should observe, repair, monitor, recommend human follow-up, or activate safety handling.
