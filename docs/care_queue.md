# Care Queue

The care queue turns the latest state of each session into a prioritized backend worklist. It is designed for the campus psychological entropy reduction loop, where the system needs to know which sessions require observation, strategy repair, human follow-up, or urgent safety handling.

## API

`GET /api/v1/analytics/care-queue`

Query parameters:

- `limit`: maximum number of queue items. Default: `100`.
- `include_low_priority`: whether to include ordinary observation sessions. Default: `false`.

Example response:

```json
{
  "total_items": 2,
  "priority_counts": {
    "high": 1,
    "medium": 1
  },
  "route_counts": {
    "human_followup_recommended": 1,
    "monitor_next_turn": 1
  },
  "outcome_counts": {
    "needs_human_followup": 1,
    "watching": 1
  },
  "items": [
    {
      "session_id": "student-001",
      "priority": "high",
      "route": "human_followup_recommended",
      "outcome_status": "needs_human_followup",
      "recommended_action": "recommend_human_followup",
      "latest_entropy_score": 72,
      "risk_level": "high"
    }
  ]
}
```

## Priority Rules

- `critical`: urgent safety route.
- `high`: human follow-up route.
- `medium`: monitor next turn or repair reply style.
- `low`: continue observation. Hidden by default to keep the queue focused.

## Project Meaning

This module makes the system operational. Instead of only reporting analysis fields, it produces a current worklist for follow-up, dashboard display, and future human-in-the-loop workflows.
