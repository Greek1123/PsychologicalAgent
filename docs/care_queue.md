# Care Queue

The care queue turns the latest state of each session into a prioritized backend worklist. It is designed for the campus psychological entropy reduction loop, where the system needs to know which sessions require observation, strategy repair, human follow-up, or urgent safety handling.

## API

`GET /api/v1/analytics/care-queue`

Query parameters:

- `limit`: maximum number of queue items. Default: `100`.
- `include_low_priority`: whether to include ordinary observation sessions. Default: `false`.
- `include_resolved`: whether to include sessions already marked `resolved` or `closed` by a human handler. Default: `false`.

Human handling APIs:

- `POST /api/v1/sessions/{session_id}/human-interventions`
- `GET /api/v1/sessions/{session_id}/human-interventions`

Supported human intervention statuses:

- `acknowledged`
- `in_progress`
- `escalated`
- `resolved`
- `closed`

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
      "risk_level": "high",
      "evidence": {
        "human_intervention": {
          "status": "acknowledged",
          "handler_id": "counselor-001"
        }
      }
    }
  ]
}
```

Example human intervention request:

```json
{
  "response_id": "support_xxx",
  "status": "acknowledged",
  "handler_id": "counselor-001",
  "note": "已查看高优先级队列，准备线下跟进。",
  "next_action": "contact_student_with_low_pressure_checkin",
  "tags": ["manual_followup", "same_day_review"]
}
```

## Priority Rules

- `critical`: urgent safety route.
- `high`: human follow-up route.
- `medium`: monitor next turn or repair reply style.
- `low`: continue observation. Hidden by default to keep the queue focused.

## Project Meaning

This module makes the system operational. Instead of only reporting analysis fields, it produces a current worklist for follow-up, dashboard display, and future human-in-the-loop workflows.

With human intervention records, the queue is now a closed loop: a session can enter the queue through risk, entropy trend, referral, or strategy deterioration, then a counselor can acknowledge, escalate, resolve, or close it. Resolved and closed sessions are hidden from the open queue by default, while still available for audit with `include_resolved=true`.
