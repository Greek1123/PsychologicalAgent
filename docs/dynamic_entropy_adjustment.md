# Campus Entropy Dynamic Adjustment Layer

## Purpose

This layer turns a single support reply into a tracked intervention loop. It uses the current psychological entropy score, risk level, state profile, and recent entropy trace to decide whether the system should maintain, soften, escalate, or refer.

The user-facing reply should remain supportive and natural. The dynamic adjustment result is an internal/system-facing field for backend routing, front-end dashboards, review logs, and future training data.

## Output Field

Each support response can include:

```json
{
  "dynamic_adjustment": {
    "adjustment_id": "rising_watch:soften_and_stabilize",
    "stability_state": "rising_watch",
    "action": "soften_and_stabilize",
    "intensity_shift": "increase",
    "trend_direction": "up",
    "trend_delta": 8,
    "should_modify_strategy": true,
    "should_refer": false,
    "review_window_hours": 24,
    "next_focus": "slow_down_and_identify_trigger",
    "reasons": ["risk:medium", "entropy_score:58", "trend:rising"]
  }
}
```

## Decision States

- `first_observation`: first entropy point in a session; use baseline support and start tracking.
- `stable_watch`: entropy is stable; keep current strategy and continue observing.
- `improving`: entropy is falling; consolidate what is working and avoid over-intervening.
- `rising_watch`: entropy is rising; reduce pressure, ask less, and stabilize the immediate moment.
- `escalating_entropy`: entropy rises sharply or high entropy keeps climbing; increase support intensity.
- `sustained_high_entropy`: entropy remains high across recent turns; recommend human follow-up watch.
- `high_risk_watch`: high risk content appears; route toward human follow-up.
- `crisis`: critical risk content appears; route to urgent safety support.

## Current Integration

- `src/campus_support_agent/dynamic_adjustment.py` calculates the decision.
- `src/campus_support_agent/agent.py` adds a baseline decision and can use previous entropy before generating the current reply.
- `src/campus_support_agent/main.py` passes recent entropy history into the agent, then stores a final session-aware decision.
- `src/campus_support_agent/strategy_execution.py` uses the decision to lower or raise reply intensity before the user sees the message.
- `src/campus_support_agent/storage.py` exposes dynamic adjustment fields in session analysis and overview analytics.

## Next Step

The next backend milestone is to connect user feedback with this layer. If a reply is marked unhelpful, the next turn should reduce question pressure, avoid repeated templates, and generate a review case for later SFT/DPO refinement.
