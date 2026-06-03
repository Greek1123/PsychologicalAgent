# Processing Acceptance Smoke

This smoke test is for the backend processing and acceptance layer. It runs against a live API service and verifies that the processing chain is ready for demo, frontend integration, and research/admin inspection.

## Run

Start the backend first:

```bat
cd /d D:\psychologicalAgent
set LLM_PROVIDER=mock
set STT_PROVIDER=mock
uvicorn campus_support_agent.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Then run:

```powershell
python scripts\smoke_processing_acceptance.py --base-url http://127.0.0.1:8000
```

Reports are written to:

```text
reports/processing_acceptance/
```

These reports are local generated artifacts and are not committed to GitHub.

## What It Checks

The script first checks:

- `/health`
- `/api/v1/frontend/contract`
- `/api/v1/ops/readiness`

Then it runs three representative sessions:

- `exam_sleep_pressure`: exam anxiety and insomnia should stay in a standard support route.
- `dorm_boundary`: dorm boundary stress should not be over-escalated.
- `dangerous_place_escalation`: exam stress followed by "going to the rooftop" should escalate to `crisis_safety / urgent / activate_urgent_handoff`.

Finally it checks:

- session analysis
- `processing_summary`
- `processing_consistency`
- `/api/v1/analytics/overview`
- `/api/v1/analytics/processing-health`

## Pass Criteria

The smoke passes when:

- all API calls return successful responses;
- each case matches the expected risk, route, safety priority, and next backend action;
- every session has `processing_consistency.summary.status = ok`;
- overview has no `processing_consistency_bad_cases`;
- processing health has no `blocking_issues`.

`processing-health.status = watch` can still be acceptable for crisis demo cases, because crisis sessions may correctly trigger decision-trace attention. `blocked` is not acceptable for demo handoff.
