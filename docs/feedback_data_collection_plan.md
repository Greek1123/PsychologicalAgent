# Feedback Data Collection Plan

This project should collect feedback data from real model failures, not from random prompts. The goal is to build small but high-quality pairs:

```text
conversation context + user input + bad model reply -> better human reply
```

These pairs can later become DPO data. If a bad model reply is missing, the rewritten answer can still become SFT data.

## Daily Target

For the current stage, collect data in three layers:

- `50` rows: quick pipeline check. This is enough to test review, conversion, and DPO scripts.
- `200` rows: first useful behavior alignment set. This may start reducing repeated bad habits.
- `1000+` rows: stable project dataset. This is where the model can noticeably learn your preferred support style.

Do not chase quantity before quality. A carefully rewritten 200-row set is more useful than 2000 vague rows.

## Useful Bad Cases

Mark a reply as bad when it has one or more of these problems:

- `privacy_missed`: user says they are afraid others will know, but the assistant does not reassure privacy and boundaries.
- `too_short`: reply is so short that it does not provide support, reflection, or a next step.
- `repetitive`: repeats the same sentence or same structure across turns.
- `pushy`: keeps asking the user to explain when the user says they do not want to.
- `topic_drift`: user talks about distress, but assistant talks about movies, food, coffee, or unrelated content.
- `number_following`: user inputs `1/2/3`, assistant continues counting or treats it as a literal answer.
- `professional_jargon`: exposes entropy, risk, or backend analysis language directly to the user.
- `unsafe`: misses self-harm, harm-to-others, abuse, panic, or emergency signals.
- `fake_identity`: claims to be ChatGPT, DeepSeek, Doubao, a human name, or gives unstable identity.
- `time_hallucination`: invents current date, current movie releases, or time-sensitive facts.

## Good Chosen Replies

A chosen reply should usually include four parts, but keep it natural:

- First, acknowledge the user's immediate feeling in plain language.
- Second, respond to the exact concern instead of changing topic.
- Third, give the user control, especially when they do not want to explain.
- Fourth, offer one small next step or one gentle question.

Example:

```text
你不用现在把细节都说出来，我会尊重你的节奏。这里的对话只用于当前支持，不会主动告诉你的舍友或同学。
如果你愿意，我们可以先不谈发生了什么，只先处理“害怕被知道”的感觉。你现在更需要我陪你安静待一会儿，还是帮你想一个保护隐私的小办法？
```

## How To Fill review_sheet.csv

Fill only these columns:

- `mark_bad`: write `1` if the model reply is bad.
- `problem_tags`: choose tags from the list above, separated by English commas.
- `chosen`: write a better answer in Chinese.
- `review_note`: optional, write why the original answer was bad.

Do not edit these columns unless necessary:

- `id`
- `response_id`
- `session_id`
- `input_text`
- `assistant_reply`
- `risk_level`
- `entropy_score`
- `local_policy`

## Minimum Quality Rules

Before a row is used for training, check:

- The chosen reply must directly answer the latest user message.
- The chosen reply should not reveal backend words like `心理熵`, `风险分级`, or `熵减`.
- The chosen reply should not sound like a questionnaire unless the user asked for assessment.
- The chosen reply should not force the user to disclose details.
- The chosen reply should not only say `别担心`.
- The chosen reply should not diagnose the user.
- For privacy fears, it must explicitly reassure confidentiality and boundaries.
- For weak input like `?`, `嗯`, `1`, it should repair the conversation rather than follow the symbol.

## Weekly Workflow

1. Start the backend and chat with the model normally.
2. Test common campus scenes: dorm conflict, exam anxiety, sleep problems, family pressure, relationship stress, privacy fear, and weak inputs.
3. When a reply feels bad, submit feedback or export review cases.
4. Build `review_sheet.csv`.
5. Each reviewer fills 20-30 rows.
6. Apply the sheet back to JSONL.
7. Convert reviewed rows into DPO data.
8. Train only after the DPO-ready count is at least 50 for testing, preferably 200 for improvement.

## Commands

Export recent replies for manual screening:

```powershell
python scripts\export_training_data.py --format review_case --limit 100 --out data\training\feedback_bad_cases\review_cases.jsonl
```

Build a CSV review sheet:

```powershell
python scripts\build_feedback_review_sheet.py build --input data\training\feedback_bad_cases\review_cases.jsonl --out data\training\feedback_bad_cases\review_sheet.csv
```

Use an external Chinese AI to draft `chosen` automatically:

```powershell
$env:REVIEW_LLM_BASE_URL="https://api.deepseek.com/v1"
$env:REVIEW_LLM_MODEL="deepseek-chat"
$env:REVIEW_LLM_API_KEY="你的API_KEY"
python scripts\autofill_feedback_review_sheet_with_ai.py --input data\training\feedback_bad_cases\review_sheet.csv --out data\training\feedback_bad_cases\review_sheet_ai.csv --limit 50
```

The script sends redacted text by default. It replaces common phone numbers, emails, ID-like numbers, and WeChat/QQ IDs before sending the prompt.

Apply the edited CSV:

```powershell
python scripts\build_feedback_review_sheet.py apply --input data\training\feedback_bad_cases\review_cases.jsonl --sheet data\training\feedback_bad_cases\review_sheet_ai.csv --out data\training\feedback_bad_cases\review_cases_reviewed.jsonl
```

Build DPO data:

```powershell
python scripts\build_feedback_dpo_dataset.py --input data\training\feedback_bad_cases\review_cases_reviewed.jsonl --out data\training\feedback_bad_cases\feedback_preference.jsonl --ms-swift-out data\training\feedback_bad_cases\feedback_dpo_ms_swift.jsonl
```
