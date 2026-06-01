# 后端处理层说明

处理层负责把用户输入转成可执行的心理支持决策。它不是前端展示层，也不是单纯模型生成层，而是连接输入、多模态信号、风险识别、心理熵、策略选择、动态调整、转介和最终回复的后端流水线。

## 当前处理链路

每次 `POST /api/v1/support/text` 或 `POST /api/v1/support/audio` 会进入同一条处理链。

```text
输入清洗
-> 噪声/错字痛苦信号修复
-> 风险识别
-> 心理熵评估
-> 状态画像
-> 校园资源检索
-> 熵减策略
-> 干预策略选择
-> 动态调整
-> 本地策略或 LLM 生成
-> 最终回复护栏
-> 处理摘要输出
```

语音输入会先经过：

```text
音频元数据/基础信号分析 -> STT transcript -> 文本处理链
```

## 新增字段：processing_summary

后端响应现在包含 `processing_summary`，用于研究端、管理端和调试端查看本轮处理层决策。学生端角色视图不会默认暴露该字段。

主要字段：

| 字段 | 含义 |
| --- | --- |
| `route` | 本轮处理路由，如 `local_policy`、`llm_or_fallback`、`crisis_safety`。 |
| `input_mode` | 输入来源，如 `text` 或 `audio`。 |
| `reply_source` | 回复来源，如本地策略、LLM、兜底或危机模板。 |
| `safety_priority` | 安全优先级：`standard`、`human_followup`、`urgent`。 |
| `risk_level` | 风险等级。 |
| `entropy_score` | 当前心理熵分数。 |
| `balance_state` | 动态平衡状态。 |
| `primary_state` | 状态画像主状态。 |
| `strategy_id` | 选中的干预策略。 |
| `dynamic_action` | 动态调整动作。 |
| `orchestration_route` | 熵减编排路线。 |
| `referral_urgency` | 转介紧急程度。 |
| `should_refer` | 是否建议人工/现实支持。 |
| `local_policy_name` | 如果命中本地策略，显示策略名。 |
| `completed_stages` | 本轮已完成的处理阶段。 |
| `decision_reasons` | 后端决策摘要。 |
| `next_backend_action` | 下一步后端动作建议。 |

## 路由说明

### local_policy

命中本地确定性策略，例如隐私担忧、考试焦虑、宿舍边界、危险地点后续等。优点是稳定、可控、适合高频校园场景。

### llm_or_fallback

未命中本地策略时，进入 LLM 生成或兜底支持计划。回复仍会经过风险、熵减、策略执行和最终护栏。

### crisis_safety

当识别到自伤、自杀、危险地点或其他高危组合时，直接进入安全优先路径。该路径不继续普通安慰，而是输出现实安全动作、紧急联系和人工转介信号。

## 展示边界

学生端默认展示：

- `reply_text`
- `safety.emergency_notice`
- `safety.human_referral`
- `entropy_reduction.core_actions`
- `campus_resources`

研究端、咨询师端、管理端可展示：

- `processing_summary`
- `risk`
- `entropy`
- `state_profile`
- `intervention_strategy`
- `dynamic_adjustment`
- `referral_decision`

学生端不默认展示 `processing_summary`，避免把后端决策标签暴露给正在求助的学生。

## 会话级处理时间线

`GET /api/v1/sessions/{session_id}/analysis` 现在会返回会话级处理信息：

| 字段 | 含义 |
| --- | --- |
| `latest_processing_summary` | 最近一轮完整处理摘要。 |
| `processing_timeline` | 最近若干轮处理摘要的时间线。 |
| `processing_summary` | 对 timeline 的聚合统计。 |
| `processing_routes` | 各处理路由计数。 |
| `processing_safety_priorities` | 安全优先级计数。 |
| `processing_next_backend_actions` | 下一步后端动作计数。 |

这能用来观察一个 session 是否从普通支持逐步进入人工关注或危机安全路线。例如：

```text
local_policy -> local_policy -> crisis_safety
standard -> human_followup -> urgent
continue_supportive_monitoring -> queue_human_followup -> activate_urgent_handoff
```

该 timeline 复用已存储的 response JSON，不需要新增数据库表；旧数据如果没有 `processing_summary`，会显示为 `legacy_or_missing`。

## 验收方式

运行：

```powershell
python -m pytest tests\test_agent.py tests\test_main.py tests\test_privacy_views.py -q
```

重点检查：

- 普通场景响应包含 `processing_summary`。
- 危机场景 `processing_summary.route = crisis_safety`。
- 危机场景 `safety_priority = urgent`。
- 学生角色视图不暴露 `processing_summary`。
- 研究/管理视图可以查看 `processing_summary`。
- 会话分析返回 `processing_timeline` 和聚合后的 `processing_summary`。
- 同一 session 中“考试失眠 -> 天台冷静”会从 `local_policy` 升到 `crisis_safety / urgent / activate_urgent_handoff`。
