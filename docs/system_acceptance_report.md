# 系统验收报告

- 生成时间：2026-06-01T15:18:39
- 当前分支/提交：codex-backend-reference-alignment@55ab8c1
- 部署自检：ready，pass=8，fail=0
- 自动测试：308 passed

## 当前系统层级

| 层级 | 状态 | 说明 |
| --- | --- | --- |
| 多模态输入层 | 已完成原型 | 文本与语音入口已接入同一 Agent 链路，语音保留基础音频信号。 |
| 心理熵与策略层 | 已完成核心闭环 | 支持风险识别、心理熵评估、熵减策略、动态调整和校园资源匹配。 |
| DOCX 质量评估层 | 已达标 | 对照长对话优秀回复做自动测评，并保留低分样例定位能力。 |
| 前端交接层 | 已完成交付包 | 提供 API contract、CORS 配置、smoke 脚本、TypeScript client 和 React 示例页。 |
| 人工干预层 | 已完成第一版闭环 | 支持 care queue、人工确认、升级、解决和关闭。 |
| 隐私边界层 | 已完成第一版 | 学生、咨询师、研究、管理员四类视图由后端投影。 |
| 部署运维层 | 已完成第一版 | 提供 readiness API 与终端自检脚本。 |
| 演示验收层 | 已完成第一版 | 提供演示手册、验收清单和系统验收报告生成器。 |

## 关键验收指标

| 指标 | 当前结果 | 来源 |
| --- | --- | --- |
| DOCX 后端平均分 | 80.81 | reports\auto_quality_pipeline\backend_docx\20260530_121926_extracted_docx_reference_eval.jsonl |
| DOCX 案例/轮次 | 100 / 299 | reports\auto_quality_pipeline\backend_docx\20260530_121926_extracted_docx_reference_eval.jsonl |
| DOCX 问题标签 | {} | 自动评估 JSONL |
| DOCX 低分样例数 | 0 | 自动评估 JSONL |
| 手动抽检 PASS/WARN | 27 / 0 | reports\manual_reply_checks\20260530_121703_manual_reply_check.json |
| 手动抽检场景/轮次 | 12 / 27 | 手动抽检 JSON |
| 部署 readiness | ready | `scripts/check_deployment_readiness.py` |
| 单元/回归测试 | 308 passed | pytest |
| 前端交接产物 | ready | `frontend_handoff/` |
| 演示验收材料 | ready | `docs/demo_acceptance_*` |

## 主要接口

- `POST /api/v1/support/text`：文本心理支持入口。
- `POST /api/v1/support/audio`：语音心理支持入口。
- `GET /api/v1/frontend/contract`：前端交接契约。
- `GET /api/v1/sessions/{session_id}/view?role=student|counselor|research|admin`：角色视图与隐私边界。
- `GET /api/v1/analytics/care-queue`：人工关注队列。
- `POST /api/v1/sessions/{session_id}/human-interventions`：人工处理记录。
- `GET /api/v1/ops/readiness`：部署自检。

## 前端与演示交付物

| 交付物 | 状态 | 路径 |
| --- | --- | --- |
| TypeScript API client | present | `frontend_handoff\campusSupportApi.ts` |
| React 学生端示例 | present | `frontend_handoff\StudentChatExample.jsx` |
| 演示验收手册 | present | `docs\demo_acceptance_playbook.md` |
| 演示验收清单 | present | `docs\demo_acceptance_checklist.md` |
| 前端 smoke 报告 | present | `reports\frontend_handoff_smoke\20260530_122832_frontend_handoff_smoke.md` |

## 推荐交付说明

- 给前端组员：优先看 `docs/frontend_integration_guide.md`，复制 `frontend_handoff/campusSupportApi.ts` 和 `StudentChatExample.jsx`。
- 给负责模型的组员：默认使用 README 中的 Qwen3 基础模型 + `refinement_pool_v5_peft` LoRA。
- 给答辩/论文材料：使用 DOCX 自动测评、手动抽检、熵轨迹导出和后端对比实验作为实验支撑。
- 给部署同学：启动前先运行 `python scripts\check_deployment_readiness.py`。
- 给演示准备：先看 `docs/demo_acceptance_playbook.md`，再按 `docs/demo_acceptance_checklist.md` 勾选。

## 下一步建议

1. 补正式咨询师工作台页面，把 care queue 和 human-interventions 可视化。
2. 接真实 ASR 服务，并把语音停顿、音量、静音比例纳入多模态展示。
3. 将正式前端仓库接入 `smoke_frontend_handoff.py`，形成前后端合并验收流程。
