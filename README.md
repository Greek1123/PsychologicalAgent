# 校园心理支持 Agent MVP

这是一个适合你当前课题的最小可用后端：围绕“文本/语音输入 -> 风险识别 -> 心理支持方案生成 -> 校园转介提示”搭了一个可扩展的 Agent 骨架。

这版的目标不是直接替代心理咨询，而是先把系统主链路跑通，并把 `EmoLLM` 放在“可替换模型后端”的位置上。这样你后面无论接本地部署的 EmoLLM、LMDeploy 服务，还是先拿别的兼容模型联调，代码都不用大改。

## 当前项目层级快照

更新时间：2026-05-25。

当前项目已经从最初的 MVP 进入到“多模态校园心理 Agent + 心理熵评估 + 熵减策略 + 会话级动态平衡闭环”的系统雏形阶段。文本和语音入口已经接入同一条 Agent 主链路，后端能够完成风险识别、心理熵评分、主要熵源识别、校园资源检索、支持方案生成、转介建议、会话记忆、熵轨迹持久化和后续策略调整。

按层级看，目前完成度如下：

- L1 输入与服务层：已完成。支持 `POST /api/v1/support/text` 文本输入、`POST /api/v1/support/audio` 语音文件输入、`/app` 轻量测试前端、`/health` 和模型状态接口。
- L2 模型与多模态适配层：基本完成。LLM 支持 `mock`、OpenAI 兼容接口和本地 `local_checkpoint`；语音转写支持 `mock` 与 OpenAI 兼容 STT。语音请求现在会同时记录基础音频信号 `multimodal_signal`，WAV 文件可提取时长、采样率、声道数、RMS 能量、峰值和静音比例。当前还不是完整的音频情绪识别，但已经不再只保留转写文本。
- L3 校园心理 Agent 决策层：已完成核心闭环。包含安全分流、状态画像、心理熵评估、熵减策略、干预策略选择、校园知识库资源匹配和最终支持回复生成。
- L4 动态平衡与纵向追踪层：已具备雏形并持续扩展。系统会保存会话历史、熵轨迹、支持回合、转介事件和用户反馈，并基于这些数据生成动态调整、会话连续性、策略重选、趋势预警、照护路径、照护计划、干预效果和熵减结果评估。
- L5 训练与评估层：已建立工具链。支持训练数据导出、反馈坏例导出、DPO/SFT 数据构建、ms-swift 脚本、策略层评估、checkpoint 场景评估和月度报告材料。
- L6 产品化与真实部署层：继续推进。已有测试前端、完整后端 API，并新增 `GET /api/v1/frontend/contract` 前端交接契约接口，用来给前端组员稳定说明文本/语音请求格式、核心响应字段、展示策略、风险徽标和演示问题；后续仍需要补齐正式前端体验、权限/隐私、真实 STT 服务、人工咨询师工作台、告警通知和真实校园流程对接。

当前一句话定位：项目已经不是单纯聊天机器人，而是一个后端能力较完整的“校园心理熵减与动态平衡 Agent 原型系统”；下一阶段应重点补齐产品化前端、真实语音能力、人工干预闭环和实验评估报告。

最新对齐进展：已将 `docs/心理助手长对话模拟测试用例50例.docx` 和 `docs/心理助手长对话模拟测试用例_新增50例_含隐性高危场景.docx` 作为参考回复库接入评估流程。当前后端已针对期末复习崩溃、任务拖延堆积、小组作业边缘化、分手后反复联系、宿舍冷落/社交隔离、冲动伤人风险、亲密关系威胁、论文查重焦虑、被跟踪安全安排、匿名攻击、隐私背叛、家庭职业冲突、父母沟通冲突、代码事故恐慌、科研/答辩/比赛压力、照护压力、关系耗竭、家庭暴力、隐性轻生信号、危险地点、断联失联、送出重要物品、账号交接、运动受伤身份感等场景补充本地策略。2026-05-25 最新自动流水线跑完整 100 例后，后端 mock 链路平均启发式评分提升到 80.93，`flag_counts` 为空，`low_score_examples` 为空；评估报告默认作为本地产物生成到 `reports/`，不随 GitHub 仓库保存。本轮达标主要来自后端最终回复层和安全路由修复；LoRA 训练仍作为自然度和泛化增强手段，隐私、危机、医疗边界继续由后端策略层兜底。

对比实验进展：新增 `scripts/compare_docx_backend_experiment.py`，可同题比较“无策略通用基线”和“当前后端策略层”。2026-05-25 完整 100 例对比结果为：通用基线平均分 67.92，存在 `weak_action_specificity=33`、`misses_crisis_safety=4`、`misses_privacy_reassurance=3`；当前后端策略层平均分 80.93，`flag_counts` 为空，低分样例为空，相对基线提升 13.01 分。该报告适合直接作为论文/答辩中“后端熵减策略层有效性”的实验支撑材料。

动态平衡实验进展：新增 `scripts/export_docx_entropy_trajectories.py`，可把 DOCX 长对话逐轮导出为心理熵轨迹、风险等级、平衡状态、主导熵源、策略序列和动态调整状态。2026-05-25 完整 100 例导出结果为：299 个 turn，平均心理熵 14.77，风险分布为 low=236、medium=54、high=7、critical=2，平衡状态分布为 stable=290、fragile=7、crisis=2；高频策略包括 `supportive_listening`、`grounding_small_step`、`dorm_boundary_support`、`sleep_stabilization`、`safety_first`。该报告适合支撑“熵减与动态平衡”的多轮案例分析。

手动抽检进展：新增 `scripts/run_manual_reply_check.py`，用于固定运行代表性后端对话，并记录“输入问题、系统回复、风险等级、心理熵、策略、动态状态”。2026-05-27 已扩展到 12 个场景 / 27 轮，并加入 expected/forbidden 关键词自动检查，最新记录输出到 `reports/manual_reply_checks/20260527_191548_manual_reply_check.md`、`.csv`、`.json`，全部 PASS。抽检中继续修复了暗恋/表白后续轮次被好友疏远或隐私模板抢走的问题；修复后全量测试 269 passed，DOCX 后端 100 例平均分提升到 81.04，`flag_counts` 为空，`low_score_examples` 为空。
前端交付层进展：2026-05-27 新增 `GET /api/v1/frontend/contract`，前端组员可以先请求这个接口确认当前推荐接入方式。学生端优先展示 `reply_text`、必要安全提示和可选的 `entropy_reduction.core_actions`；研究/管理面板再展示 `risk`、`entropy`、`state_profile`、`intervention_strategy`、`dynamic_adjustment`、`referral_decision`、`multimodal_signal` 和 `system_flags`。
人工干预闭环进展：2026-05-27 在已有 `GET /api/v1/analytics/care-queue` 的基础上新增人工处理记录接口 `POST /api/v1/sessions/{session_id}/human-interventions` 和 `GET /api/v1/sessions/{session_id}/human-interventions`。咨询师/辅导员端现在可以把队列项标记为 `acknowledged`、`in_progress`、`escalated`、`resolved` 或 `closed`；默认 care queue 会隐藏已 `resolved/closed` 的会话，需要审计时加 `include_resolved=true`。
隐私边界进展：2026-05-27 新增 `GET /api/v1/sessions/{session_id}/view?role=student|counselor|research|admin`，由后端直接生成不同角色视图。学生端只拿回复、安全提示、轻量风险标签、平衡状态和用户可见行动；研究端保留结构化风险/熵/策略指标但隐藏自由文本和人工备注；管理员视图保留完整内部字段用于本地审计。
部署自检进展：2026-05-27 新增 `GET /api/v1/ops/readiness` 和 `scripts/check_deployment_readiness.py`，用于检查 provider 配置、数据库目录、日志目录、校园知识库、local checkpoint、基础模型和 Python 版本。`scripts/run_local_checkpoint_api.ps1` 启动前会先运行自检，避免模型路径或数据目录错误时服务半启动。
验收报告进展：2026-05-28 新增 `scripts/generate_acceptance_report.py`，可自动汇总部署自检、DOCX 后端测评、手动抽检、核心接口和系统层级，生成 `docs/system_acceptance_report.md`，方便组会、答辩或交给组员查看当前项目完成度。
演示材料进展：2026-05-28 新增 `scripts/generate_demo_workspace.py`，可自动运行 6 个代表性校园心理场景并生成 `docs/demo_workspace_report.md`，覆盖期末焦虑、宿舍边界、暗恋不确定、隐私威胁、被尾随安全安排和危险地点危机优先。

## 协作与进展记录

后续每次由 Codex 继续开发或检查时，同步更新两类记录：

- `README.md`：只记录当前能力层级、运行方式、重要接口和阶段性结论。
- `docs/project_progress_log.md`：按日期追加“本次做了什么、验证结果、遗留问题、下一步建议”。

运行时日志仍保存在 `logs/app.log`，用于查看接口请求、风险判断、熵评估、资源检索、存储和异常信息。

## 仓库整理与上传边界

当前 GitHub 仓库只保留能复现实验和后端能力的核心文件：`src/`、`tests/`、`scripts/`、`docs/` 下的正式说明和参考 Word 文档、`data/campus_knowledge.json`、`training/` 下的训练脚本、`.env.example`、`requirements.txt` 和 `README.md`。

以下内容已经作为本地/可再生成产物处理，不随仓库提交：`logs/`、`reports/`、`tmp*/`、`.pytest_cache/`、测试数据库 `test_*.db`、原始公开语料压缩包、`data/training/` 导出的训练集、`docs/model_evaluations/`、月度报告草稿和 Word 渲染缓存。需要重新评估或导出训练数据时，按 README 中对应脚本重新生成。

本地运行配置和模型权重不上传 GitHub：`.env`、`.venv/`、`.idea/` 和 `training/ms_swift/outputs/` 保留在本机。给组员传递模型时，优先传 `training/ms_swift/outputs/` 里的 LoRA checkpoint 目录，以及它依赖的基础模型路径说明；不要把这些大文件提交到 GitHub。

2026-06-03 处理层优化进展：新增 `followup_reply_overrides` 场景推进层，接入 `final_reply_guardrails`，用于修复短 follow-up 轮次被首轮模板、错误场景模板或泛化安抚模板覆盖的问题。该层不按用例编号或标准答案检索，而是根据当前用户句子 + 历史上下文识别可迁移的心理任务，例如室友作息边界、网暴后停止刷新与恢复表达、班级活动孤立、交通惊吓后的渐进恢复、匿名投稿/隐私泄露保护、毕业去向最小验证任务、恋爱查岗安全感、经济自责、普通感/自我价值、主动社交、项目反馈退缩等。最新完整 100 例 DOCX 后端评估为 `average_score=84.27`，`low_score_examples=[]`；本轮没有新增 LoRA 权重，主要通过后端处理层和回复护栏提升泛化对话质量。

2026-06-05 人工抽检补充：`scripts/generate_random_reply_audit.py` 已继续用于随机抽取 30 条真实回复，并为每条额外追加 2 个源文档没有的问题，生成“我询问的内容 / 模型回复 / 文档优秀回复 / 额外追问”的 Markdown 报告。最新输出位于 `reports/random_reply_audits/20260605_224223_random_reply_audit.md` 和 `.json`；脚本现在默认会实时调用当前后端刷新源文档轮次的模型回复，避免旧 JSONL 输出误导人工检查。本轮据此修复了考试弱输出、低动力/心理中心抗拒、夜里崩溃孤单、班级活动、专业迷茫、宠物离世、暴食小步、欺凌触发和疑似被监控等 follow-up 误路由问题；该报告用于人工检查真实对话质量，不作为分数优化依据。

## 当前推荐模型

给组员复现时，优先说明两类路径：

- 基础模型：`D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507`
- 稳定推荐 LoRA：`D:\psychologicalAgent\training\ms_swift\outputs\refinement_pool_v5_peft\v0-20260520-215838\checkpoint-final`

本轮额外训练的实验 LoRA 位于 `D:\psychologicalAgent\training\ms_swift\outputs\auto_docx_reference_distill_patch\checkpoint-final`。它先从 DOCX 后端评估中抽取 `score <= 64` 的 68 个低分 turn，以文档优秀回复作为 SFT 目标，重复后得到 136 条参考蒸馏样本，并与 302 条安全/还原补丁样本合并为 438 条训练数据；训练从 `auto_docx_safety_patch/checkpoint-final` 继续进行 2 epoch，学习率 `4e-6`。正式交给组员默认复现时仍优先使用稳定推荐 LoRA；如果要试最新实验效果，可以同时传 `auto_docx_reference_distill_patch/checkpoint-final` 做对照。2026-05-25 后续冲刺 80 分阶段没有新增训练权重，主要通过后端策略层把 DOCX 平均分从 71.53 拉到 80.93。

## 自动测评与训练流水线

现在可以用一条命令自动生成训练数据、跑 DOCX 参考回复评估、生成 Markdown/JSONL 对比报告，并汇总低分样例和问题标签：

```powershell
python scripts\auto_quality_pipeline.py --mode backend --limit 20 --start 1
```

如果确认要自动继续训练 LoRA，显式加 `--train` 或使用 `--mode full`：

```powershell
python scripts\auto_quality_pipeline.py --mode full --limit 20 --start 1 --epochs 2 --learning-rate 8e-6
```

默认输出：

- 自动训练数据：`data/training/`，不上传 GitHub。
- 自动评估报告：`reports/auto_quality_pipeline/`，不上传 GitHub。
- 可选训练输出：`training/ms_swift/outputs/auto_docx_safety_patch/`，不上传 GitHub。

这条流水线适合每轮迭代后做“生成回复 -> 对照优秀回复 -> 汇总问题 -> 生成补丁训练集 -> 可选继续训练”的闭环。

如果要生成论文/答辩用的策略层对比实验报告，运行：

```powershell
python scripts\compare_docx_backend_experiment.py --limit 100 --start 1
```

默认输出：

- Markdown 实验报告：`reports/docx_backend_comparison/*_docx_backend_comparison.md`
- JSON 完整结果：`reports/docx_backend_comparison/*_docx_backend_comparison.json`
- turn 级 CSV：`reports/docx_backend_comparison/*_turn_level_comparison.csv`

报告包含总体均分、flags 对比、低分样例、按场景类别的均分，以及 baseline/backend 每轮回复和参考回复，方便后续整理实验表格。

如果要导出长对话心理熵轨迹和动态平衡案例表，运行：

```powershell
python scripts\export_docx_entropy_trajectories.py --limit 100 --start 1
```

默认输出：

- Markdown 轨迹报告：`reports/docx_entropy_trajectories/*_docx_entropy_trajectories.md`
- JSON 完整轨迹：`reports/docx_entropy_trajectories/*_docx_entropy_trajectories.json`
- turn 级 CSV：`reports/docx_entropy_trajectories/*_docx_entropy_trajectories.csv`

报告包含每个案例的熵变化、风险分布、策略序列和逐轮表格，适合进一步制作折线图或答辩案例页。

如果要做一组人工可读的固定回复抽检，运行：

```powershell
python scripts\run_manual_reply_check.py
```

默认输出：

- Markdown 抽检记录：`reports/manual_reply_checks/*_manual_reply_check.md`
- CSV 明细：`reports/manual_reply_checks/*_manual_reply_check.csv`
- JSON 明细：`reports/manual_reply_checks/*_manual_reply_check.json`

这份记录会保存每一轮“我输入的问题”和“系统回复”，适合人工逐条检查回复质量。

如果要从 DOCX 评估结果里随机抽取真实回复，并额外追加源文档没有的模糊/弱输出追问，运行：

```powershell
python scripts\generate_random_reply_audit.py --sample-size 30 --seed 20260603
```

默认输出：

- Markdown 人工抽检报告：`reports/random_reply_audits/*_random_reply_audit.md`
- JSON 明细：`reports/random_reply_audits/*_random_reply_audit.json`

## 目前已经实现

- 文本支持接口：`POST /api/v1/support/text`
- 语音支持接口：`POST /api/v1/support/audio`
- 校园心理场景安全分流：高风险/危机表达时直接进入人工转介模板
- 可替换的 LLM Provider：默认 `mock`，可切到 OpenAI 兼容接口
- 可替换的语音转写 Provider：默认 `mock`，可切到 OpenAI 兼容接口
- 语音基础信号分析：音频请求返回 `multimodal_signal`，WAV 文件可提取基础声学特征
- 自动读取项目根目录 `.env`
- 基础会话记忆：支持同一 `session_id` 下的多轮上下文
- 本地校园知识库检索：根据学生表达自动匹配心理中心、校医院、教务、宿舍调解等资源
- 运行日志：同时输出到控制台与 `logs/app.log`
- 可见心理熵评估：返回 `entropy.score`、`entropy.level`、`entropy.balance_state`
- 可见减熵策略：返回 `entropy_reduction.targeted_drivers`、`core_actions`、`expected_delta_score`
- SQLite 持久化：会话历史和熵轨迹默认保存到本地数据库
- 训练数据导出：支持把完整支持回合导出成 JSONL
- 双语训练模板：支持生成中英双语 style / analysis 数据模板
- 双语风格语料转换：支持把 `cn_data_version7.json` / `en_data_version7.json` 转成 `style_sft`
- 风格数据筛选：支持把 `style_sft` 自动分成 keep / review / drop
- 风格对齐模板：支持生成 `style_dpo` 偏好标注模板
- Word 参考用例对照评估：支持从 100 个长对话参考案例中抽取优秀回复，并生成后端回复对比报告
- 后端参考用例自动评估：`scripts/evaluate_backend_docx_reference_cases.py` 可直接用当前后端 mock 链路逐轮跑 Word 案例
- 后端策略层对比实验：`scripts/compare_docx_backend_experiment.py` 可生成通用基线 vs 当前后端策略层的 Markdown/JSON/CSV 报告
- 长对话动态平衡导出：`scripts/export_docx_entropy_trajectories.py` 可生成心理熵轨迹、风险/平衡状态和策略序列报告
- 手动回复抽检：`scripts/run_manual_reply_check.py` 可固定运行代表性问题并保存逐轮问题/回复记录
- 结构化输出：情绪评估、压力源、保护因子、熵水平、平衡状态、支持计划、安全提示
- 单元测试：覆盖文本低风险、危机分流、语音转写链路

## 项目结构

```text
.
├─ .env.example
├─ requirements.txt
├─ docs/
│  ├─ project_progress_log.md
│  ├─ *.md
│  └─ 心理助手长对话模拟测试用例*.docx
├─ scripts/
│  └─ *.py
├─ src/
│  └─ campus_support_agent/
│     └─ 后端 Agent、API、Provider、策略、存储与训练导出模块
├─ data/
│  └─ campus_knowledge.json
├─ training/
│  └─ ms_swift/
└─ tests/
   └─ test_*.py
```

## 怎么跑

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 复制环境变量

```bash
copy .env.example .env
```

说明：这版现在会自动读取项目根目录的 `.env`，不需要你每次手动在终端里重新设置 `LLM_PROVIDER`。

如果要让后端直接调用本地 Qwen3 + LoRA checkpoint，把 `.env` 改成：

```env
LLM_PROVIDER=local_checkpoint
LOCAL_CHECKPOINT_PATH=D:\psychologicalAgent\training\ms_swift\outputs\public_phase0_sft\v0-20260426-134431\checkpoint-465
LOCAL_BASE_MODEL_PATH=D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507
LOCAL_MODEL_CACHE_ROOT=D:\llm_cache
LOCAL_GENERATION_TEMPERATURE=0
LLM_MAX_TOKENS=512
```

3. 启动服务

```bash
uvicorn campus_support_agent.main:app --app-dir src --reload --port 8000
```

如果使用本地 checkpoint，推荐直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_checkpoint_api.ps1
```

4. 打开接口文档

```text
http://127.0.0.1:8000/docs
```

5. 打开内置测试前端

```text
http://127.0.0.1:8000/app
```

## 日志说明

默认日志配置：

- 日志级别：`INFO`
- 日志文件：`logs/app.log`

你可以在 `.env` 里改：

```env
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/app.log
DATABASE_PATH=data/campus_agent.db
LLM_MAX_TOKENS=512
```

日志会记录这些关键节点：

- 接口请求进入与完成
- 风险等级判断
- 校园知识库命中情况
- LLM / STT 调用
- 模型失败后的 fallback 降级
- 内置前端访问

## 数据持久化说明

系统现在默认使用本地 SQLite 持久化，数据库路径由 `.env` 里的 `DATABASE_PATH` 控制。

默认会保存：

- 会话消息历史
- 每一轮的心理熵快照
- 熵等级、平衡状态、主要驱动

这意味着服务重启后，这些内容仍然保留，适合你做纵向实验和动态平衡分析。

相关代码在：

- [storage.py](d:/psychologicalAgent/src/campus_support_agent/storage.py)
- [main.py](d:/psychologicalAgent/src/campus_support_agent/main.py)

如果你想从某个案例重新开始，可以：

- 在测试前端点击“清空当前会话”
- 或调用 `DELETE /api/v1/sessions/{session_id}`

## 训练数据导出

系统现在会把完整支持回合保存到 SQLite，包含：

- 当前输入文本
- 学生上下文
- 参与推理的历史消息
- 风险判断
- 心理熵
- 减熵策略
- 支持方案
- 校园资源

导出器在：

- [training_export.py](d:/psychologicalAgent/src/campus_support_agent/training_export.py)

你后续可以在项目目录运行：

```bash
python scripts/export_training_data.py --out data/training/train_sft.jsonl --format sft
```

或者导出成更适合研究分析的记录格式：

```bash
python scripts/export_training_data.py --out data/training/train_record.jsonl --format record
```

### 两种格式的区别

- `sft`：适合指令微调，包含 `system/user/assistant` 形式的 `messages`
- `record`：适合研究和二次处理，保留更完整的 `target` 字段

导出结果现在会额外包含：

- `language`
- `task_type`
- `stage_goal`

这样更适合你做“先训练说话习惯，再训练分析层”的双路训练。

### 推荐训练顺序

1. 先持续采集和清洗你的校园心理支持样本
2. 导出 `record` 格式，检查熵标签和减熵策略是否合理
3. 导出 `sft` 格式，用于 LoRA / QLoRA 微调
4. 后续再基于人工偏好做 DPO / ORPO

配套的熵标签说明见：

- [entropy_labeling_guide.md](d:/psychologicalAgent/docs/entropy_labeling_guide.md)
- [bilingual_training_workflow.md](d:/psychologicalAgent/docs/bilingual_training_workflow.md)
- [style_data_selection_guide.md](d:/psychologicalAgent/docs/style_data_selection_guide.md)
- [style_quality_review_guide.md](d:/psychologicalAgent/docs/style_quality_review_guide.md)
- [human_eval_guide.md](d:/psychologicalAgent/docs/human_eval_guide.md)

## 双语训练模板

如果你想先搭建中英双语训练集结构，可以直接生成模板：

```bash
python scripts/generate_bilingual_templates.py
```

默认会生成到：

- `data/training_templates/style_sft_template.jsonl`
- `data/training_templates/style_dpo_template.jsonl`
- `data/training_templates/analysis_sft_template.jsonl`

相关代码在：

- [dataset_templates.py](d:/psychologicalAgent/src/campus_support_agent/dataset_templates.py)
- [generate_bilingual_templates.py](d:/psychologicalAgent/scripts/generate_bilingual_templates.py)

### 三种模板分别做什么

- `style_sft_template.jsonl`
  用来训练“很会聊、自然、共情、会追问”的支持风格
- `style_dpo_template.jsonl`
  用来做风格偏好对齐，让模型学会更自然、更安全的表达
- `analysis_sft_template.jsonl`
  用来训练第二层分析能力，输出 `risk + entropy + entropy_reduction + plan`

### 双语建议

- 字段名统一用英文
- 中文和英文共用任务结构
- 风格层和分析层分开训练
- 不要只用直译英文，最好做人工润色

## 原始双语对话转 style_sft

如果你要把这两个原始文件：

- [cn_data_version7.json](d:/psychologicalAgent/data/cn_data_version7.json)
- [en_data_version7.json](d:/psychologicalAgent/data/en_data_version7.json)

转换成适合当前目标的双语风格训练集，可以运行：

```bash
python scripts/build_style_sft_dataset.py
```

默认输出到：

- `data/processed/style_sft_bilingual.jsonl`

相关代码在：

- [style_dataset_builder.py](d:/psychologicalAgent/src/campus_support_agent/style_dataset_builder.py)
- [build_style_sft_dataset.py](d:/psychologicalAgent/scripts/build_style_sft_dataset.py)

### 转换时会保留什么

- `dialog` 转成标准 `messages`
- `language`
- `task_type`
- `stage_goal`
- `topic/theme/summary` 作为 `meta`

### 转换时会过滤什么

- `reasoning`
- `guide`
- 不够长的短对话
- 过强的内部分析说明

这样做的目的是先把模型训练成“自然支持地聊天”，而不是训练成直接暴露内部推理的治疗师脚本。

## 按风格优先路线的数据流程

如果你想按“先训练说话习惯，再训练分析层”的路线推进，建议顺序是：

1. 原始双语对话转 style_sft

```bash
python scripts/build_style_sft_dataset.py
```

2. 对 style_sft 做质量筛选

```bash
python scripts/triage_style_dataset.py
```

3. 把保留下来的风格数据切成 train/dev/test

```bash
python scripts/split_style_dataset.py
```

4. 把单轮支持数据扩成多轮

```bash
python scripts/expand_single_turn_dataset.py
```

5. 单独生成风格偏好对齐模板

```bash
python scripts/build_style_preference_templates.py
```

6. 导出人工评测表

```bash
python scripts/build_human_eval_sheet.py
```

## style_sft 质量筛选

把双语 `style_sft` 生成出来之后，你可以继续运行：

```bash
python scripts/triage_style_dataset.py
```

默认会读取：

- `data/processed/style_sft_bilingual.jsonl`

并输出到：

- `data/processed/triaged_style/style_keep.jsonl`
- `data/processed/triaged_style/style_review.jsonl`
- `data/processed/triaged_style/style_drop.jsonl`

相关代码在：

- [style_data_filter.py](d:/psychologicalAgent/src/campus_support_agent/style_data_filter.py)
- [triage_style_dataset.py](d:/psychologicalAgent/scripts/triage_style_dataset.py)

### 三类文件怎么理解

- `keep`
  可以优先进入第一版风格训练
- `review`
  质量尚可，但可能有治疗学派痕迹太重、表达过硬、需要人工复核
- `drop`
  对话太短、结构不完整、训练价值较低

## 风格对齐模板

如果你要把风格对齐单独做，而不是只靠原始语料，可以继续运行：

```bash
python scripts/build_style_preference_templates.py
```

默认会读取：

- `data/processed/triaged_style/style_review.jsonl`

并输出到：

- `data/processed/style_preference/style_dpo_template.jsonl`

相关代码在：

- [preference_template_builder.py](d:/psychologicalAgent/src/campus_support_agent/preference_template_builder.py)
- [build_style_preference_templates.py](d:/psychologicalAgent/scripts/build_style_preference_templates.py)

这份模板会保留：

- `prompt`
- `chosen`
- 空的 `rejected`
- `review_notes`

你后续可以让人工补齐 `rejected`，再进入 DPO / ORPO 训练。

## 接口示例

### 前端交接契约

前端组员优先看这个接口，它返回当前后端推荐的请求格式、核心响应字段、展示策略、风险徽标和演示问题：

```powershell
curl "http://127.0.0.1:8000/api/v1/frontend/contract"
```

### 部署自检

不开服务时，可以直接在终端检查当前配置：

```powershell
python scripts\check_deployment_readiness.py
```

服务启动后，也可以调接口：

```powershell
curl "http://127.0.0.1:8000/api/v1/ops/readiness"
```

自检会返回 `ready`、`degraded` 或 `blocked`。如果是 `blocked`，优先检查 `.env` 里的 `LOCAL_CHECKPOINT_PATH`、`LOCAL_BASE_MODEL_PATH`、`DATABASE_PATH`、`LOG_FILE_PATH` 和 `CAMPUS_KB_PATH`。

### 生成验收报告

每轮较大改动后，可以生成一份当前系统验收摘要：

```powershell
python scripts\generate_acceptance_report.py --test-summary "278 passed"
```

默认输出：

```text
docs/system_acceptance_report.md
```

这份报告会汇总系统层级、DOCX 后端平均分、手动抽检 PASS/WARN、部署 readiness、核心接口和下一步建议。

### 生成演示工作台样例

如果觉得材料太少，建议先生成演示工作台报告：

```powershell
python scripts\generate_demo_workspace.py
```

默认输出：

```text
docs/demo_workspace_report.md
```

这份报告会实际调用当前后端，生成 6 个代表性校园场景的学生回复摘录、风险等级、心理熵、策略、转介状态和人工处理状态。

推荐接入顺序：

1. 学生对话页先接 `POST /api/v1/support/text`，主气泡只展示 `reply_text`。
2. 语音页再接 `POST /api/v1/support/audio`，如果返回 `multimodal_signal`，可以在调试面板展示音频证据。
3. 研究/管理面板展示 `risk`、`entropy`、`state_profile`、`intervention_strategy`、`dynamic_adjustment`、`referral_decision` 和 `system_flags`。
4. 普通学生端默认隐藏 `hidden_clinical_goal`、`backend_reason`、`backend_actions` 和 `system_flags.reasons`。

### 人工干预队列

查看当前需要人工关注的会话：

```powershell
curl "http://127.0.0.1:8000/api/v1/analytics/care-queue"
```

标记人工处理状态：

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/sessions/demo-student-001/human-interventions" ^
  -H "Content-Type: application/json" ^
  -d "{\"response_id\":\"support_xxx\",\"status\":\"acknowledged\",\"handler_id\":\"counselor-001\",\"note\":\"已查看高优先级队列，准备线下跟进。\",\"next_action\":\"same_day_checkin\",\"tags\":[\"manual_followup\"]}"
```

查看处理记录：

```powershell
curl "http://127.0.0.1:8000/api/v1/sessions/demo-student-001/human-interventions"
```

### 角色视图与隐私边界

学生端、咨询师端、研究端和管理员端不要直接共用完整后端 JSON。建议改用角色视图接口：

```powershell
curl "http://127.0.0.1:8000/api/v1/sessions/demo-student-001/view?role=student"
curl "http://127.0.0.1:8000/api/v1/sessions/demo-student-001/view?role=counselor"
curl "http://127.0.0.1:8000/api/v1/sessions/demo-student-001/view?role=research"
curl "http://127.0.0.1:8000/api/v1/sessions/demo-student-001/view?role=admin"
```

默认建议：

- `student`：正式学生端使用。
- `counselor`：咨询师/辅导员工作台使用。
- `research`：论文实验、统计面板和导出分析使用。
- `admin`：仅本地开发、排错和审计使用。

### 文本输入

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/support/text" ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"demo-student-001\",\"text\":\"最近考试很多，我晚上总睡不好，还总担心挂科。\",\"student_context\":{\"grade\":\"大二\",\"major\":\"计算机\"}}"
```

### 自动评测 API 回复质量

先启动后端，再运行：

```bash
python scripts/evaluate_api_quality.py --base-url http://127.0.0.1:8000
```

查看当前后端模型配置：

```bash
curl "http://127.0.0.1:8000/api/v1/model/status"
```

如果只想测 checkpoint 命令行回复质量：

```bash
D:\Anaconda\python.exe scripts/evaluate_chat_quality.py --mode checkpoint --checkpoint "D:\psychologicalAgent\training\ms_swift\outputs\public_phase0_sft\v0-20260426-134431\checkpoint-465" --temperature 0
```

### 语音输入

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/support/audio" ^
  -F "file=@sample.wav" ^
  -F "session_id=demo-student-001" ^
  -F "student_context={\"grade\":\"大一\"}"
```

### 查看某个会话历史

```bash
curl "http://127.0.0.1:8000/api/v1/sessions/demo-student-001"
```

### 单独查看心理熵

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/entropy/evaluate" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"最近考试很多，我晚上睡不好，也很担心挂科。\",\"student_context\":{\"grade\":\"大二\"}}"
```

## 心理熵字段说明

接口返回里的 `entropy` 用来直接展示当前对话的心理熵估计，适合你在测试和研究阶段观察：

- `entropy.score`：0-100，总体心理熵分数
- `entropy.level`：1-5，离散等级
- `entropy.balance_state`：`stable | strained | fragile | crisis`
- `entropy.dominant_drivers`：当前主要熵来源
- `entropy.dimensions`：六个维度的子分数
- `entropy.trend`：如果使用同一个 `session_id` 连续对话，会显示和上一轮相比的变化

接口返回里的 `entropy_reduction` 用来直接展示当前回合的减熵设计：

- `entropy_reduction.target_state`：目标平衡状态
- `entropy_reduction.targeted_drivers`：本轮重点处理的高熵驱动
- `entropy_reduction.rationale`：为什么优先处理这些驱动
- `entropy_reduction.core_actions`：优先级最高的减熵动作
- `entropy_reduction.expected_delta_score`：预估熵值变化，负数表示期望下降
- `entropy_reduction.review_window_hours`：建议多久后复盘一次

如果未来不想让普通用户看到这些字段，可以只在前端隐藏，后端仍然保留它们用于研究、日志和随访分析。

## 内置前端说明

项目现在自带一个轻量测试前端，文件在：

- [app.html](d:/psychologicalAgent/src/campus_support_agent/static/app.html)
- [app.css](d:/psychologicalAgent/src/campus_support_agent/static/app.css)
- [app.js](d:/psychologicalAgent/src/campus_support_agent/static/app.js)

它支持：

- 文本测试
- 语音文件上传测试
- 会话历史查看
- 心理熵、减熵策略、校园资源可视化
- 通过页面右上角开关隐藏熵相关区域

这比较适合研究演示和联调。未来如果你要给学生正式使用，可以保留后端字段，但在前端隐藏 `entropy` 与 `entropy_reduction`。

## 校园知识库怎么改

本地知识库文件在 [data/campus_knowledge.json](d:/psychologicalAgent/data/campus_knowledge.json)。

你后面最值得替换掉的是这里面的示例内容，把它改成你自己学校的真实信息，比如：

- 心理中心预约方式
- 值班电话和工作时间
- 校医院门诊入口
- 请假、缓考、学业预警流程
- 辅导员、班主任协同干预流程
- 宿舍冲突调解流程
- 校园危机处置联系人

改完后重启服务，Agent 就会优先引用这些校园资源。

## 如何接入 EmoLLM

对你当前这个课题，`EmoLLM` 仓库里最值得直接复用的是“模型服务能力”，不是整个仓库全搬过来。

### 现在建议直接用的部分

- 已训练好的 EmoLLM 模型权重
- `deploy` 相关能力
- README 里提到的 `LMDeploy` 量化部署路径
- 后续可选的 `rag` 目录思路

### 现在先不要接进来的部分

- `datasets`
- `generate_data`
- `xtuner_config`
- `swift`
- `evaluate`
- Demo 页面脚本和原型界面

原因很简单：你现在最需要的是“可用后端 Agent”，不是重新训练一个心理模型平台。

### 推荐接法

1. 先把本项目用 `mock` 跑通。
2. 把 EmoLLM 用 LMDeploy 或其他 OpenAI 兼容服务方式部署起来。
3. 把 `.env` 里的这几项改掉：

```env
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://你的-emollm-服务/v1
LLM_MODEL=你的-emollm-模型名
LLM_API_KEY=如果服务需要就填
```

4. 如果你有单独语音识别服务，再配置：

```env
STT_PROVIDER=openai_compatible
STT_BASE_URL=http://你的-ASR-服务/v1
STT_MODEL=whisper-1
```

## 适合你课题的下一步

这一版更像“心理支持中台”。如果你要继续贴近“校园心理熵减与动态平衡系统”，下一步最值得加的是：

- 校园知识库 RAG：校医院、心理中心、请假流程、危机干预流程、宿舍冲突处理流程
- 学生状态记忆：睡眠、作息、考试周、社交事件、家庭压力
- 多轮跟踪任务：7 天睡眠计划、复盘打卡、辅导员转介闭环
- 风险升级策略：班主任/辅导员/心理中心的分层触发

## 验证

```bash
python -m unittest discover -s tests
```

## Style-First Training Pack

如果你现在按“先训练说话风格，再做分析层”的路线推进，建议在完成下面这些脚本之后：

- `python scripts/build_style_sft_dataset.py`
- `python scripts/triage_style_dataset.py`
- `python scripts/split_style_dataset.py`
- `python scripts/expand_single_turn_dataset.py`
- `python scripts/build_style_preference_templates.py`

再执行：

```bash
python scripts/build_style_training_pack.py
```

它会生成这一阶段最关键的 5 个文件：

- `data/training/style_first_pack/style_phase1_train.jsonl`
- `data/training/style_first_pack/style_phase1_dev.jsonl`
- `data/training/style_first_pack/style_phase1_test.jsonl`
- `data/training/style_first_pack/style_phase2_preference.jsonl`
- `data/training/style_first_pack/style_training_manifest.json`

这套训练包的目标是：

- Phase 1: 先用真实多轮对话为主、单轮扩写为辅来做 `style_sft`
- Phase 2: 再单独做 `style_dpo / style_orpo`
- Phase 3: 最后再接入分析层和人工评测

详细说明见：

- [style_first_training_pack.md](d:/psychologicalAgent/docs/style_first_training_pack.md)

## General Multi-turn Warmup

If the current model still feels too rigid, repetitive, or unable to continue normal dialogue, add a general multi-turn warmup stage before style-support SFT.

Build the warmup dataset from `data/training/dialog_release.json`:

```bash
python scripts/build_general_multiturn_dataset.py
```

Default output:

- `data/training/general_multiturn/general_phase0_train_ms_swift.jsonl`

Then regenerate the ms-swift recipes. If the warmup dataset exists, the repo will also generate:

- `training/ms_swift/run_general_phase0_sft.ps1`
- `training/ms_swift/run_general_phase0_sft.sh`

Reference:

- [general_multiturn_warmup.md](d:/psychologicalAgent/docs/general_multiturn_warmup.md)

## Style DPO Annotation

After phase-1 SFT, the next priority is reducing canned responses and repetitive support phrasing.

Build a team annotation sheet with:

```bash
python scripts/build_style_dpo_annotation_sheet.py
```

The generated CSV lives at:

- `data/processed/style_preference/style_dpo_annotation_sheet.csv`

Use it to fill `rejected` responses for phase-2 DPO/ORPO style alignment. The sheet includes:

- `annotation_goal`
- `failure_modes`
- `chosen`
- `candidate_rejected`
- `rejected`
- `annotator_notes`

Reference:

- [style_dpo_annotation_guide.md](d:/psychologicalAgent/docs/style_dpo_annotation_guide.md)

When the team finishes editing the CSV, merge the annotations back into JSONL with:

```bash
python scripts/apply_style_dpo_annotations.py
```

## ms-swift 训练准备

如果你准备把这套风格优先训练包接到 `ms-swift`，建议按这个顺序继续：

先确保当前 Python 环境里已经装好了 `ms-swift`，否则会出现 `swift` 命令找不到。

```bash
python scripts/build_ms_swift_style_datasets.py
python scripts/generate_ms_swift_recipes.py
```

如果你安装的 `ms-swift` 版本不接受 `--torch_dtype auto`，可以改为显式生成：

```bash
python scripts/generate_ms_swift_recipes.py --torch-dtype float16
```

如果你是在本机 RTX 4060 8GB 这类环境上先做第一版实验，更建议直接生成轻量档位：

```bash
python scripts/generate_ms_swift_recipes.py --profile local_8gb
```

这样会额外生成两类文件：

- `data/training/ms_swift/*.jsonl`
  这部分是 `ms-swift` 可直接读取的标准训练数据
- `training/ms_swift/run_style_phase1_sft.ps1`
- `training/ms_swift/run_style_phase2_dpo.ps1`

详细说明见：

- [ms_swift_style_training.md](d:/psychologicalAgent/docs/ms_swift_style_training.md)
- [ms_swift_installation_notes.md](d:/psychologicalAgent/docs/ms_swift_installation_notes.md)

## 参考来源

- EmoLLM 仓库主页：https://github.com/SmartFlowAI/EmoLLM
- EmoLLM README 中明确包含 `部署指南`、`RAG`、`评测指南` 等模块：https://github.com/SmartFlowAI/EmoLLM#readme
- EmoLLM README 的免责声明强调其仅提供情绪支持与建议，不能替代专业心理咨询：https://github.com/SmartFlowAI/EmoLLM#readme

## Intervention Feedback API

The backend stores whether a support reply was actually helpful. This closes the project loop:
strategy generation -> intervention feedback -> dynamic tracking.

Submit feedback for one model reply:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/sessions/demo-student-001/feedback" ^
  -H "Content-Type: application/json" ^
  -d "{\"response_id\":\"support_xxx\",\"helpful_score\":2,\"mood_after\":70,\"user_note\":\"helpful reply\",\"tags\":[\"helpful\",\"clear\"]}"
```

Read feedback for one session:

```bash
curl "http://127.0.0.1:8000/api/v1/sessions/demo-student-001/feedback"
```

Feedback fields:

- `helpful_score`: `-2` to `2`, where negative means not helpful and positive means helpful.
- `mood_after`: optional `0` to `100`, used to track whether the student feels more stable after the reply.
- `tags`: optional labels such as `helpful`, `too_short`, `pushy`, `clear`.

The feedback summary is also included in:

- `GET /api/v1/sessions/{session_id}/analysis`
- `GET /api/v1/analytics/overview`

Export negatively rated replies for review:

```bash
python scripts/export_training_data.py ^
  --format bad_case ^
  --out data/training/feedback_bad_cases/bad_cases.jsonl
```

By default this exports feedback with `helpful_score <= -1`. Each JSONL row contains the user input,
assistant reply, risk/entropy metadata, feedback tags, and an empty `chosen` field for human rewrite.

If there is no feedback yet, export existing replies for manual screening first:

```bash
python scripts/export_training_data.py ^
  --format review_case ^
  --limit 50 ^
  --out data/training/feedback_bad_cases/review_cases.jsonl
```

Review `review_cases.jsonl`, keep the bad replies, and fill `sft_draft.chosen` with a better answer.

For easier review, convert `review_cases.jsonl` into a CSV sheet:

```bash
python scripts/build_feedback_review_sheet.py build ^
  --input data/training/feedback_bad_cases/review_cases.jsonl ^
  --out data/training/feedback_bad_cases/review_sheet.csv
```

Edit `review_sheet.csv`:

- Fill `mark_bad` with `1` for bad replies.
- Fill `problem_tags` with labels such as `privacy_missed,too_short`.
- Fill `chosen` with the better answer.
- Fill `review_note` if you want to record why the reply was bad.

Apply the edited CSV back to JSONL:

```bash
python scripts/build_feedback_review_sheet.py apply ^
  --input data/training/feedback_bad_cases/review_cases.jsonl ^
  --sheet data/training/feedback_bad_cases/review_sheet.csv ^
  --out data/training/feedback_bad_cases/review_cases_reviewed.jsonl
```

After reviewers fill either `sft_draft.chosen` or `failure_review.preferred_reply`, convert the reviewed
bad cases into DPO-ready files:

```bash
python scripts/build_feedback_dpo_dataset.py ^
  --input data/training/feedback_bad_cases/review_cases_reviewed.jsonl ^
  --out data/training/feedback_bad_cases/feedback_preference.jsonl ^
  --ms-swift-out data/training/feedback_bad_cases/feedback_dpo_ms_swift.jsonl
```

Rows without a rewritten `chosen` answer are skipped and counted as `pending_chosen`.

When `feedback_dpo_ms_swift.jsonl` is ready and not empty, run feedback-driven DPO:

```powershell
powershell -ExecutionPolicy Bypass -File .\training\ms_swift\run_feedback_phase2_dpo.ps1
```

If you want to train on a newer SFT adapter, set it first:

```powershell
$env:FEEDBACK_BASE_ADAPTER="D:\psychologicalAgent\training\ms_swift\outputs\your_sft_run\checkpoint-xxx"
powershell -ExecutionPolicy Bypass -File .\training\ms_swift\run_feedback_phase2_dpo.ps1
```
## 2026-05-30 前端联调交接层

本轮进入接口联调层，新增 `docs/frontend_integration_guide.md`，把前端组员需要的启动方式、文本/语音接口、学生端展示字段、角色视图、人工队列和模型边界整理成独立交接文档。新增 `scripts/smoke_frontend_handoff.py`，后端启动后可一键检查 `/health`、`/api/v1/frontend/contract`、`/api/v1/ops/readiness`、文本对话、学生角色视图和 care queue，并在 `reports/frontend_handoff_smoke/` 生成联调报告。随后补齐正式前端独立端口联调所需的 `FRONTEND_ALLOWED_ORIGINS` 配置，contract 会返回允许的 CORS origins，readiness 会在生产环境使用通配符时提示降级。本轮验证：`python -m pytest tests\test_main.py tests\test_deployment_readiness.py -q` 通过 13 项；`python -m pytest` 完整套件通过 303 项；`python -m py_compile scripts\smoke_frontend_handoff.py` 通过；临时端口 `http://127.0.0.1:8765` smoke 结果 `ok=true`，报告为 `reports/frontend_handoff_smoke/20260530_122832_frontend_handoff_smoke.md`。

继续补充前端交接产物：新增 `frontend_handoff/campusSupportApi.ts`，前端组可直接复制到 Vite/React 项目中使用，已封装文本、语音、角色视图、care queue、人工处理记录和学生端展示模型转换。验证：`python -m pytest tests\test_frontend_handoff_artifacts.py tests\test_main.py tests\test_deployment_readiness.py -q` 通过 15 项；`npx.cmd --yes -p typescript tsc --noEmit --lib DOM,ES2020 --target ES2020 frontend_handoff\campusSupportApi.ts` 通过。

继续补充 React 接入样例：新增 `frontend_handoff/StudentChatExample.jsx` 和 `frontend_handoff/README.md`，展示学生端如何调用 API client、维护 loading/error 状态、展示安全提示、人工支持和熵减行动，同时避免把 `system_flags` 等后端调试字段直接暴露给学生端。验证：`python -m pytest tests\test_frontend_handoff_artifacts.py -q` 通过 3 项；`npx.cmd --yes -p typescript tsc --allowJs --checkJs false --noEmit --jsx react-jsx --lib DOM,ES2020 --target ES2020 frontend_handoff\StudentChatExample.jsx` 通过。

## 2026-06-01 演示与验收层

新增 `docs/demo_acceptance_playbook.md`，把组会/答辩演示流程整理成可照着走的手册，覆盖启动、前端工作台、普通压力、宿舍边界、隐私威胁、危险地点危机、care queue、角色视图和字段展示边界。新增 `scripts/generate_demo_acceptance_checklist.py`，可生成 `docs/demo_acceptance_checklist.md` 作为答辩前人工勾选清单。验证：`python scripts\generate_demo_acceptance_checklist.py` 成功生成清单；`python -m pytest tests\test_demo_acceptance_checklist.py -q` 通过 2 项；`python -m pytest` 完整套件通过 308 项。

继续升级 `scripts/generate_acceptance_report.py` 和 `docs/system_acceptance_report.md`：系统验收报告现在会自动汇总前端交接产物、React 示例、演示手册、演示清单和最新 frontend smoke 报告，方便答辩时只打开一份总报告说明当前完成度。验证：`python -m pytest tests\test_acceptance_report.py -q` 通过 4 项；`python scripts\generate_acceptance_report.py --test-summary "308 passed"` 成功更新报告。

## 2026-06-01 后端处理层摘要

暂停前端继续开发，转回后端处理层。本轮新增 `processing_summary` 响应字段，汇总每一轮输入经过的处理路由、输入模式、回复来源、安全优先级、风险等级、心理熵、状态画像、干预策略、动态调整、熵减编排、转介紧急程度、已完成阶段和下一步后端动作。学生端角色视图不默认暴露该字段，研究/管理/调试视图可以使用它证明处理链路已经闭环。新增 `docs/processing_layer.md` 说明处理层结构。验证：`python -m pytest` 完整套件通过 309 项；临时端口 smoke 确认 `POST /api/v1/support/text` 返回 `processing_summary.route=local_policy` 和 `safety_priority=standard`。

继续优化处理层沉淀：`GET /api/v1/sessions/{session_id}/analysis` 现在会返回 `latest_processing_summary`、`processing_timeline`、聚合 `processing_summary`、处理路由计数、安全优先级计数和下一步后端动作计数。这样可以观察一个会话多轮中是否从普通支持进入人工关注或危机安全路线。新增 API 级回归用例覆盖“考试失眠 -> 天台冷静”同一 session，确认第二轮会升到 `crisis_safety / urgent / activate_urgent_handoff`。验证：`python -m pytest -q` 全量通过 311 项；TestClient smoke 显示 timeline 为 `local_policy -> crisis_safety`。

继续强化处理层自检：新增 `processing_consistency`，自动审计风险等级、处理路由、安全优先级、回复来源和下一步后端动作是否一致。例如 `critical` 却没有进入 `crisis_safety`、`urgent` 却没有触发 `activate_urgent_handoff`、已标记转介但仍只做普通监测，都会被标记为 `needs_review`。该字段面向研究/管理/调试视图，不直接给学生展示。验证：处理层相关测试 39 项通过；TestClient smoke 中“考试失眠 -> 天台冷静”返回 `processing_consistency.summary.status = ok`。

继续扩展全局观测：`GET /api/v1/analytics/overview` 新增 `processing_consistency_summary`、`current_processing_consistency_summary` 和 `processing_consistency_bad_cases`，用于从全局层面发现处理链矛盾。这样不需要逐个 session 打开，也能看到最近是否存在“高危风险被普通路线处理”的记录。验证：overview 聚合相关测试 3 项通过；TestClient smoke 返回整体和当前一致性均为 `ok`，bad cases 为空。

继续补齐处理层验收出口：新增 `GET /api/v1/analytics/processing-health`，汇总部署 readiness、overview 处理一致性、回复质量和决策轨迹，返回 `ok/watch/blocked/no_data`、阻塞问题、关注项和下一步建议。这个接口用于联调、演示前自检和研究/管理面板，不作为学生端展示内容。TestClient smoke 中“考试失眠 -> 天台冷静”返回 `status=watch`、处理一致性 `ok`、阻塞项为空，watch 原因是危机场景触发了决策轨迹关注。

进入验收与演示层：新增 `scripts/smoke_processing_acceptance.py`，用于对运行中的后端执行多案例验收 smoke。脚本会检查 `/health`、`/api/v1/frontend/contract`、`/api/v1/ops/readiness`，再跑考试失眠、宿舍边界、危险地点升级三个代表性 session，并校验 `processing_summary`、`processing_consistency`、`processing-health` 和 overview bad cases。报告输出到 `reports/processing_acceptance/`，作为本地验收产物不随仓库提交。2026-06-03 live smoke 输出 `ok=true`，`processing_health_status=watch`，watch 原因来自危机场景的决策关注，处理一致性无阻塞。

## 2026-05-30 DOCX 低分 turn 定向优化

本轮继续跑 `auto_quality_pipeline.py --mode backend --limit 100 --start 1` 和手动抽检，针对 DOCX 低分 turn 做了两类细化：作业堆积场景中“打开文档就想逃/怕写得很烂”会直接给出实验报告可提交骨架、三到五句话、补图改语病和 25 分钟计时；兼职工资拖欠场景中“怕被拉黑/觉得自己太弱”会转为证据化沟通、维护边界和劳动报酬权益。最新 DOCX 100 例结果：平均分 `80.81`，`flag_counts={}`，低分样例为空；手动抽检 12 场景 / 27 轮 `WARN=0`。

## 2026-05-30 手动抽检 WARN 清零优化

本轮继续优化多轮高危场景路由。`scripts/run_manual_reply_check.py` 在 12 个场景 / 27 轮抽检中暴露 3 个 WARN：账号交接隐性高危的后续解释被“危险地点”模板误抢走；危险地点场景中“别把事情想严重”和“我在楼梯口，还没上去”没有输出足够具体的离开天台路线。已修复最终回复护栏和危机上下文继承逻辑，最新手动抽检 27 轮结果为 `WARN=0`。

## 2026-05-28 危机多轮重复回复修复

本轮继续修复天台场景后的多轮对话问题：当用户反复说“我想去天台冷静一下”或继续追问“我该怎么办”“？”时，系统不再重复同一段“羞耻/疼痛冷静”模板，而是按阶段给出更短、更具体的下一步安全动作。当前策略为：首次提示不要去天台/高处并联系现实支持；重复表达时切换为 30 秒步骤；追问时要求先离开危险路线、去有人处、给可信任的人发送求助句，并让用户只回复“发了”。

同时，未解除的天台/高处危机场景下，如果用户后续只说“我该怎么办”“？”或“很烦躁”，后端会继承上一轮高危上下文，继续保持 `critical` 和 `urgent`，避免后台 care queue 因短输入降级。新增回归测试覆盖重复天台表达、后续问号追问和危机上下文继承。

## 2026-05-28 天台冷静场景安全修复

本轮发现粗略前端测试时，输入“我好难受，我想去天台冷静一下”会被普通考试/睡眠模板覆盖，回复没有优先处理危险地点安全。已修复后端风险识别顺序：当文本同时包含天台、楼顶、高处、桥边、窗边、河边等危险地点，以及难受、冷静一下、一个人、不想回去、撑不住等当前痛苦或隔离信号时，直接升为 `critical`，进入 urgent referral 和 safety-first 回复路径。

已新增回归测试 `test_rooftop_cooling_off_routes_to_crisis_response`，并在 `/app` 快捷问题中加入“天台冷静”按钮，方便复测。

## 2026-05-28 粗略前端测试工作台

本轮已把内置 `/app` 从简单测试页整理成一个可直接联调的三栏工作台，方便在前端组正式页面完成前先测试后端能力：

- 学生对话区：支持文本输入、语音文件上传、快捷测试问题、会话历史加载和清空。
- 系统状态区：展示风险等级、心理熵、动态平衡状态、熵减策略、转介建议、校园资源和熵轨迹。
- 后台面板区：展示角色视图、care queue、部署 readiness、frontend contract 和最近一次完整 JSON。

本地启动后访问：

```powershell
$env:LLM_PROVIDER='mock'
$env:STT_PROVIDER='mock'
uvicorn campus_support_agent.main:app --app-dir src --host 127.0.0.1 --port 8000
```

```text
http://127.0.0.1:8000/app
```

本轮验证结果：`python -m pytest tests/test_main.py` 通过 9 项；`node --check src/campus_support_agent/static/app.js` 通过；`/app` 可正常返回页面；文本接口 smoke test 成功；`/api/v1/ops/readiness` 返回 `ready`。截图级浏览器检查暂未执行，因为当前 Codex 打包环境缺少 `playwright-core`，已用 HTTP/API 检查替代。

## 2026-06-05 后端接口安全成熟化

本轮停止生成新的随机报告，转向项目成熟度建设：管理/研究类接口现在支持轻量 API Key 保护。本地开发时如果不配置 `ADMIN_API_KEY`，会话分析、角色视图、人工干预和 analytics 接口仍可直接调试；一旦配置 `ADMIN_API_KEY`，这些接口需要带 `X-Admin-API-Key: <key>` 或 `Authorization: Bearer <key>`。

生产环境 `APP_ENV=production` 下必须配置 `ADMIN_API_KEY`，否则 `/api/v1/ops/readiness` 会返回 blocked。前端组可以通过 `/api/v1/frontend/contract` 的 `security` 字段读取受保护接口组和 header 约定。学生聊天入口 `/api/v1/support/text`、语音入口 `/api/v1/support/audio`、健康检查和模型状态保持开放，方便学生端联调。

## 2026-06-05 角色视图隐私脱敏层

本轮继续做成熟后端，不生成新报告。新增 `src/campus_support_agent/privacy_redaction.py`，统一识别并遮蔽手机号、邮箱、证件号、学号/工号、微信和 QQ 等直接身份标识。`student` 视图只返回学生端需要字段，并对可见回复做脱敏；`counselor` 视图保留风险、状态、照护和处理上下文，但移除后端内部字段并遮蔽直接身份标识；`research` 视图继续移除自由文本和人工干预备注，并附带 `privacy_redaction` 统计；`admin` 视图保留原始数据，用于授权审计。

验证：`python -m pytest tests\test_privacy_views.py tests\test_main.py -q` 通过 19 项；`python -m pytest -q` 全量通过 355 项。

## 2026-06-05 人工干预工作流摘要

本轮继续推进人机协同工作台所需后端能力。`/api/v1/analytics/care-queue` 的每个队列项现在会在 `evidence.human_workflow` 中返回工作流摘要，包括 `workflow_state`、`owner`、`sla_hours`、`elapsed_hours`、`is_overdue` 和 `next_action`。这些字段由当前风险优先级、队列推荐动作和最新人工干预记录推导，不需要新增数据库表。

当前状态映射：未处理为 `unassigned`，已确认且有处理人为 `assigned`，处理中为 `in_progress`，升级为 `escalated`，结案为 `resolved/closed`。SLA 默认按优先级推导：critical 1 小时、high 4 小时、medium 24 小时、low 72 小时。前端/后台可以据此做认领、逾期和升级提示。

`/api/v1/analytics/care-queue` 也支持工作流筛选：`workflow_state=assigned`、`owner=counselor-001`、`only_overdue=true`，并会在响应 `filters` 中回显当前筛选条件，方便前端工作台做“我的待办、未认领、逾期、已升级”等列表。

## 2026-06-05 人工干预动作接口

新增 `POST /api/v1/sessions/{session_id}/human-interventions/action`，把前端工作台常用动作封装为稳定接口：`claim`、`start`、`escalate`、`resolve`、`close`。接口会自动映射到底层状态 `acknowledged/in_progress/escalated/resolved/closed`，追加 `action:<name>` 标签，并返回最新 `care_queue_item`。`claim/start/escalate` 必须传 `handler_id`，避免无人认领的处理中状态。

## 2026-06-05 Care Queue 角色化降敏视图

`/api/v1/analytics/care-queue` 新增 `role=admin|counselor|research` 查询参数。默认 `admin` 保持完整审计数据；`counselor` 会保留处理上下文但遮蔽人工备注里的手机号、邮箱等直接身份标识；`research` 会进一步移除人工干预备注和下一步文本，只保留结构化统计和状态。这样前端工作台和研究面板可以共用同一个队列接口，但看到不同粒度的数据。

## 2026-06-05 隐私策略与保留配置

新增 `GET /api/v1/privacy/policy`，返回机器可读的角色视图、脱敏类别、受保护接口和数据保留策略。新增环境变量 `SESSION_DATA_RETENTION_DAYS` 与 `AUDIT_LOG_RETENTION_DAYS`，当前只作为策略声明和 readiness 检查，不在请求处理中自动删除数据。`/api/v1/ops/readiness` 会检查保留天数必须为正，并在 session 数据保留时间长于 audit log 时提示 degraded。
