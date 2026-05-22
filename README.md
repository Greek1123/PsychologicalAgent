# 校园心理支持 Agent MVP

这是一个适合你当前课题的最小可用后端：围绕“文本/语音输入 -> 风险识别 -> 心理支持方案生成 -> 校园转介提示”搭了一个可扩展的 Agent 骨架。

这版的目标不是直接替代心理咨询，而是先把系统主链路跑通，并把 `EmoLLM` 放在“可替换模型后端”的位置上。这样你后面无论接本地部署的 EmoLLM、LMDeploy 服务，还是先拿别的兼容模型联调，代码都不用大改。

## 当前项目层级快照

更新时间：2026-05-22。

当前项目已经从最初的 MVP 进入到“多模态校园心理 Agent + 心理熵评估 + 熵减策略 + 会话级动态平衡闭环”的系统雏形阶段。文本和语音入口已经接入同一条 Agent 主链路，后端能够完成风险识别、心理熵评分、主要熵源识别、校园资源检索、支持方案生成、转介建议、会话记忆、熵轨迹持久化和后续策略调整。

按层级看，目前完成度如下：

- L1 输入与服务层：已完成。支持 `POST /api/v1/support/text` 文本输入、`POST /api/v1/support/audio` 语音文件输入、`/app` 轻量测试前端、`/health` 和模型状态接口。
- L2 模型与多模态适配层：基本完成。LLM 支持 `mock`、OpenAI 兼容接口和本地 `local_checkpoint`；语音转写支持 `mock` 与 OpenAI 兼容 STT。语音请求现在会同时记录基础音频信号 `multimodal_signal`，WAV 文件可提取时长、采样率、声道数、RMS 能量、峰值和静音比例。当前还不是完整的音频情绪识别，但已经不再只保留转写文本。
- L3 校园心理 Agent 决策层：已完成核心闭环。包含安全分流、状态画像、心理熵评估、熵减策略、干预策略选择、校园知识库资源匹配和最终支持回复生成。
- L4 动态平衡与纵向追踪层：已具备雏形并持续扩展。系统会保存会话历史、熵轨迹、支持回合、转介事件和用户反馈，并基于这些数据生成动态调整、会话连续性、策略重选、趋势预警、照护路径、照护计划、干预效果和熵减结果评估。
- L5 训练与评估层：已建立工具链。支持训练数据导出、反馈坏例导出、DPO/SFT 数据构建、ms-swift 脚本、策略层评估、checkpoint 场景评估和月度报告材料。
- L6 产品化与真实部署层：部分完成。已有测试前端和完整后端 API，但还需要进一步做正式前端体验、权限/隐私、真实 STT 服务、人工咨询师工作台、告警通知和真实校园流程对接。

当前一句话定位：项目已经不是单纯聊天机器人，而是一个后端能力较完整的“校园心理熵减与动态平衡 Agent 原型系统”；下一阶段应重点补齐产品化前端、真实语音能力、人工干预闭环和实验评估报告。

最新对齐进展：已将 `docs/心理助手长对话模拟测试用例50例.docx` 和 `docs/心理助手长对话模拟测试用例_新增50例_含隐性高危场景.docx` 作为参考回复库接入评估流程。当前后端已针对期末复习崩溃、任务拖延堆积、小组作业边缘化、分手后反复联系、宿舍冷落/社交隔离等场景补充本地策略，并修正“崩溃/控制不住”等普通压力语境被误判为危机的边界。最新小批量参考评估平均分 76.48，前 8 例未再出现 `weak_action_specificity` 主问题计数，报告见 `reports/docx_reference_backend_eval/20260522_215755_extracted_docx_reference_eval.md`。

## 协作与进展记录

后续每次由 Codex 继续开发或检查时，同步更新两类记录：

- `README.md`：只记录当前能力层级、运行方式、重要接口和阶段性结论。
- `docs/project_progress_log.md`：按日期追加“本次做了什么、验证结果、遗留问题、下一步建议”。

运行时日志仍保存在 `logs/app.log`，用于查看接口请求、风险判断、熵评估、资源检索、存储和异常信息。

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
- 结构化输出：情绪评估、压力源、保护因子、熵水平、平衡状态、支持计划、安全提示
- 单元测试：覆盖文本低风险、危机分流、语音转写链路

## 项目结构

```text
.
├─ .env.example
├─ requirements.txt
├─ src/
│  └─ campus_support_agent/
│     ├─ agent.py
│     ├─ config.py
│     ├─ main.py
│     ├─ memory.py
│     ├─ prompts.py
│     ├─ providers.py
│     ├─ retrieval.py
│     ├─ safety.py
│     └─ schemas.py
├─ data/
│  └─ campus_knowledge.json
└─ tests/
   └─ test_agent.py
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
- EmoLLM README é¨å‹«åŽ¤ç’ï½…ï¼é„åº¡å·±ç’‹å†¨å¾æµ å‘®å½æ¸šæ¶™å„ç¼î…æ•®éŽ¸ä½·ç¬Œå¯¤é¸¿î†…é”›å±¼ç¬‰é‘³èŠ¥æµ›æµ ï½„ç¬“æ¶“æ°¬ç¸¾éžå——æŒ©ç’‡î®ç´°https://github.com/SmartFlowAI/EmoLLM#readme

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
