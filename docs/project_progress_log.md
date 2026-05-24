# 项目进展日志

本文件用于记录 Codex 每次对项目的检查、修改、验证结果和下一步建议。运行时接口日志仍查看 `logs/app.log`。

## 2026-05-22 项目层级检查

### 本次做了什么

- 梳理了当前项目结构、README、核心 Agent 编排、API 路由、Schema、SQLite 存储和文档目录。
- 确认项目主题已经与“多模态校园心理熵减与动态平衡系统”基本对齐。
- 重新运行完整测试套件，确认当前代码可通过自动化测试。
- 将当前项目层级快照写入 README，新增本进展日志文件，后续每次继续开发时按此格式追加。

### 当前做到的层级

- 输入层：已支持文本输入和语音文件输入。
- 多模态适配层：已支持 STT provider 抽象，语音会先转写为文本后进入统一 Agent 链路。
- Agent 主链路：已支持风险识别、心理熵评估、状态画像、熵减策略、校园资源检索、支持方案生成和转介建议。
- 动态平衡层：已支持会话记忆、熵轨迹、动态调整、反馈适配、会话连续性、策略重选、趋势预警、照护路径、照护计划和干预效果分析。
- 训练评估层：已支持训练数据导出、反馈坏例构建、SFT/DPO 数据处理、ms-swift 训练脚本和策略/checkpoint 评估脚本。
- 产品化层：已有轻量测试前端和后端 API，但正式前端、权限隐私、真实 STT、人工工作台和告警流程仍需继续补齐。

### 验证结果

```text
python -m pytest
209 passed
```

### 主要判断

当前项目已经超过普通校园心理聊天 Agent 的 MVP，进入“后端闭环原型”阶段。它能够把单轮支持回复扩展成可追踪、可复盘、可训练的数据闭环：输入 -> 风险/熵评估 -> 熵减方案 -> 回复 -> 会话状态持久化 -> 动态调整/反馈适配 -> 后续训练与评估。

### 主要短板

- 多模态目前主要是“语音转文本”，还没有利用音频情绪、语速、停顿等声学特征。
- 前端仍偏研究测试用途，未形成正式用户端和咨询师/管理端工作台。
- 人工转介目前是后端决策和事件记录，还未接入真实通知、排班或工单系统。
- README 内容较长，部分历史段落混有旧说明，后续应整理成“快速运行、系统架构、接口、训练评估、进展日志入口”几块。

### 下一步建议

1. 画清楚系统架构图和数据流图，作为论文/开题/答辩材料的主图。
2. 把 `/app` 前端升级成三栏结构：用户对话、实时熵/风险面板、后台照护建议面板。
3. 给语音输入增加真实 STT 联调记录，并规划声学情绪特征扩展。
4. 做一组固定长对话案例，输出每轮熵值变化、策略变化和最终减熵效果表。
5. 整理 README，减少历史训练细节堆叠，把细节迁移到 docs。

## 2026-05-22 多模态音频信号接入

### 本次做了什么

- 新增 `src/campus_support_agent/multimodal_signal.py`，用于分析上传音频的基础信号。
- `CampusSupportAgent.handle_audio` 现在会先提取音频元数据和基础声学特征，再调用 STT 转写。
- 语音请求的 `student_context` 会注入 `multimodal_signal`，最终响应也会返回 `multimodal_signal`。
- 给 `SupportResponse` 增加 `multimodal_signal` 字段，方便前端、日志、训练导出和后续实验使用。
- 新增测试覆盖：无效 WAV/伪音频只保留元数据和分析备注；有效 WAV 可提取采样率、声道、RMS 能量等字段。

### 当前多模态状态

语音链路现在是：

```text
音频文件 -> 基础音频信号分析 -> STT 转写 -> 文本风险/熵评估 -> 支持方案生成 -> 响应返回 transcript + multimodal_signal
```

目前提取的 WAV 基础特征包括：

- `duration_seconds`
- `sample_rate_hz`
- `channels`
- `sample_width_bits`
- `rms_energy`
- `peak_amplitude`
- `silence_ratio`

这一步把项目从“语音只是转文本”推进到“语音作为多模态证据进入后端闭环”。它还不是完整情绪声学模型，但为后续加入语速、停顿、音高、情绪分类器或外部音频模型预留了稳定接口。

### 验证结果

```text
python -m pytest
210 passed
```

### 下一步建议

1. 在 `/app` 前端显示 `multimodal_signal`，让语音测试时能看到音频证据。
2. 接入真实 STT 服务，记录真实语音样本的转写质量和音频特征。
3. 增加“音频信号 -> 熵维度修正”的轻量规则，例如高静音比例、极低音量时提示表达困难或低能量状态，但避免把声学特征误判为诊断结论。

## 2026-05-22 参考优秀回复对齐与风险边界修正

### 本次做了什么

- 使用 `scripts/evaluate_docx_reference_cases.py` 解析两个 Word 参考文档，共识别出 100 个长对话案例。
- 先抽取前 8 个案例作为小批量对照样本，生成参考回复报告。
- 用当前后端对同一批案例逐轮生成回复，并用现有 `compare_reply` 启发式指标对照参考回复。
- 修正风险识别边界：学业、任务、人际、分手等具体压力语境中的“崩溃、控制不住、撑不住”等表达，不再直接误路由到 high/crisis，除非伴随自伤、轻生、危险地点、伤害自己等明确安全信号。
- 新增本地回复策略：
  - `task_overload_procrastination`：任务拖延堆积、自责、启动困难。
  - `group_work_marginalized`：小组作业被边缘化、贡献不可见、强势组员。
  - `breakup_contact_loop`：分手后反复想联系、查看动态、价值怀疑。
- 增强 `exam_anxiety`：覆盖期末、复习、图书馆等表达，并加入“封卷仪式、证件拍照确认、明早 3 个复习点”等更具体的低压力动作。
- 调整社交/宿舍场景：减少“宿舍”一词过度触发室友冲突策略，增加被冷落、不叫我、担心被讨厌等社交隔离入口。

### 对照评估结果

初始后端小批量对照：

```text
案例数：8
轮数：23
平均启发式评分：69.3
主要问题：weak_action_specificity = 4
额外发现：部分普通压力表达误触发 high/crisis
```

修正后小批量对照：

```text
案例数：8
轮数：23
平均启发式评分：70.74
风险分布：medium = 9, low = 14
危机误触发：0
主要问题：weak_action_specificity = 4
报告：reports/docx_reference_backend_eval/20260522_214820_extracted_docx_reference_eval.md
```

分数提升不大，原因是当前启发式评分较依赖字面重合度；但从行为上看，危机误判已消失，场景策略命中更稳定。后续应继续加强“每轮具体动作”和“根据前文延续具体语境”。

### 验证结果

```text
python -m pytest
214 passed
```

### 下一步建议

1. 把参考文档 100 个案例分成学业、人际、亲密关系、隐性高危、隐私边界等桶，按桶生成差距报告。
2. 把 `weak_action_specificity` 作为下一轮优化目标，要求每轮回复至少包含一个具体、低压力、可执行动作。
3. 将参考回复里的优秀结构沉淀成策略模板：承接情绪 -> 区分事实/解释 -> 缩小问题 -> 给低门槛动作。
4. 优先处理隐性高危 50 例，确保不会漏掉真正安全风险，也不会把普通压力全部升级为危机。

## 2026-05-22 后端参考回复具体行动增强

### 本次做了什么

- 新增 `scripts/evaluate_backend_docx_reference_cases.py`，可直接调用当前 FastAPI 后端 mock 链路，对 Word 长对话参考案例逐轮生成回复并复用 `compare_reply` 评分。
- 清理 `local_response_policy.py` 中重复的小组作业边缘化匹配调用。
- 在 `response_guardrails.py` 增加三类参考回复兜底：
  - 任务拖延/截止期压力：优先级、可提交骨架、25 分钟启动。
  - 小组作业无人回应：具体群消息、可见成果、聊天记录、事实说明。
  - 考前失眠/反复检查：封卷仪式、证件文具拍照、明早 3 个复习点、考场空白三步。
- 修正 `final_reply_guardrails.py` 中“控制不住”在作业/考试/项目等任务语境下被误当成危机语境的问题。
- 给最终回复护栏增加任务截止期 fallback，避免重复回复被替换成过泛的支持话术。

### 对照评估结果

```text
评估样本：Word 前 8 个长对话案例
轮数：23
上一轮平均启发式评分：70.74
本轮平均启发式评分：76.48
主要问题计数：无
最新报告：reports/docx_reference_backend_eval/20260522_215755_extracted_docx_reference_eval.md
```

### 验证结果

```text
python -m pytest
218 passed
```

### 下一步建议

1. 跑完整 100 个 Word 参考案例，按学业、人际、亲密关系、隐性高危、隐私边界分桶统计弱项。
2. 优先检查新增 50 个隐性高危场景，确认真正风险不漏判，同时普通压力不被过度危机化。
3. 把当前兜底回复继续沉淀成更通用的“低压力具体动作增强器”，减少每个场景手写规则的数量。

## 2026-05-22 仓库目录清理与上传边界整理

### 本次做了什么

- 清理项目根目录下的临时文件、测试数据库、缓存目录和旧运行日志，包括 `.pytest_cache/`、`logs/`、`tmp_test_artifacts/`、`test_tmp/`、`tmp*/`、`test_*.db` 等。
- 清理旧报告和本地评估产物，包括 `reports/`、`docs/model_evaluations/`、`docs/monthly_reports/`、月度报告草稿和 Word 渲染缓存。
- 清理可再生成的原始/导出数据，只保留 Git 跟踪的正式校园知识库 `data/campus_knowledge.json`。
- 删除已跟踪但属于旧评估产物的 `reports/api_quality_eval.json` 和 `reports/chat_quality_eval.json`，后续评估报告默认本地生成，不随 GitHub 保存。
- 保留 `.env`、`.venv/`、`.idea/` 和 `training/ms_swift/outputs/`，因为它们分别对应本地配置、虚拟环境、IDE 配置和模型 checkpoint，不适合直接提交但也不应随便删除。
- 更新 `.gitignore`，补充 `tmp*/`、`docs/model_evaluations/`、`docs/monthly_reports/` 和 `docs/monthly_progress_report_*.md`。
- 更新 README，新增“仓库整理与上传边界”，说明 GitHub 应保留的核心文件、应排除的本地产物，以及给组员传递模型时应传 LoRA checkpoint 而不是提交大文件。

### 当前整理后的目录边界

- GitHub 核心内容：`src/`、`tests/`、`scripts/`、`docs/` 正式说明和参考 Word 文档、`data/campus_knowledge.json`、`training/` 训练脚本、`.env.example`、`requirements.txt`、`README.md`。
- 本地保留但不上传：`.env`、`.venv/`、`.idea/`、`training/ms_swift/outputs/`。
- 可再生成且不保留：运行日志、评估报告、测试数据库、pytest 缓存、原始公开语料、训练导出数据、月度报告草稿和渲染缓存。

### 验证结果

```text
python -m pytest
218 passed
```

测试通过后再次清理了测试过程中重新生成的本地日志、pytest 缓存和测试数据库。

### 下一步建议

1. 后续需要给组员模型时，发送 `training/ms_swift/outputs/` 中实际要用的 LoRA checkpoint 目录，并同时说明基础模型路径，例如 Qwen3-4B-Instruct-2507。
2. 如果要重新生成评估报告，先运行对应评估脚本，报告会重新出现在 `reports/`，但默认不提交。
3. 若要把项目交给组员复现，优先让他们从 GitHub 拉代码，再按 README 配置 `.env` 和本地模型路径。

## 2026-05-24 DOCX 参考回复 100 例对齐增强

### 本次做了什么

- 使用 `scripts/evaluate_backend_docx_reference_cases.py --limit 100 --start 1` 跑完整两个 Word 文档中的 100 个长对话参考案例。
- 解析本轮低分和问题标签，确认主要短板仍是回复过泛、缺少具体低压力动作，尤其出现在隐性高危和复杂生活场景的后续轮次。
- 在 `response_guardrails.py` 增加高置信场景路由，覆盖冲动报复/可能伤人、家庭催回县城工作、比赛作品被质疑抄袭、朋友修复不确定、拒绝别人后的内疚、好友疏远、匿名攻击、隐私背叛、考研二战孤独、表白失败、亲密关系自我压低、交通惊吓闪回、嫉妒朋友成功等场景。
- 在 `final_reply_guardrails.py` 增加更靠后的可见回复兜底，修复部分前序策略被最终通用模板覆盖的问题；随后收窄过宽触发条件，避免把考试脑子空白、小组作业、隐私泄露等误路由到上台汇报模板。
- 为新增场景补充回归测试，防止后续退回“泛泛承接但没有行动”的回复。

### 对照评估结果

```text
评估样本：Word 参考案例 100 个
本轮初始平均分：67.75
修正后平均分：69.05
weak_action_specificity：16 -> 12
misses_crisis_safety：1 -> 0
最新报告：reports/docx_reference_backend_eval/20260524_135911_extracted_docx_reference_eval.md
```

### 验证结果

```text
python -m pytest
226 passed
```

### 下一步建议

1. 下一轮优先处理剩余低分场景：酒后失控羞耻、状态好转后担心复发、亲人重病照护压力、连续兴奋冲动消费和睡眠减少、兼职拖欠工资、连续失眠麻木。
2. 把“场景回复”进一步抽象成可组合模板：承接具体处境 -> 区分责任/边界/事实 -> 给一个可执行动作 -> 给转介或现实支持路径。
3. 继续避免过宽关键词触发，尤其“脑子空白”“害怕别人说我想太多”这类跨场景表达，需要结合上下文判断。

## 2026-05-24 模型问题判断与 DOCX 风格 LoRA 补丁训练

### 本次做了什么

- 检查本机训练条件：基础模型 `D:\llm_cache\modelscope\models\Qwen\Qwen3-4B-Instruct-2507` 存在，当前稳定 LoRA `training/ms_swift/outputs/refinement_pool_v5_peft/v0-20260520-215838/checkpoint-final` 存在，CUDA 可用，GPU 为 NVIDIA GeForce RTX 4060 Laptop GPU。
- 确认训练依赖可导入：`transformers 5.3.0`、`peft 0.18.1`、`bitsandbytes 0.49.2`、`datasets 3.6.0`、`accelerate 1.13.0`。
- 由于旧的 `data/training/feedback_bad_cases/eval_behavior_sft_20260511_ms_swift.jsonl` 已在仓库整理中清理，重新用 `scripts/build_targeted_refinement_seed.py` 生成 94 条本地补丁训练数据，覆盖隐私边界、弱输入、宿舍语境、医疗边界、错别字噪声、危机边界、身份边界、风险校准、非模板化支持等类别。
- 先跑 1 step smoke test，确认 Qwen3 4bit 基座和 LoRA 可正常加载训练。
- 基于稳定 LoRA 继续训练实验补丁模型：`training/ms_swift/outputs/docx_targeted_patch_v1/checkpoint-final`，参数为 94 条样本、2 epoch、188 steps、learning rate `2e-6`、`max_length=640`。

### 验证结果

```text
smoke test: 1 step passed, loss ~= 2.687
docx_targeted_patch_v1: 188 steps completed, train_loss ~= 2.611
checkpoint-final files: adapter_config.json, adapter_model.safetensors, tokenizer.json, tokenizer_config.json, training_args.bin
```

短推理验证显示，新 LoRA 能加载并生成中文回复。例如对“我不太敢说，我怕你会告诉辅导员。”，模型能回应“你担心被发现，这很正常。我们先不提具体细节，只说你现在最害怕的是什么。”但它仍没有明确说出“不会主动告诉别人/不需要透露身份信息”等隐私承诺。因此这轮补丁只能算可运行的实验模型，不能替代后端隐私和危机策略兜底。

### 主要判断

当前问题不是单纯模型问题。100 例 DOCX 后端评测走的是 `mock`/规则链路，低分主要来自后端最终回复仍偏泛、部分场景缺少具体低压动作；训练模型可以改善自然表达和泛化，但不能替代隐私、危机、医疗边界这些安全策略。下一步更稳的路线是：先继续把剩余低分场景固化到后端策略层，再扩充高质量 SFT 样本做正式模型对比。

### 下一步建议

1. 暂时继续把 `refinement_pool_v5_peft/v0-20260520-215838/checkpoint-final` 作为给组员的稳定 LoRA；`docx_targeted_patch_v1/checkpoint-final` 标记为实验补丁。
2. 补充一批更明确的隐私边界样本，要求回复稳定包含“不需要透露身份信息”“不会主动告诉别人”“若出现明确危险才建议联系现实支持”等表达。
3. 正式采用新 LoRA 前，应跑完整 checkpoint 场景评估，并与 `refinement_pool_v5_peft` 做同题对照；本轮长评测脚本生成较慢，未完成完整 checkpoint 对比。

## 2026-05-24 DOCX 安全边界补充训练 v2/v3

### 本次做了什么

- 新增 `scripts/build_docx_safety_refinement_seed.py`，专门生成 DOCX 风格的安全边界补丁样本，覆盖显式隐私承诺、少追问、危机边界、医疗边界、上下文纠偏、宿舍边界、身份边界和普通聊天边界。
- 生成 26 条高密度安全样本，并和上一轮 94 条目标补丁样本合成 120 条 v2 训练集。
- 从稳定 LoRA `refinement_pool_v5_peft/v0-20260520-215838/checkpoint-final` 训练 `docx_safety_patch_v2`：120 条样本、3 epoch、360 steps、learning rate `2e-6`。
- v2 短验证后发现隐私、危机、医疗边界仍偏软，于是构建 v3 过采样训练集：94 条基础样本 + 26 条安全样本重复 8 次，共 302 条。
- 从稳定 LoRA 重新训练 `docx_safety_patch_v3`：302 条样本、2 epoch、604 steps、learning rate `8e-6`。
- 补强 `LocalCheckpointLLMProvider` 的 system prompt，明确写入隐私、危机、医疗和用药边界，避免本地 checkpoint 裸生成时只做情绪安抚。

### 验证结果

```text
docx_safety_patch_v2: checkpoint-final generated
docx_safety_patch_v3: checkpoint-final generated
关键单测: python -m pytest tests/test_agent.py tests/test_main.py
结果: 16 passed
```

v3 裸推理验证中，隐私样例已经能输出“不主动联系辅导员、保护隐私边界”；少追问样例能尊重用户不想解释的边界。加入接近后端的 system prompt 后，隐私样例进一步稳定为“不主动联系学校、不透露给第三方，除非涉及人身安全”；医疗样例能提示不能仅凭症状判断，并建议校医院或医生排除身体原因。危机样例已有“先不要见对方或直接冲突”，但仍需要后端规则兜底补上“联系现实支持/校园安保/紧急电话”等更强动作。

### 当前模型结论

- 稳定推荐给组员复现：`training/ms_swift/outputs/refinement_pool_v5_peft/v0-20260520-215838/checkpoint-final`
- 最新实验安全补丁：`training/ms_swift/outputs/docx_safety_patch_v3/checkpoint-final`
- 当前不建议只靠 LoRA 处理危机安全；必须保留后端风险识别、guardrails 和最终回复兜底。

### 下一步建议

1. 用 v3 跑一轮完整 checkpoint 场景评估，并与稳定 LoRA 同题对照。
2. 把危机场景继续固化到 `response_guardrails.py` 和 `final_reply_guardrails.py`，确保无论模型输出如何，最终回复都包含立即安全动作。
3. 下一轮训练数据应增加多轮上下文样本，而不是继续堆单轮样本；目前模型对“同一句话不同上下文”的边界判断仍依赖后端策略层。
