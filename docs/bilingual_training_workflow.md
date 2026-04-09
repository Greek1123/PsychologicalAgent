# 中英双语训练工作流

这份文档对应当前项目更推荐的训练顺序：

1. 先训练说话习惯，而不是先训练诊断能力
2. 先用多轮支持对话做 SFT，让模型“很会聊”
3. 再做风格偏好对齐
4. 最后单独训练分析层

## 推荐的数据拆分

### 1. style_sft

目标：

- 自然
- 共情
- 会追问
- 会总结
- 不急着讲道理

特点：

- 多轮对话
- 中英双语都要有
- 不要求输出风险或心理熵 JSON

### 2. style_dpo

目标：

- 做风格对齐
- 让模型学会更合适、更安全、更像咨询支持的表达

特点：

- 同一个 prompt 下给出 `chosen / rejected`
- 中英双语都要覆盖

### 3. analysis_sft

目标：

- 单独训练第二层分析能力
- 输出结构化结果

特点：

- 输出统一 JSON
- 包括 `risk / entropy / entropy_reduction / plan`
- 字段名统一用英文

## 双语数据建议

- 中文与英文保持同一任务结构
- 中文优先可以占 60%-70%
- 英文建议占 30%-40%
- 英文最好做人工润色，不建议直接大规模机翻

## 阶段目标建议

为了让多轮支持更自然，可以在训练数据里加入 `stage_goal`：

- `emotional_containment`
- `gentle_probe_and_reframe`
- `summary_and_next_step`
- `safety_stabilization`

## 当前项目里的对应工具

- 导出分析训练样本：
  [export_training_data.py](d:/psychologicalAgent/scripts/export_training_data.py)
- 生成双语模板：
  [generate_bilingual_templates.py](d:/psychologicalAgent/scripts/generate_bilingual_templates.py)
- 熵标签参考：
  [entropy_labeling_guide.md](d:/psychologicalAgent/docs/entropy_labeling_guide.md)
