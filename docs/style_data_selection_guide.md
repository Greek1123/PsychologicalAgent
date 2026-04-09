# 风格层语料筛选说明

这份说明专门对应当前项目的第一阶段目标：

- 先训练模型“会聊”
- 先训练自然、共情、多轮支持风格
- 暂时不把复杂分析能力融进每一句回复里

## 适合进入 style_sft 的语料

- 多轮支持对话
- 至少包含 1 轮完整的 user / assistant 往返
- 助手语言自然、温和、共情
- 助手会轻追问、反映、总结
- 不强行讲大道理

## 不建议直接进入 style_sft 的内容

- 过长的内部推理说明
- 显式的治疗理论讲解
- 过强的诊断口吻
- 明显像教材答案而不是自然对话的内容
- 太短的单轮片段

## 当前项目中的处理策略

对于原始文件：

- [cn_data_version7.json](d:/psychologicalAgent/data/cn_data_version7.json)
- [en_data_version7.json](d:/psychologicalAgent/data/en_data_version7.json)

转换脚本会：

1. 抽取 `dialog`
2. 把 `Seeker / Supporter` 映射成 `user / assistant`
3. 自动补 `system`
4. 根据 `stage` 推断 `stage_goal`
5. 去掉 `reasoning` / `guide` 等不适合直接训练风格层的字段

## 为什么这样做

因为你的第一阶段目标不是训练一个会暴露分析过程的模型，而是训练一个自然、稳定、支持性的校园心理陪伴对话模型。
