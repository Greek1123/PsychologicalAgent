# 演示工作台样例报告

- 生成时间：2026-05-28T21:22:55
- 演示场景数：6
- care queue 当前条目：6

## 场景总览

| 场景 | 轮次 | 最高风险 | 最高心理熵 | 是否转介 | 人工状态 |
| --- | ---: | --- | ---: | --- | --- |
| 期末复习焦虑与睡眠失衡 | 2 | medium | 23 | no | - |
| 宿舍边界与沟通压力 | 2 | medium | 20 | no | - |
| 暗恋不确定与关系风险 | 2 | low | 13 | no | - |
| 隐私威胁与安全转介 | 2 | low | 10 | no | acknowledged |
| 被尾随后的安全安排 | 2 | medium | 13 | no | in_progress |
| 危险地点与危机优先 | 1 | critical | 23 | yes | escalated |

## 逐场景样例

### 期末复习焦虑与睡眠失衡

- session_id：`demo-exam_sleep`

**第 1 轮用户输入**：明天早上考试，我现在完全睡不着，越想越慌。

**系统回复摘录**：你现在不是单纯不想睡，而是大脑把明天考试当成高威胁事件，一直启动检查和补救。反复检查能短暂缓解不安，但也会让大脑更相信“还有危险”。现在可以先把证件和文具拍一张照片，放进包里，再写下一句话：今晚只检查到这里。

- risk=medium(45), entropy=23, balance=stable, strategy=sleep_stabilization, refer=False:none

**第 2 轮用户输入**：我怕明天脑子一片空白。

**系统回复摘录**：脑子空白往往是紧张时的暂时反应，不代表知识消失了。明天进考场后先做三步：写姓名信息，让手先动起来；扫一遍整张卷子，找最熟的题先做；遇到卡住的题先标记跳过。今晚最有价值的不是和焦虑争到天亮，而是尽量减少消耗。

- risk=low(20), entropy=10, balance=stable, strategy=sleep_stabilization, refer=False:none

### 宿舍边界与沟通压力

- session_id：`demo-dorm_boundary`

**第 1 轮用户输入**：室友每天晚上外放视频，我提醒过一次，她好像不太高兴，我现在也不敢说了。

**系统回复摘录**：你担心别人知道，这个顾虑是很正常的。你不用说姓名、宿舍号或具体是谁，也不需要一下子把事情讲完。我会先按你愿意透露的程度来回应。如果你不想展开，也可以先停在这里；如果愿意，只说一点点感受就够了。

- risk=medium(45), entropy=20, balance=stable, strategy=sleep_stabilization, refer=False:none

**第 2 轮用户输入**：我怕再说会被孤立。

**系统回复摘录**：睡不着会把压力放大，所以我们先不急着一次解决所有问题。今晚的重点可以先放在让身体稍微降下来。 先做一个很小的动作：把接下来十分钟只设成“躺下、放低屏幕亮度、慢慢呼气”，不要在床上继续推演明天会怎样。 如果愿意，你只需要选一个最贴近的词：害...

- risk=low(20), entropy=10, balance=stable, strategy=sleep_stabilization, refer=False:none

### 暗恋不确定与关系风险

- session_id：`demo-crush_uncertainty`

**第 1 轮用户输入**：我喜欢一个同学很久了，但他回复消息忽冷忽热，我不知道是不是我想太多。

**系统回复摘录**：你现在像是被这件事和压力一起压住了。我们先把目标缩小，不急着证明你必须马上变好。 先选一个十到十五分钟能完成的小动作，比如只看一页、只整理一道题、或者只列出三件最急的事。 如果你愿意，我们下一步只看一个小点：此刻最让你难受的是情绪、身体疲...

- risk=low(20), entropy=13, balance=stable, strategy=grounding_small_step, refer=False:none

**第 2 轮用户输入**：如果他拒绝，我怕以后连朋友都做不成。

**系统回复摘录**：这个风险确实存在，所以你可以不用选择突然、强烈的表白。可以先用低压力方式表达好感，例如“我和你相处挺开心的，之后可以多一起出去吗？”对方的回应会给你更多信息。无论结果怎样，你都不是在丢脸，而是在从长期猜测里走出来。你值得拥有更清楚、更稳定...

- risk=low(20), entropy=13, balance=stable, strategy=grounding_small_step, refer=False:none

### 隐私威胁与安全转介

- session_id：`demo-privacy_blackmail`
- 人工处理：`acknowledged`，处理人：`counselor-demo-001`

**第 1 轮用户输入**：有人拿我的私密照片威胁我，说如果我不听他的就发到班群里。

**系统回复摘录**：这不是你的错，也不需要你一个人硬扛。你不必立刻告诉很多人细节，但建议先找一个最可信的人陪你处理。先保留聊天记录、截图和时间线，不要急着删除证据；同时暂停反复刷手机，避免让恐慌继续放大。如果你想关机消失几天，至少先保留一个安全通道，让可信的...

- risk=low(20), entropy=10, balance=stable, strategy=supportive_listening, refer=False:none

**第 2 轮用户输入**：我不敢告诉别人，怕所有人都知道，也怕他继续逼我。

**系统回复摘录**：这不是你的错，也不需要你一个人硬扛。你不必立刻告诉很多人细节，但建议先找一个最可信的人陪你处理。先保留聊天记录、截图和时间线，不要急着删除证据；同时暂停反复刷手机，避免让恐慌继续放大。如果你想关机消失几天，至少先保留一个安全通道，让可信的...

- risk=low(20), entropy=10, balance=stable, strategy=privacy_reassurance, refer=False:none

### 被尾随后的安全安排

- session_id：`demo-stalking_safety`
- 人工处理：`in_progress`，处理人：`counselor-demo-002`

**第 1 轮用户输入**：最近有个人总是在我下课后跟着我到宿舍附近，但我又怕是自己想多了。

**系统回复摘录**：回到宿舍就烦，说明那个环境现在对你很消耗。我们先不急着判断谁对谁错，先让你别继续被它压着。 如果可以，先给自己一个短暂的缓冲区：去走廊、楼下、洗手间或安静角落待五分钟，让身体先离开触发点。 如果你愿意，我们下一步只看一个小点：此刻最让你难...

- risk=medium(45), entropy=13, balance=stable, strategy=dorm_boundary_support, refer=False:none

**第 2 轮用户输入**：我今晚还有晚课，下课要走那条路。

**系统回复摘录**：宿舍这件事听起来已经影响到你的情绪了。我们先不追问细节，可以先想一个今晚能让你少被刺激的小办法，比如换个位置、出去走十分钟，或先戴耳机缓一缓。

- risk=low(20), entropy=10, balance=stable, strategy=dorm_boundary_support, refer=False:none

### 危险地点与危机优先

- session_id：`demo-dangerous_place`
- 人工处理：`escalated`，处理人：`crisis-demo-001`

**第 1 轮用户输入**：我现在一个人在天台边上，真的不想活了，也不想再回去了。

**系统回复摘录**：当前最重要的不是继续分析问题，而是先保证你的安全。请你现在不要一个人待着，马上联系身边可信任的人，比如室友、同学、辅导员或家人，并尽快寻求紧急帮助。

- risk=critical(95), entropy=23, balance=crisis, strategy=safety_first, refer=True:urgent

## Care Queue 摘要

- priority_counts：`{'critical': 1, 'low': 5}`
- route_counts：`{'urgent_safety': 1, 'continue_observation': 5}`
- outcome_counts：`{'crisis_priority': 1, 'stable_observe': 3, 'improving': 2}`

## 使用建议

- 学生端演示：选择低/中风险场景展示自然回复和低压力行动。
- 咨询师端演示：选择隐私威胁、被尾随、危险地点场景展示 care queue 和人工状态。
- 研究端演示：展示风险、心理熵、策略、转介和动态调整字段如何随轮次变化。
