from __future__ import annotations


def maybe_build_followup_reply(user_text: str, history_text: str) -> str | None:
    user = _compact(user_text)
    history = _compact(history_text)
    combined = f"{history}{user}"

    if _major_direction_vague_or_ten_minute(user, combined):
        return (
            "不喜欢现在的专业时，先不要把问题压成“必须马上转走或彻底认命”。你现在更需要的是重新找一点可验证的方向感。"
            "如果只能做十分钟，就写下三列：现在专业里最不排斥的一门课或技能、你真正好奇的领域、它们之间可能连接的一个选修/辅修/项目关键词。"
            "这一步不是立刻决定人生，而是把“我很空”变成一个可以继续试探的小入口。"
        )
    if _pet_grief_photo_followup(user, combined):
        return (
            "看到照片就受不了，说明这段告别还在很新鲜的位置，不是你太脆弱。宠物离开后，家里少掉的是一个真实陪伴过你的日常成员。"
            "今晚可以先别反复翻相册，选三张最想留下的照片，给其中一张写一句话，比如“谢谢你陪我到这里”。"
            "写完后把手机放远一点，允许自己难过，但不要让照片循环把你一直拉回最痛的那一刻。"
        )
    if _bullying_next_time_response(user, combined):
        return (
            "如果下次又有人这样开玩笑，你可以先用很短、很稳的一句把边界说出来："
            "“这个模仿让我不舒服，别这样学我说话。”说完就停，不需要马上解释童年经历，也不需要证明自己是不是小题大做。"
            "如果对方继续，优先离开现场，事后找一个可信同学、老师或辅导员说明具体发生了什么。重点是保护现在的你，而不是在现场硬撑。"
        )
    if _paranoia_vague_followup(user, combined):
        return (
            "你说不清、觉得堵，同时前面已经提到被盯着、被监控和连续睡不好，我会更重视你的睡眠和安全感。"
            "今晚先不要继续查证手机或反复观察别人反应，找一个可信的人陪你待一会儿；明天尽快联系校医院、心理中心或正规医疗机构做评估。"
        )
    if _exam_vague_block(user, combined):
        return (
            "你现在说“很堵”，更像是考试前压力卡在身体里，不一定需要马上分析清楚。"
            "先做一个很小的卸压动作：把最担心的一句话写下来，再写旁边一句“明天先做会的题”。"
            "如果还睡不着，就把目标改成闭眼休息 20 分钟，而不是逼自己立刻睡着。"
        )
    if _exam_minimal_step(user, combined):
        return (
            "只做一小步的话，就先把明天要带的证件、文具和准考信息放到同一个地方，拍一张照片确认。"
            "做完这一步就停，不再反复检查。这个动作的目标不是复习更多，而是让大脑知道“最基本的安全项已经准备好了”。"
        )
    if _interview_start_block(user, combined):
        return (
            "启动不了时不要先整理完整项目。只做十分钟：打开一个空文档，写下今天面试里被问倒的一个问题，再写三行：题目问什么、我当时卡在哪里、下次第一句话怎么开头。"
            "十分钟后可以停。先让材料从脑子里落到纸面上，比继续自责更重要。"
        )
    if _mental_center_stigma(user, combined):
        return (
            "去心理中心不代表你“有问题”，更像是在状态持续低下时做一次支持和评估。"
            "你可以把它理解成给自己找一个现实里的协助点，而不是给自己贴标签。"
            "如果现在还抗拒，可以先不预约，只先查一下学校心理中心的预约方式和开放时间，把信息放在手边。"
        )
    if _low_motivation_vague_or_step(user, combined):
        return (
            "这种“堵”和没动力更像是状态持续低下后，大脑和身体都在省电，不是简单懒或者废。"
            "现在先不要把任务、论文、未来都拉到一起想。只做一个能把你拉回现实的小动作：坐起来喝水、洗脸、打开窗帘，或者把今天必须做的一件事写成一句话。"
            "如果这种灰暗感已经持续一段时间，心理中心可以先当作一次状态评估，而不是给自己贴标签。"
        )
    if _lonely_crying_friend_fear(user, combined):
        return (
            "怕打扰朋友很能理解，但人在夜里崩溃时，完全一个人扛会更孤单。"
            "你不需要把所有细节讲完，可以只发一句低压力消息：“我今晚有点撑不住，能不能陪我聊十分钟，不方便也没关系。”"
            "真正的求助不一定是把痛苦全部交出去，而是先让一个人知道你现在不太好。"
        )
    if _night_crying_lonely_followup(user, combined):
        return (
            "你觉得自己很多余，通常不是事实本身，而是长期强撑后在夜里冒出来的孤立感。"
            "明天如果还要见到别人，不需要装得完全没事；先维持最基本的节奏，比如上课、吃饭、回消息只回必要内容。"
            "今晚更重要的是不要继续一个人憋到最深处，可以给一个低压力的人发一句：“我今晚有点难受，不用你解决，能陪我说几句就好。”"
        )
    if _body_image_vague_or_small_step(user, combined):
        return (
            "如果只做一小步，先暂停一次检查：把镜子、相册或修图软件离开 20 分钟，去做一件和外貌无关的事，比如喝水、洗脸、走到楼下。"
            "目标不是马上喜欢自己的样子，而是先把注意力从反复扫描缺点里拉出来一点。"
        )
    if _social_embarrassment_initial(user, combined):
        return (
            "发错消息后的尴尬会被大脑放大成“所有人都会记得”，但多数人的注意力不会停留那么久。"
            "现在先不要反复回看群聊或猜别人怎么想。可以准备一句简短解释：“刚才发错了，已经撤回，不好意思。”说到这里就够了，不需要继续自我审判。"
        )
    if _social_embarrassment_next_day(user, combined):
        return (
            "明天见到他们时，先保持正常节奏就好，不需要主动把事情重新翻出来。"
            "如果有人提起，你可以简短说：“昨天发错了，有点尴尬，已经处理了。”然后把话题转回眼前的课或事情。"
            "重点是不要用退缩让这件小插曲变成你生活里的大事件。"
        )
    if _class_activity_manyu_followup(user, combined):
        return (
            "你觉得自己很多余，是因为那次活动里确实没有被带进去，而不是你这个人没有位置。"
            "下一步先不要把目标定成融入整个圈子，可以只选一个低压力入口：课后问一个同学作业、下次活动前找一个人同行，或主动接一个具体小任务。"
            "存在感可以从一两个稳定的小连接开始，不需要一次证明自己属于所有人。"
        )
    if _class_activity_next_day_followup(user, combined):
        return (
            "明天见到他们时，先维持正常节奏，不需要急着解释自己为什么那天沉默。"
            "你可以只做一个很小的连接动作：和相对熟悉的人打个招呼，或问一句具体问题，比如“上次作业你做到哪了？”"
            "目标不是马上加入圈子，而是让关系有一个可继续的小入口。"
        )
    if _binge_eating_initial(user, combined):
        return (
            "昨晚停不下来和今天想不吃饭补回来，说明你现在被羞耻和补偿冲动夹住了。"
            "但用不吃饭惩罚自己，往往会让下一次更容易失控。今天先不要极端补偿，至少保留一顿温和、规律的进食，比如粥、鸡蛋、面包或你能接受的简单食物。"
            "暴食不是你失败了，而是身体和情绪都需要被更稳定地照顾。"
        )
    if _binge_eating_small_step(user, combined):
        return (
            "只做一小步的话，先不要称体重、不要跳过下一顿。给自己准备一份容易入口的正常食物，吃到七分就停。"
            "这一步不是放纵，而是在打断“暴食-羞耻-节食-再暴食”的循环。"
        )
    if _bullying_trigger_initial(user, combined):
        return (
            "别人学你说话、周围人笑起来，会把你一下拉回过去被嘲笑的记忆里。你的身体发冷、想逃，不是小题大做，而是旧经历被触发了。"
            "先把现在和过去分开：今天这件事让你难受，但你已经不是当年那个只能被困住的人。今晚先做一件稳定身体的事，比如踩实地面、喝水、离开反复回想的场景。"
        )
    if _bullying_trigger_small_step(user, combined):
        return (
            "只做一小步的话，先把身体从那一刻带回来：双脚踩地，慢慢呼气，然后说出三个现在能看到的东西。"
            "等身体没那么紧，再决定要不要和一个可信同学或老师说这件事。现在先不急着证明他们有没有恶意，先照顾被触发的你。"
        )
    if _relationship_checking_initial(user, combined):
        return (
            "你不是单纯想控制对方，更像是在用查岗确认“我没有被丢下”。这种不安很真实，但如果总靠对方即时回复来缓解，你会越来越累，对方也会有压力。"
            "先不要急着责备自己，可以把冲动分成两部分：我现在需要安全感，以及我准备用什么方式获得安全感。"
            "下次想追问时先等 15 分钟，写下自己真正担心的句子，再等平静一点和对方商量一个具体规则，比如忙的时候能不能提前说一声。"
        )
    if _relationship_need_fear(user, combined):
        return (
            "健康的关系里，需求可以被讨论，不等于麻烦。关键是表达方式：少用指责，多说自己的感受和希望。"
            "你可以说：“我有时候会不安，也在努力调整；如果你很久不能回，能不能提前告诉我一声？”"
            "这样既照顾你的安全感，也尊重对方空间。对方是否愿意一起协商，也能帮助你判断这段关系是不是让你更稳定。"
        )
    if _family_money_initial(user, combined):
        return (
            "你难受的不是简单的省钱压力，而是把家庭经济紧张和“我是不是负担”连在了一起。爸妈没有怪你，但你已经在心里替他们责备自己。"
            "读大学确实有成本，但这不等于你是负担。现在可以先把羞耻感和现实问题分开：哪些开支可以规划，哪些学校支持资源可以了解。"
            "你能做的是逐步承担，而不是用“马上回报很多钱”来证明自己有用。"
        )
    if _family_money_private(user, combined):
        return (
            "不想被同情、不想让家里更担心，都能理解。你可以选择更低暴露的方式处理现实问题，比如查看学校勤工助学、奖助学金、临时困难补助，或找辅导员只问政策。"
            "和家里沟通也可以只谈计划：“我会控制生活费，也会看看学校有没有补助。”这能让他们看到你在行动，而不是增加焦虑。"
        )
    if _family_money_earning_pressure(user, combined):
        return (
            "想减轻家里负担是很负责任的想法，但“马上赚很多钱”可能会把你推向更大的压力。"
            "学生阶段更现实的目标是：控制非必要支出、争取奖助资源、做不影响学业的兼职或项目、提升未来就业能力。"
            "你不是现在不能立刻回报就没用，你正在走一条让自己以后更有能力的路。"
        )
    if _ordinary_self_worth_initial(user, combined):
        return (
            "你现在像是站在一面只放大别人亮点的镜子前看自己，所以越看越觉得自己平淡。普通不等于没有价值，只是它没有像朋友圈那样被包装出来。"
            "你提到成绩、外貌、社交，都是容易被比较的维度；但一个人值得被喜欢，也包括稳定、真诚、负责、能陪伴、愿意成长这些不那么显眼的部分。"
        )
    if _ordinary_self_worth_evidence(user, combined):
        return (
            "那我们先不用空泛夸你，改成找证据。过去一个月里，有没有哪怕很小的事情是你完成了、坚持了、帮到别人了，或者比以前进步了一点？"
            "“拿得出手”不一定是大奖，也可能是按时交了一个很难的作业、陪朋友度过低落、在害怕时还是去了课堂。价值感可以从具体证据里重新建立。"
        )
    if _ordinary_self_worth_defect_filter(user, combined):
        return (
            "当人处在低落比较里，大脑会自动过滤优点，只留下缺点，这不是事实完整版本。"
            "你可以先做一个很机械的练习：每天记录三件“没有更糟”的事，比如按时起床、认真听了十分钟课、没有把情绪发泄给别人。"
            "它们很小，但能帮你重新看见自己在维持生活。你不需要先变得闪闪发光，才值得被善待。"
        )
    if _social_opening_followup(user, combined):
        return (
            "可以先不用“插入中心话题”，而是从低风险回应开始。比如别人聊课程，你可以说：“这个作业我也卡住了，你们做到哪了？”"
            "别人聊活动，你可以问：“你们是怎么报名的？”这种问题不需要你很有趣，只需要和现场有关。"
            "社交初期的目标不是让别人立刻喜欢你，而是让对话能自然延续两三句。"
        )
    if _class_activity_isolation_initial(user, combined):
        return (
            "你去了活动，却没有感到被接纳，这种落差很伤人。不是你矫情，因为人到了集体里仍然被忽视，会比一个人待着更孤单。"
            "现在可以先承认：今天的体验确实不好，但它不一定代表你在班里永远没有位置。"
        )
    if _class_connection_followup(user, combined):
        return (
            "不需要先变成活跃气氛的人。你可以从一对一连接开始，比融入大群更容易。"
            "比如选一个看起来相对友善的同学，课后问作业、一起去食堂，或者在下次活动前主动找一个人结伴。存在感不一定靠成为中心建立，也可以靠稳定的小连接慢慢形成。"
        )
    if _class_circle_fear_followup(user, combined):
        return (
            "圈子不是完全封闭的，但确实需要时间。你可以不把目标设成“加入一个圈子”，而是设成“和两三个具体的人建立可聊天关系”。"
            "同时给自己保留班级之外的社交来源，比如社团、项目组、兴趣活动。一个班级的冷淡体验，不应该决定你全部的人际价值。"
        )
    if _social_approach_initial(user, combined):
        return (
            "你不是不想社交，而是每次准备靠近时，大脑都提前替你想象了被拒绝、被嫌弃的结果，所以你一直停在门口。"
            "认识朋友不一定要从很精彩的聊天开始，很多关系都是从普通、甚至有点笨拙的互动开始的。"
        )
    if _graduation_overload_initial(user, combined):
        return (
            "你现在不是没有努力，而是信息太多，导致决策系统过载。毕业选择看起来像决定一生，但更实际的做法是把它拆成可测试的阶段，而不是一次性选出完美答案。"
        )
    if _graduation_choice_fear(user, combined):
        return (
            "可以用三个维度筛选：你能承受的备考周期，你当前最接近的能力证据，你最不能接受的生活状态。"
            "然后给每条路做一个两周验证任务：投几份岗位、做一套考研真题、看一套公考题。用真实体验替代纯想象。"
        )
    if _graduation_start_block(user, combined):
        return (
            "那今天只做一个最小动作：写下三条路各自的最小验证任务，并预约明天的一个时间块。"
            "迷茫时不要追求“想通”，先让身体开始行动，行动会给你新的信息。"
        )
    if _future_path_initial(user, combined):
        return (
            "你不是没有想法，而是同时看到了每条路的收益和代价，所以被选择压力卡住。重大选择很少有绝对正确，更多是看它和你的能力、资源、时间、风险承受度是否匹配。"
            "现在可以先别问“哪条最好”，而是问“哪条更适合我目前的条件”。"
        )
    if _future_path_specific_worries(user, combined):
        return (
            "这两个担心都很现实。可以做一个对照表：考研需要的准备时间、目标院校难度、英语补强计划；就业需要的项目、实习、简历和面试准备。"
            "然后给每条路打分，不是凭感觉，而是看三个月内能不能做出明显进展。也可以设置六周观察期，同时做考研信息收集和简历项目梳理，再根据行动反馈调整。"
        )
    if _future_path_delay_fear(user, combined):
        return (
            "所以现在需要的是阶段性决策，而不是一次定终身。你可以从今天开始定两个并行任务：每周固定英语和专业基础复习，同时完善一个项目写进简历。"
            "六周后看哪条路推进得更稳定。迷茫时最怕只想不做，哪怕暂时不确定方向，也可以先做对两条路都有帮助的事。"
        )
    if _peer_offer_comparison_initial(user, combined):
        return (
            "同辈进展特别容易触发焦虑，因为它看起来像一张排名表。但朋友圈展示的是结果，不是完整过程，也不是你的唯一参照系。"
            "你现在需要的不是强迫自己不比较，而是把比较带来的压力转成自己的下一步行动。"
        )
    if _peer_offer_start_fear(user, combined):
        return (
            "当差距看起来很大时，最有效的是缩小战场。你可以先选一个目标：找实习、准备考研或准备考公，暂时不要同时追所有人的路。"
            "然后列出最短路径：如果是实习，就是简历、一个项目、基础题、投递渠道。每项只定本周任务，不要一次规划一年。"
        )
    if _peer_offer_too_slow(user, combined):
        return (
            "慢不等于停。很多人的节奏不是直线，有人早拿 offer，也有人后期补上。"
            "你可以把评价从“我有没有超过别人”换成“我这周有没有比上周更清楚一点”。今晚先完成一件可见的小事：更新简历一页，或整理一个项目说明。"
        )
    if _dorm_noise_initial(user, combined):
        return (
            "你已经忍了很久，所以现在的烦不是突然小题大做，而是长期睡眠被打扰后的累积反应。你一边想维护关系，一边又需要基本休息，这两个需求都合理。"
            "现在的问题不是你该不该生气，而是怎样把边界说清楚，同时尽量降低冲突升级。你不用先证明自己不是“针对她”，也不需要向无关的人透露太多宿舍细节；先把重点放在作息规则和基本休息权利上。"
        )
    if _dorm_targeting_fear(user, combined):
        return (
            "可以用“我感受+具体时间+可替代做法”的方式说，避免上来评价她。比如：“我最近睡眠很差，晚上 12 点后听到电话声会很难入睡。你能不能 12 点后去走廊接，或者尽量用耳机小声说？”"
            "如果担心单独说尴尬，可以先在群里提一个共同规则，例如熄灯后尽量降低声音。"
        )
    if _dorm_snark_fear(user, combined):
        return (
            "你不需要吵架，只需要重复边界。她如果阴阳怪气，你可以保持一句话：“我不是针对你，我只是需要晚上能睡觉。”"
            "不要被带到人身评价里。若多次沟通无效，可以记录具体时间，再找宿舍长、辅导员或宿管协调。寻求协调不是告状，而是在基本休息权利被持续影响时保护自己。"
        )
    if _academic_specific_block(user, combined):
        return (
            "考不好会难受，但它不会把这学期的努力全部清零。现在焦虑把结果放大成了“全盘失败”，所以今晚先不要和整门课对抗，只和一小块内容对抗。"
            "建议你用一张纸写下：今晚最小任务、明天可补任务、实在不会就放弃的部分。数据库这里可以先只抓范式和依赖的两类典型题，各做一道，做完就停。"
        )
    if _project_shallow_feedback(user, combined):
        return (
            "“浅”听起来确实会伤人，尤其你已经投入了很多时间。可以把这个词翻译成更可操作的版本：数据支撑不够？场景设计不够细？算法只是调用接口？"
            "反馈如果停留在评价层面，会让人羞耻；如果变成修改清单，就能重新服务项目。你可以先列三栏：老师原话、可能含义、下一步能改的一个动作。"
        )
    if _project_withdrawal_after_feedback(user, combined):
        return (
            "想退缩是正常反应，尤其是在投入后被否定的时候。但你不必现在决定“以后还做不做项目”。"
            "先给自己一个短暂停顿，然后只做复盘：哪些地方是能力不足，哪些是时间不足，哪些是题目定位没想清楚。下一次可以先让老师或同学看早期方案，减少后期被整体否定的冲击。"
        )
    if _teacher_ability_fear(user, combined):
        return (
            "老师对你的印象不是由一次作业永久决定的。相反，如果你能根据反馈修改，并在下一次作业里体现改进，这会比一次失误更有说服力。"
            "可以在提交时附一句：“根据上次反馈，我重点调整了结构和格式。”这不是卑微解释，而是让老师看见你在回应问题。"
        )
    if _task_pile_procrastination_initial(user, combined):
        return (
            "你现在的痛苦有两层：一层是任务真的堆起来了，另一层是你一直用“我很废”来攻击自己。自责会制造压力，但通常不会带来行动。"
            "先把评价放一边，只看眼前局面：哪些任务有硬截止时间，哪些可以降低完成标准，哪些可以先交一个可用版本。今晚只选最急的一项，先做 25 分钟可提交骨架。"
        )
    if _task_deadline_report_followup(user, combined):
        return (
            "那优先级已经很清楚：先救明天晚上的实验报告。你可以把“写好报告”改成“先凑出可提交骨架”：实验目的、环境、核心步骤、运行结果、问题分析。"
            "每一部分先写三到五句话，不追求漂亮。写出来之后再补图、改语病。对于怕写得烂，可以先允许第一版很粗糙，因为粗糙版本比空白文档更容易修改。"
        )
    if _task_time_comparison_followup(user, combined):
        return (
            "你看到的是别人交出来的结果，不一定看到他们也焦虑、卡住、临时赶工的过程。现在不需要把自己和“完美时间管理的人”比较。"
            "今天可以做一个很小的修复：设一个 25 分钟计时，只写实验报告的目录和第一节，手机放远，结束后休息 5 分钟。完成一次，大脑会知道自己还能重新启动。"
        )
    if _breakup_contact_fear(user, combined):
        return (
            "这个害怕背后是“我还想被记得、被重视”。但用不断联系来确认存在感，通常会让你更被动，也让伤口一直被重新打开。"
            "可以先给自己设一个短期边界：三天不主动查看、不主动发消息。难受时转向朋友、运动、写下来或做一件固定的小事。你不是假装不在乎，而是在把注意力一点点从对方那里拿回来。"
        )
    if _breakup_self_worth(user, combined):
        return (
            "被承诺过又被离开，确实容易让人怀疑自己的价值。但一个人没能继续兑现关系，不等于你不值得被认真对待。"
            "可以把问题从“我哪里不够好”换成“这段关系里哪些需求没有被满足”。如果想发消息，先写在备忘录里，不发送，等 24 小时后再决定。"
        )
    if _crush_uncertainty_initial(user, combined):
        return (
            "你卡在不确定里：对方的关心给了你期待，回复慢又让你不安。暗恋最消耗人的地方，就是很多情绪都只能自己解释。"
            "你不一定马上表白，但需要先看见这段关系对你的影响：它有没有让你越来越失去自己的节奏？可以先把事实和猜测分开，再决定要不要用更清楚的方式靠近。"
        )
    if _crush_waiting_message_followup(user, combined):
        return (
            "这说明你已经把很多情绪控制权交给了他的反应。可以先做两个小调整：第一，给聊天设一个边界，比如不因为等回复打断学习；第二，观察他是否也主动投入，而不是只看他偶尔的温柔。"
            "如果你想推进关系，可以选择一个自然场景，约他一起做具体事情，比如一起吃饭或看展，用真实互动替代反复猜测。"
        )
    if _crush_rejection_fear_followup(user, combined):
        return (
            "这个风险确实存在，所以你可以不用选择突然、强烈的表白。可以先用低压力方式表达好感，例如：“我和你相处挺开心的，之后可以多一起出去吗？”"
            "对方的回应会给你更多信息。无论结果怎样，你都不是在丢脸，而是在从长期猜测里走出来。你值得拥有更清楚、更稳定的关系。"
        )
    if _traffic_avoidance_followup(user, combined):
        return (
            "可以用渐进方式恢复，而不是逼自己一下子完全不怕。先从短距离、熟悉路线、有人陪同开始；上车前告诉自己“我现在是在新的场景，不是那天”。"
            "紧张时把脚踩实，观察车内具体物品，把注意力带回当下。如果反应持续很强，影响出行，可以考虑找心理咨询师帮助处理。"
        )
    if _traffic_shame_followup(user, combined):
        return (
            "“差点出事”也足以让人害怕，因为你的身体经历的是危险信号。你不需要用事故严重程度来证明自己有资格难受。"
            "可以对别人简单说：“那次之后我坐车会紧张，还在慢慢适应。”重点不是让所有人理解，而是允许自己按安全的节奏恢复。"
        )
    if _anonymous_explain_fear(user, combined):
        return (
            "可以先不急着公开回应。公开争辩容易被带节奏。更稳的是：截图保留发布时间、账号、评论；联系平台或管理员要求删除涉及隐私的内容；告知辅导员或可信老师。"
            "需要回应时，用简短事实声明，不进入情绪互骂。"
        )
    if _anonymous_class_avoidance(user, combined):
        return (
            "可以允许自己短暂缓冲，但不要长期躲起来。找一个同学陪你去第一节课，坐在让你安全的位置。"
            "你被攻击不等于你做错了。把支持系统拉进来，比一个人对抗匿名恶意更有效。"
        )
    if _online_refresh_followup(user, combined):
        return (
            "这是很自然的反应，你想寻找支持来抵消攻击。但不断刷新会让你的情绪被评论区牵着走。"
            "可以先做一个边界：把评论通知关掉，半天内不再查看；如果有明显辱骂，可以删除、拉黑或举报。然后找一个真实朋友看作品，问具体建议，而不是让匿名评论决定你的价值。"
        )
    if _online_publish_fear(user, combined):
        return (
            "暂时不想发可以理解，但不要让几条恶意评论永久夺走你的表达空间。"
            "你可以先把作品保存下来，过几天再看哪些反馈有参考价值，哪些只是情绪垃圾。下次发布时可以选择更安全的平台、限制评论或先发给小范围朋友。你可以调整保护方式，不必放弃创作。"
        )
    if _privacy_disappear_followup(user, combined):
        return (
            "我理解你想消失是因为太害怕，但现在越孤立越危险。请先找一个最可信的人，只说必要事实：“我遇到隐私泄露风险，现在很慌，需要你陪我一下。”"
            "同时保留聊天记录、截图、时间线，不要删除证据。如果你有伤害自己的冲动，或者觉得自己控制不住，请立刻联系身边的人、学校老师或当地紧急求助。"
        )
    if _privacy_bed_cry_followup(user, combined):
        return (
            "先把今晚目标降到最低：不单独处理、不联系前任、不删除证据、让一个可信的人知道你的位置。"
            "之后可以联系辅导员、学校心理中心、平台投诉渠道，必要时咨询警方或法律援助。你现在需要的是保护和支持，不是把羞耻全揽到自己身上。"
        )
    if _group_strong_member_followup(user, combined):
        return (
            "面对强势组员，表达可以尽量具体、短、可执行，减少对抗感。比如可以在群里说：“我看目前方案已经定了，为了保证我这边也有实际贡献，我可以负责第 3 部分资料和 PPT 排版，今晚 10 点前给初稿。需要你们把参考资料发我一下。”"
            "这句话重点不是抱怨，而是把你的参与变成可见任务。"
        )
    if _document_criticism_resistance(user, combined):
        return (
            "抗拒是因为文档已经和那次尴尬绑定在一起了。可以先降低进入难度，不要求你马上完整修改。"
            "第一步只做“标记问题”：格式、逻辑、引用、语言各用一种颜色圈出来。第二步再选最容易改的一类先动手。这样你是在处理文件，不是在重新经历被批评。"
        )
    if _presentation_stuck_followup(user, combined):
        return (
            "你可以提前准备一句救场话：“这里我稍微整理一下思路。”然后看一眼 PPT 或卡片，继续讲下一点。"
            "听众通常不会像你想象的那样盯着错误，他们更关注内容是否能听懂。你需要练的不是完全不卡，而是卡住后能回来。能回来，就是一次成功的展示。"
        )

    if _parent_keeps_asking(user, combined):
        return (
            "如果她继续问，你可以温和重复边界，而不是每个问题都解释到她放心。"
            "可以说：“这个我现在不想细说，但我会照顾好自己；我每周固定和你说一次近况。”"
            "这样既给她一点安全信息，也保留你的空间。边界不是冷漠，而是为了让你们不用每次都靠争吵证明彼此在乎。"
        )
    if _family_choice_guilt(user, combined):
        return (
            "内疚说明你在乎家人，不说明你的选择错了。"
            "可以把“照顾家人”从“完全按他们安排走”换成更具体的责任：保持联系、让他们知道你的计划、为选择准备备选方案。"
            "成年后的孝顺不一定是放弃自己，也可以是带着规划去承担自己的路。"
        )
    if _relationship_how_to_break_up(user, combined):
        return (
            "如果要分，先把安全和边界放在前面。优先选择文字方式，或有第三方在场的公开安全环境；不要单独见面拉扯。"
            "分手内容可以短一点：“我已经决定结束这段关系，之后不再继续争论。你如果状态危险，请联系家人、朋友或紧急支持。”"
            "保存威胁证据，提前告诉一个可信朋友；如果对方再用伤害自己威胁，就联系能实际到场的人，而不是你一个人去救。"
        )
    if _relationship_decision_exhaustion(user, combined):
        return (
            "这说明问题不是简单的喜欢或不喜欢，而是关系里同时有依恋和消耗。"
            "可以先不急着逼自己回答“分不分”，而是看三个条件：这段关系长期让你安心还是紧绷，沟通过后有没有真实改善，如果未来半年都这样你能不能接受。"
            "这些问题比让别人替你下结论更接近你的真实答案。"
        )
    if _interview_future_fear(user, combined):
        return (
            "以后不一定完全不紧张，但你可以减少“被问倒”的比例。"
            "把项目整理成固定模板：背景、你的职责、技术栈、难点、解决方案、结果、可以改进的地方。"
            "再找同学模拟追问两轮，把答不上来的问题补成 2 分钟回答骨架。面试能力是可以训练的，不是一次失败就定型。"
        )
    if _social_cold_response(user, combined):
        return (
            "对方反应冷会尴尬，但它不一定等于你被否定，也可能只是对方累、不熟、没接住话。"
            "你可以把一次互动当成尝试，而不是人格考试。先设一个很小的目标：本周主动问三个人一个具体问题，比如“这个作业你开始了吗？”"
            "结果先不评价，只记录自己做到了靠近。"
        )
    if _self_worth_after_rejection(user, combined):
        return (
            "喜欢没有成功，不一定是你“不够好”，更多时候是对方的感受、时机和需求不同。"
            "先把自尊从这次结果里拿回来，减少反复回看聊天记录，做几件能让你重新稳定的事：按时吃饭、运动十分钟、和朋友正常说句话。"
            "你认真表达过喜欢，这不是可笑的事。"
        )
    if _online_attack_initial(user, combined):
        return (
            "被陌生人当众贬低会痛，因为作品里有你的投入和表达。你难受不是太脆弱，而是评论越过了作品反馈，变成了对你这个人的攻击。"
            "现在最重要的是先别反复刷那些评论，让伤口被一次次刺激。可以截图保留证据、关掉通知、删除或举报人身攻击。"
            "等情绪降一点，再找一个真实可信的人看作品，问具体建议，而不是让匿名评论决定你的价值。"
        )
    if _privacy_leak_initial(user, combined):
        return (
            "这件事会带来强烈恐慌和羞耻，但责任不在你。未经同意传播私密内容是严重侵犯。"
            "现在先确认安全：你此刻身边有没有可以陪你的人？不要一个人硬扛，也不要急着和对方单独对质。"
            "先保留聊天记录、截图和时间线；如果需要处理，可以找可信同学、辅导员、学校支持渠道或法律援助一起看下一步。"
        )
    if _stalking_initial(user, combined):
        return (
            "这个经历会让人很不安，你现在的警觉是身体在保护你。先不要急着判断是不是“想多了”，而是按安全事件来处理。"
            "可以把时间、地点、对方特征和路线写下来，必要时告知宿管、辅导员或校园安保。"
            "短期先不要独自走夜路，尽量结伴、走人多有灯的路线；如果再次发生，优先进入便利店、值班室这类有人场所并联系现实支持。"
        )
    if _sleep_numb_safe_followup(user, combined):
        return (
            "那今晚先不要独自熬着。既然舍友在，可以只告诉她一句：“我最近失眠很严重，今晚状态有点撑不住，能不能知道一下我的情况。”"
            "明天尽快联系校医院或心理中心，连续失眠加情绪麻木需要支持。你不需要把问题讲完整，只要说“我一周睡不好，并出现不想醒来的念头”，这就足够求助。"
        )
    if _work_pay_blacklist_fear(user, combined):
        return (
            "怕被拉黑很正常，所以这一步更要留下证据，而不是只靠口头催。"
            "可以先发一条文字确认：“我想确认本月兼职工资的具体发放时间，请今天给我明确答复。”如果对方不回复或拉黑，聊天记录、工作时间和约定工资都能作为后续维权材料。"
            "如果继续拖延，可以找学校老师、法律援助或劳动维权渠道一起处理。"
        )
    if _work_pay_assertion(user, combined):
        return (
            "你不是在求他，你是在要求对方履行约定。不好意思很正常，很多学生第一次面对拖欠工资都会紧张，但劳动报酬是你的正当权益。"
            "下一步先把沟通改成文字确认：“我想确认本月兼职工资的具体发放时间，请今天给我明确答复。”"
            "同时整理工作时间、约定工资和聊天记录；如果继续拖延，再找学校老师、法律援助或劳动维权渠道一起处理。"
        )

    return None


def _compact(text: str) -> str:
    return "".join(str(text or "").split())


def _major_direction_vague_or_ten_minute(user: str, combined: str) -> bool:
    return any(term in user for term in ("很空", "找不到方向", "十分钟", "一小步", "先做哪一步", "不能转专业")) and any(
        term in combined for term in ("不喜欢现在的专业", "转专业", "专业内外", "上课听不进去", "辅修", "选修")
    )


def _pet_grief_photo_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("看到照片", "反复翻照片", "受不了", "脑子很乱", "很堵")) and any(
        term in combined for term in ("宠物", "离世", "走了", "家里少了一块")
    )


def _bullying_next_time_response(user: str, combined: str) -> bool:
    return any(term in user for term in ("下次", "又有人这样", "开玩笑", "怎么回", "说什么")) and any(
        term in combined for term in ("学我说话", "被嘲笑", "周围人都笑", "手发冷", "想逃")
    )


def _exam_vague_block(user: str, combined: str) -> bool:
    return any(term in user for term in ("很堵", "不知道怎么说")) and any(
        term in combined for term in ("明天", "考试", "考场", "脑子空白", "睡不着")
    ) and not _danger_or_self_harm_context(combined) and not _paranoia_context(combined)


def _exam_minimal_step(user: str, combined: str) -> bool:
    return any(term in user for term in ("一小步", "先做哪一步")) and any(
        term in combined for term in ("明天", "考试", "考场", "脑子空白", "准考", "文具")
    ) and not _danger_or_self_harm_context(combined) and not _paranoia_context(combined)


def _interview_start_block(user: str, combined: str) -> bool:
    if any(term in combined for term in ("不喜欢现在的专业", "转专业", "专业内外")):
        return False
    return any(term in user for term in ("启动不了", "只能做十分钟", "十分钟")) and any(
        term in combined for term in ("面试", "项目细节", "基础问题", "被问倒", "实习")
    )


def _mental_center_stigma(user: str, combined: str) -> bool:
    return any(term in user for term in ("心理中心", "代表我有问题", "去了就代表")) and any(
        term in combined for term in ("没动力", "生活", "灰暗", "游戏", "逃避", "睡")
    )


def _low_motivation_vague_or_step(user: str, combined: str) -> bool:
    return any(term in user for term in ("很堵", "不知道怎么说", "一小步", "先做哪一步", "启动不了")) and any(
        term in combined for term in ("没动力", "灰暗", "游戏", "逃避", "睡到很晚", "心理中心", "生活变得")
    )


def _lonely_crying_friend_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("怕打扰朋友", "不敢找人", "哭的时候", "很孤单")) and any(
        term in combined for term in ("晚上", "崩溃", "哭", "白天强撑", "孤单")
    )


def _night_crying_lonely_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("很多余", "明天还要见到", "见到他们", "一小步", "很堵")) and any(
        term in combined for term in ("白天强撑", "晚上崩溃", "哭", "哭的时候", "怕打扰朋友")
    )


def _body_image_vague_or_small_step(user: str, combined: str) -> bool:
    return any(term in user for term in ("很堵", "一小步", "先做哪一步")) and any(
        term in combined for term in ("脸肿", "修图", "照镜子", "外貌", "照片", "不像自己")
    )


def _social_embarrassment_initial(user: str, combined: str) -> bool:
    if any(term in user for term in ("明天还要见到", "见到他们", "怎么回一句")):
        return False
    return any(term in combined for term in ("发到了班群", "发错", "撤回", "尴尬到想退学", "社死"))


def _social_embarrassment_next_day(user: str, combined: str) -> bool:
    return any(term in user for term in ("明天还要见到", "见到他们", "先做什么")) and any(
        term in combined for term in ("班群", "发错", "撤回", "尴尬", "社死")
    )


def _class_activity_manyu_followup(user: str, combined: str) -> bool:
    return "很多余" in user and any(
        term in combined for term in ("班级活动", "拍照", "没人叫我", "站在旁边", "班里没有存在感")
    )


def _class_activity_next_day_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("明天还要见到", "见到他们", "先做什么")) and any(
        term in combined for term in ("班级活动", "班里", "没人叫我", "站在旁边", "圈子", "插不进去")
    )


def _binge_eating_initial(user: str, combined: str) -> bool:
    if any(term in user for term in ("一小步", "先做哪一步")):
        return False
    return any(term in combined for term in ("吃了很多", "停不下来", "恶心自己", "不吃饭补回来", "接受不了")) and any(
        term in combined for term in ("昨晚", "今天", "暴食", "吃完")
    )


def _binge_eating_small_step(user: str, combined: str) -> bool:
    return any(term in user for term in ("一小步", "先做哪一步")) and any(
        term in combined for term in ("吃了很多", "停不下来", "不吃饭补回来", "暴食", "恶心自己")
    )


def _bullying_trigger_initial(user: str, combined: str) -> bool:
    if any(term in user for term in ("下次", "又有人这样", "开玩笑", "怎么回", "说什么", "一小步", "先做哪一步", "很堵")):
        return False
    return any(term in combined for term in ("学我说话", "周围人都笑", "初中被嘲笑", "被嘲笑")) and any(
        term in combined for term in ("手发冷", "想逃", "身体一下子僵住", "旧经历")
    )


def _bullying_trigger_small_step(user: str, combined: str) -> bool:
    return any(term in user for term in ("一小步", "先做哪一步", "很堵")) and any(
        term in combined for term in ("学我说话", "被嘲笑", "周围人都笑")
    ) and any(
        term in combined for term in ("手发冷", "旧时", "想逃", "身体一下子僵住")
    )


def _paranoia_vague_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("很堵", "不知道怎么说", "一小步", "先做哪一步")) and any(
        term in combined for term in ("被监控", "盯着我", "很多人都在盯", "几晚没睡", "正规医疗机构", "校医院")
    )


def _paranoia_context(text: str) -> bool:
    return any(term in text for term in ("被监控", "盯着我", "很多人都在盯", "几晚没睡", "正规医疗机构", "校医院"))


def _danger_or_self_harm_context(text: str) -> bool:
    return any(
        term in text
        for term in (
            "不想活",
            "想死",
            "伤害自己",
            "天台",
            "楼顶",
            "高处",
            "消失几天",
            "不在了",
            "账号密码",
            "控制不住",
        )
    )


def _relationship_checking_initial(user: str, combined: str) -> bool:
    if any(term in user for term in ("提需求", "觉得我麻烦", "怕我麻烦")):
        return False
    return any(term in combined for term in ("谈恋爱", "对象", "男朋友", "女朋友")) and any(
        term in combined for term in ("几个小时不回", "胡思乱想", "查岗", "问他在哪里", "问她在哪里", "控制不住")
    )


def _relationship_need_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("提需求", "觉得我麻烦", "怕我麻烦")) and any(
        term in combined for term in ("不回", "安全感", "查岗", "恋爱", "对象")
    )


def _family_money_initial(user: str, combined: str) -> bool:
    return any(term in combined for term in ("家里说钱", "钱有点紧", "家庭经济", "花了很多钱")) and any(
        term in combined for term in ("负担", "罪恶感", "没什么回报", "花钱")
    )


def _family_money_private(user: str, combined: str) -> bool:
    return any(term in user for term in ("不想跟同学说", "怕别人觉得我可怜", "不想跟家里提", "怕他们更担心")) and any(
        term in combined for term in ("钱", "经济", "生活费", "负担", "罪恶感")
    )


def _family_money_earning_pressure(user: str, combined: str) -> bool:
    return any(term in user for term in ("马上赚很多钱", "很没用", "应该马上赚")) and any(
        term in combined for term in ("家里", "钱", "经济", "负担", "回报")
    )


def _ordinary_self_worth_initial(user: str, combined: str) -> bool:
    if any(term in user for term in ("很虚", "拿得出手", "想不到", "都是缺点", "脑子里都是缺点")):
        return False
    if any(term in combined for term in ("班级活动", "拍照", "班里", "很多余")):
        return False
    return any(term in combined for term in ("很普通", "长相也一般", "社交也不厉害", "每天上课", "吃饭", "回宿舍")) and any(
        term in combined for term in ("值得被喜欢", "朋友圈里大家", "没有存在感")
    )


def _ordinary_self_worth_evidence(user: str, combined: str) -> bool:
    return any(term in user for term in ("很虚", "拿得出手")) and any(
        term in combined for term in ("很普通", "值得被喜欢", "长相", "社交")
    )


def _ordinary_self_worth_defect_filter(user: str, combined: str) -> bool:
    return any(term in user for term in ("想不到", "都是缺点", "脑子里都是缺点")) and any(
        term in combined for term in ("很普通", "值得被喜欢", "拿得出手", "缺点")
    )


def _social_approach_initial(user: str, combined: str) -> bool:
    if any(term in user for term in ("不知道开口", "插进去", "反应很冷", "很尴尬")):
        return False
    return any(term in combined for term in ("想认识新朋友", "怕打扰别人", "觉得我无聊", "加入聊天", "站在旁边"))


def _social_opening_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("不知道开口", "插进去很奇怪", "聊得很热闹")) and any(
        term in combined for term in ("新朋友", "打扰别人", "加入聊天", "社交")
    )


def _class_activity_isolation_initial(user: str, combined: str) -> bool:
    return any(term in combined for term in ("班级活动", "拍照", "没人叫我", "班里没有存在感", "很多余")) and any(
        term in combined for term in ("三三两两", "站在旁边", "回宿舍", "没有存在感")
    )


def _class_connection_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("怎么改变", "活跃气氛")) and any(
        term in combined for term in ("班级活动", "班里", "没有存在感", "很多余")
    )


def _class_circle_fear_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("自己的圈子", "插不进去")) and any(
        term in combined for term in ("班级活动", "班里", "没有存在感", "很多余")
    )


def _graduation_overload_initial(user: str, combined: str) -> bool:
    return any(term in combined for term in ("马上毕业", "考研", "考公", "找工作")) and any(
        term in user for term in ("刷经验帖", "越刷越乱", "每条路都有人成功", "每条路")
    )


def _graduation_choice_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("怕选错", "选错")) and any(
        term in combined for term in ("马上毕业", "考研", "考公", "找工作")
    )


def _graduation_start_block(user: str, combined: str) -> bool:
    return any(term in user for term in ("启动不了", "就是启动不了")) and any(
        term in combined for term in ("马上毕业", "考研", "考公", "找工作", "经验帖")
    )


def _future_path_initial(user: str, combined: str) -> bool:
    if any(term in user for term in ("成绩一般", "英语", "项目不够", "简历不好看", "拖到最后", "什么都没准备好", "怕选错", "启动不了")):
        return False
    return any(term in combined for term in ("考研", "就业", "考公")) and any(
        term in combined for term in ("未来特别迷茫", "怕自己选错", "越想越乱", "每条路")
    )


def _future_path_specific_worries(user: str, combined: str) -> bool:
    return any(term in user for term in ("成绩一般", "英语", "项目不够", "简历不好看")) and any(
        term in combined for term in ("考研", "就业", "考公", "选错")
    )


def _future_path_delay_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("拖到最后", "什么都没准备好")) and any(
        term in combined for term in ("考研", "就业", "考公", "简历")
    )


def _peer_offer_comparison_initial(user: str, combined: str) -> bool:
    if any(term in user for term in ("不知道从哪里开始", "要补的东西太多", "一焦虑", "太慢", "怕自己太慢")):
        return False
    return any(term in combined for term in ("offer", "大厂实习", "考研上岸", "落后太多")) and any(
        term in combined for term in ("朋友圈", "同学", "比较", "往前走")
    )


def _peer_offer_start_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("不知道从哪里开始", "要补的东西太多", "一焦虑")) and any(
        term in combined for term in ("offer", "大厂实习", "考研上岸", "落后")
    )


def _peer_offer_too_slow(user: str, combined: str) -> bool:
    return any(term in user for term in ("太慢", "怕自己太慢")) and any(
        term in combined for term in ("offer", "大厂实习", "考研上岸", "落后")
    )


def _dorm_noise_initial(user: str, combined: str) -> bool:
    if any(term in user for term in ("针对她", "不敢说", "阴阳怪气", "不会吵架")):
        return False
    return any(term in combined for term in ("室友", "舍友", "宿舍")) and any(
        term in combined for term in ("很晚打电话", "笑得很大声", "戴耳塞", "关系弄僵", "晚上她一开口")
    )


def _dorm_targeting_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("针对她", "她们好像没那么介意", "更不敢说")) and any(
        term in combined for term in ("室友", "打电话", "睡眠", "戴耳塞")
    )


def _dorm_snark_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("阴阳怪气", "不会吵架")) and any(
        term in combined for term in ("室友", "打电话", "睡眠", "针对她")
    )


def _academic_specific_block(user: str, combined: str) -> bool:
    return any(term in combined for term in ("数据库", "范式", "依赖")) and any(
        term in user for term in ("做题", "心跳", "来不及", "一看就乱", "考试")
    )


def _project_shallow_feedback(user: str, combined: str) -> bool:
    return any(term in user for term in ("实现思路比较浅", "创新性不够", "小学生作品", "浅")) and any(
        term in combined for term in ("老师", "项目", "作品", "反馈")
    )


def _project_withdrawal_after_feedback(user: str, combined: str) -> bool:
    return any(term in user for term in ("不想再做项目", "下一次还是被否定", "怕下一次")) and any(
        term in combined for term in ("项目", "老师", "反馈", "否定", "实现思路")
    )


def _teacher_ability_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("老师以后觉得我能力差", "老师觉得我能力差", "能力差")) and any(
        term in combined for term in ("作业", "实验报告", "反馈", "老师")
    )


def _task_pile_procrastination_initial(user: str, combined: str) -> bool:
    return any(term in user for term in ("拖延", "任务全堆")) and any(
        term in combined for term in ("实验报告", "改代码", "英语展示", "刷手机", "凌晨", "头皮发麻", "我很废")
    )


def _task_deadline_report_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("实验报告明天晚上交", "打开文档就想逃", "写出来很烂")) and any(
        term in combined for term in ("实验报告", "英语展示", "代码", "拖延")
    )


def _task_time_comparison_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("别人都能安排好时间", "永远这样")) and any(
        term in combined for term in ("拖延", "实验报告", "刷手机", "任务")
    )


def _breakup_contact_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("怕不联系", "彻底忘了我", "真的忘了我")) and any(
        term in combined for term in ("分手", "前任", "他以前说过", "她以前说过", "联系")
    )


def _breakup_self_worth(user: str, combined: str) -> bool:
    return any(term in user for term in ("不值得被认真对待", "以前说过会一直陪", "像没事人")) and any(
        term in combined for term in ("分手", "前任", "联系", "忘了我")
    )


def _crush_uncertainty_initial(user: str, combined: str) -> bool:
    if any(term in user for term in ("等他消息", "反复琢磨", "如果他拒绝", "连朋友都做不成")):
        return False
    return any(term in combined for term in ("喜欢一个同学", "一直不敢说", "分不清", "普通朋友", "回复慢一点")) and any(
        term in combined for term in ("心情", "暗恋", "关心我", "不敢说")
    )


def _crush_waiting_message_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("等他消息", "反复琢磨", "什么意思")) and any(
        term in combined for term in ("喜欢一个同学", "普通朋友", "回复慢", "暗恋")
    )


def _crush_rejection_fear_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("如果他拒绝", "连朋友都做不成")) and any(
        term in combined for term in ("喜欢一个同学", "表白", "暗恋", "普通朋友")
    )


def _group_strong_member_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("觉得我事多", "组员很强势", "大家就跟着走")) and any(
        term in combined for term in ("小组", "组员", "分工", "贡献")
    )


def _document_criticism_resistance(user: str, combined: str) -> bool:
    return any(term in user for term in ("打开文档", "老师的语气", "很抗拒")) and any(
        term in combined for term in ("老师", "作业", "文档", "反馈", "能力差")
    )


def _presentation_stuck_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("真的卡住", "卡住了怎么办")) and any(
        term in combined for term in ("展示", "PPT", "忘词", "演讲", "公开表达")
    )


def _traffic_avoidance_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("不太敢坐车", "不坐又不现实")) and any(
        term in combined for term in ("坐车", "差点出事故", "急刹车", "闪回")
    )


def _traffic_shame_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("别人觉得我矫情", "没有真的出事")) and any(
        term in combined for term in ("坐车", "差点出事故", "急刹车", "闪回")
    )


def _anonymous_explain_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("想解释", "越解释越糟")) and any(
        term in combined for term in ("表白墙", "匿名", "朋友圈截图", "评论里")
    )


def _anonymous_class_avoidance(user: str, combined: str) -> bool:
    return any(term in user for term in ("不想去上课", "认出我")) and any(
        term in combined for term in ("表白墙", "匿名", "朋友圈截图", "评论里")
    )


def _online_refresh_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("忍不住想刷新", "有没有人帮我说话")) and any(
        term in combined for term in ("网上", "评论", "阴阳怪气", "作品")
    )


def _online_publish_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("不敢再发东西", "不敢发东西")) and any(
        term in combined for term in ("网上", "评论", "阴阳怪气", "作品")
    )


def _privacy_disappear_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("把手机关了", "消失几天", "太丢脸")) and any(
        term in combined for term in ("私密照片", "照片发给", "泄露", "前任")
    )


def _privacy_bed_cry_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("宿舍床上哭", "床上哭", "没有做什么")) and any(
        term in combined for term in ("私密照片", "照片发给", "泄露", "前任")
    )


def _parent_keeps_asking(user: str, combined: str) -> bool:
    return any(term in combined for term in ("妈妈", "我妈", "父母", "爸妈", "她越问", "问得太细")) and any(
        term in user for term in ("继续问", "还会问", "一直问", "问个不停")
    )


def _family_choice_guilt(user: str, combined: str) -> bool:
    return "内疚" in user and any(term in combined for term in ("回县城", "考编", "大城市", "开发", "家里", "父母", "爸妈"))


def _relationship_how_to_break_up(user: str, combined: str) -> bool:
    return any(term in user for term in ("怎么分", "怎么提分手", "怎么离开")) and any(
        term in combined for term in ("分手", "威胁", "做傻事", "伤害自己", "控制")
    )


def _relationship_decision_exhaustion(user: str, combined: str) -> bool:
    return any(term in user for term in ("在一起很累", "分开也很难受", "分开也难受")) and any(
        term in combined for term in ("该不该", "分手", "看不清", "关系")
    )


def _interview_future_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("以后面试", "还是这样", "还会这样")) and any(
        term in combined for term in ("面试", "项目", "被问倒", "基础问题")
    )


def _social_cold_response(user: str, combined: str) -> bool:
    return any(term in user for term in ("反应很冷", "很冷淡", "会很尴尬")) and any(
        term in combined for term in ("主动认识", "新朋友", "打招呼", "社交")
    )


def _self_worth_after_rejection(user: str, combined: str) -> bool:
    return any(term in user for term in ("不够好", "不值得", "很差劲")) and any(
        term in combined for term in ("表白", "被拒", "喜欢", "拒绝")
    )


def _online_attack_initial(user: str, combined: str) -> bool:
    return any(term in combined for term in ("网上", "评论", "表白墙", "匿名", "阴阳怪气")) and any(
        term in combined for term in ("难受", "羞辱", "攻击", "贬低")
    )


def _privacy_leak_initial(user: str, combined: str) -> bool:
    return any(term in combined for term in ("私密照片", "照片发给", "照片泄露", "前任")) and any(
        term in combined for term in ("抖", "完了", "没脸", "恐慌", "奇怪的语气")
    )


def _stalking_initial(user: str, combined: str) -> bool:
    return any(term in combined for term in ("跟在我后面", "尾随", "一直跟", "加快他也加快")) and any(
        term in combined for term in ("天黑", "害怕", "回头看", "便利店")
    )


def _sleep_numb_safe_followup(user: str, combined: str) -> bool:
    return any(term in user for term in ("舍友在", "只是很累", "宿舍")) and any(
        term in combined for term in ("醒不醒都差不多", "不想醒", "连续失眠", "睡不好")
    )


def _work_pay_assertion(user: str, combined: str) -> bool:
    return any(term in combined for term in ("兼职", "工资", "老板", "拖欠", "下周发")) and any(
        term in combined for term in ("不好意思", "求他", "自己赚的钱", "需要这笔钱")
    )


def _work_pay_blacklist_fear(user: str, combined: str) -> bool:
    return any(term in user for term in ("拉黑", "问急了")) and any(
        term in combined for term in ("兼职", "工资", "老板", "拖欠", "下周发")
    )
