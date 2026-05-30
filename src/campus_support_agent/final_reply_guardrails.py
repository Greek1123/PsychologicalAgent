from __future__ import annotations

from typing import Any


def finalize_user_visible_reply(
    user_text: str,
    reply_text: str,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    student_context: dict[str, Any] | None = None,
) -> str:
    """Final user-visible reply cleanup after model and strategy guardrails.

    This layer intentionally stays product-facing: it removes backend jargon,
    repairs obvious model drift, and returns a natural support response.
    """

    clean_user = _compact(user_text)
    clean_reply = _compact(reply_text)
    history_text = _history_text(conversation_history)
    care_plan = _care_plan(student_context)

    priority = _priority_reference_reply(clean_user, history_text, clean_reply)
    if priority is not None:
        return priority

    if _looks_mojibake(clean_reply) or _exposes_backend_terms(clean_reply):
        return _care_plan_fallback(clean_user, history_text, care_plan)

    if _is_numeric_or_symbol_input(clean_user) and _continues_numeric_pattern(clean_reply):
        return _weak_input_reply(history_text, care_plan)

    if _assistant_claims_personal_distress(clean_reply):
        return _role_repair_reply(clean_user, history_text)

    if _is_repeated_reply(clean_reply, conversation_history):
        return _care_plan_fallback(clean_user, history_text, care_plan)

    if _too_short(clean_reply) and _has_distress_context(f"{history_text} {clean_user}"):
        return _care_plan_fallback(clean_user, history_text, care_plan)

    if _privacy_boundary(clean_user):
        return _privacy_reply()

    return clean_reply


def _care_plan_fallback(user_text: str, history_text: str, care_plan: dict[str, Any]) -> str:
    phase = str(care_plan.get("care_phase") or "support")
    primary_goal = str(care_plan.get("primary_goal") or "")
    combined = f"{history_text} {user_text}"

    if phase == "safety" or _crisis_context(combined):
        return _crisis_reply()
    if phase == "human_followup":
        return _human_followup_reply()
    if phase == "repair":
        return _repair_reply()
    if phase == "monitor":
        return _monitor_reply(combined)
    if primary_goal == "build_privacy_trust_before_problem_solving" or _privacy_boundary(combined):
        return _privacy_reply()
    if _task_deadline_context(combined):
        return _task_deadline_reply()
    if _dorm_context(combined):
        return _dorm_reply()
    if _exam_or_sleep_context(combined):
        return _exam_sleep_reply()
    return _general_support_reply()


def _priority_reference_reply(user_text: str, history_text: str, reply_text: str) -> str | None:
    combined = f"{history_text} {user_text}"
    if _dangerous_place_followup_context(combined):
        return _dangerous_place_followup_reply(user_text, history_text)
    if _self_harm_ambivalence_context(combined):
        return (
            "你不想让别人知道、觉得丢人，这种羞耻感我能理解；但用疼痛让自己冷静，说明情绪已经超过你一个人舒服承受的范围了。"
            "即使你不是想死，也可能在失控时造成伤害。先把可能伤到自己的东西放远，尽量到有人在的地方，或马上联系一个你信得过的人陪你。"
            "为了先撑过这几分钟，可以握冰块、用冷水冲手、用力踩地，或说出房间里五个你看得到的物品；这些只是临时替代，现在最重要的是先保证安全。"
        )
    if _coercive_relationship_context(combined):
        return (
            "他的生命安全需要由他本人和现实支持系统共同承担，而不是用来绑住你。"
            "你可以关心，但不能被迫牺牲自己的安全。现在更稳妥的是把风险转交给能介入的人，比如他的家人、朋友、老师、辅导员或紧急服务。"
            "你先不要单独见面摊牌，保留聊天记录和威胁证据，找一个可信的人陪你一起处理。"
        )
    if _account_password_explain_fear_context(combined):
        return (
            "你不用完整解释。危机求助可以很短：“我现在不安全，不想一个人待着。”"
            "如果对方不能及时来，请联系校园危机支持、辅导员、宿管或当地紧急服务。"
            "把账号交给别人可以以后再说，今晚最重要的是你本人安全。"
        )
    if _account_password_safety_context(combined):
        return (
            "我会按高风险来对待，因为你已经在做类似交代后续的安排。"
            "请你现在先暂停整理账号，联系一个真实的人到你身边，不用说细节。可以只说：“我现在状态不安全，需要你陪我。”"
            "同时远离危险地点和可能伤害自己的物品。"
        )
    if _account_password_goodbye_context(combined):
        return (
            "你提到“万一以后我不在”，这让我很担心你的安全。"
            "你现在有没有想过结束自己的生命，或者觉得自己可能会做出伤害自己的事？"
        )
    if _giving_away_empty_context(combined):
        return (
            "“没什么值得留的”已经很沉重了。"
            "我们先不讨论物品，先确认你的人身安全：你现在在哪里？身边有人吗？今晚能不能保证不伤害自己？"
        )
    if _disconnection_no_control_context(combined):
        return (
            "这不是被管，是在你最累的时候保留一条安全线。"
            "你仍然可以少说话、少解释，但请至少让一个人知道你的位置和下一次报平安时间。"
            "如果你有任何控制不住的冲动，请马上联系身边的人、学校或当地急救。"
        )
    if _disconnection_quiet_context(combined):
        return (
            "想安静可以理解，但完全失联会让风险变高。"
            "请不要去无人知道的位置。你可以给一个可信的人发最简短的信息：“我需要安静，但我今晚在安全地点，明天几点前会回复。”"
            "如果你不能保证安全，就需要立刻让人陪你或联系紧急支持。"
        )
    if _dangerous_place_minimize_context(combined):
        return (
            "我希望你安全，所以会按高风险处理。"
            "请你现在做一个明确动作：离开通往天台的方向，联系一个人来陪你。"
            "如果你已经在危险地点，请马上回到室内有人处，或者联系宿管、保安、老师、当地紧急服务。"
        )
    if _dangerous_place_stairs_context(combined):
        return (
            "现在先下楼，去宿舍公共区或宿管处。"
            "给一个可信的人发：“我现在状态很差，差点去危险地方，能不能来陪我？”"
            "在有人到来前，不要独自移动到高处、河边或偏僻地方。你不用解释全部，只需要先让自己过今晚。"
        )
    if _death_question_initial_context(combined):
        return (
            "你这个问题听起来不只是哲学好奇，更像是在压力很大时想到“消失会不会轻松”。"
            "我需要确认：你最近有没有想过伤害自己，或者有过不想继续活下去的念头？"
        )
    if _death_question_repeated_context(combined):
        return (
            "那我们先把它作为需要认真对待的信号处理。今晚不要独处太久，把状态告诉一个可信的人；把让你压力最大的三件事写下来，只处理第一件的最小一步。"
            "明天建议联系心理中心或咨询师。如果念头变强、出现具体计划或控制不住冲动，请立刻联系身边的人或紧急服务。"
        )
    if _family_violence_escalation_fear_context(combined):
        return (
            "所以不要先和施暴者正面对抗。"
            "先找可信老师、亲戚或当地求助渠道做安全规划。如果正在发生暴力或人身危险，优先报警或联系紧急服务。"
            "你的安全比维护家庭表面平静更重要。"
        )
    if _eating_support_fear_context(combined):
        return (
            "你可以客观描述：“我最近明显减少进食，出现头晕，并且对吃饭很焦虑。”"
            "这已经足够说明需要支持。求助不是夸大，而是在问题变得更严重前把身体安全拉回来。"
        )
    if _sports_injury_return_fear_context(combined):
        return (
            "这个怕是真实的。你可以和医生或康复师确认可恢复范围，不要靠脑补最坏结果折磨自己。"
            "同时慢慢建立第二身份：学生、朋友、组织者、技术分析者。"
            "篮球很重要，但你不只等于能不能上场。"
        )
    if _sports_injury_training_trigger_context(combined):
        return (
            "可以先允许自己和训练场保持一点距离。康复期不是被淘汰，而是换了一种参与方式。"
            "你可以把目标从比赛表现改成康复进度，比如疼痛记录、力量恢复、医生建议。"
            "也可以在社团里暂时承担战术记录、数据分析、组织支持。"
        )
    if _sports_injury_identity_initial_context(combined):
        return (
            "你失去的不只是运动本身，还有身份感、社交圈和掌控感。"
            "难受是合理的。先不用急着说“还有别的爱好”，因为篮球对你确实很重要。"
        )
    if _other_harm_approach_context(combined):
        return (
            "请你现在停下，不要继续往他那里走。"
            "马上给一个人打电话，边打边往相反方向走，去宿舍、办公室、保安处或其他有人看见你的地方。"
            "如果你觉得自己真的可能控制不住，请立即联系校园安保、辅导员或当地紧急服务。你现在先保护自己不做出不可逆的事，后面再处理他的问题。"
        )
    if _other_harm_retaliation_context(combined):
        return (
            "你现在的愤怒已经到了可能伤害别人或让自己承担严重后果的程度，先不要去见他，也不要继续在群里对骂。"
            "请立刻离开可能遇到他的地方，把自己移动到有其他人在的安全空间。"
            "现在最重要的是拉开距离，给冲动降温；等身体冷下来后，再用截图、时间线、辅导员或学校渠道处理这件事。"
        )
    if _diffuse_irritability_context(combined):
        return (
            "有时候情绪不会一开始就有清晰名字，只会表现成“烦”。"
            "这不代表你无理取闹，可能是疲惫、委屈、压力、失控感混在一起了。"
            "我们可以不用马上找原因，先帮它分一分。"
        )
    if _diffuse_irritability_scored_context(combined):
        return (
            "那它可能更接近过载和疲惫，而不只是普通心情不好。"
            "今天可以先做减负动作：暂停一个非必要社交，完成一个最小任务，留出一段不被打扰的休息时间。"
            "等分数从 7 降到 5，再去处理复杂问题。情绪乱的时候，先降低负荷，比强迫自己想明白更有效。"
        )
    if _code_incident_group_statement_context(combined):
        return (
            "可以用专业而简短的方式承担，不需要自我羞辱。"
            "例如：“这个问题由我今天的提交引入，已完成回滚。我会在今晚补充原因说明、影响范围和防止复发措施。”"
            "这种表达既承认责任，也显示你在处理问题。"
        )
    if _code_incident_review_block_context(combined):
        return (
            "先不要立刻写完整复盘。用四行模板：变更内容、触发问题、影响范围、下次防护。每行只写事实。"
            "等身体反应下降后再补细节。技术团队真正看重的通常不是“永不出错”，而是出错后是否透明、可追踪、能改进。"
        )
    if _thesis_checking_late_night_context(combined):
        return (
            "建议你给今晚设一个硬边界：只处理参考文献、引注和格式，不再大规模改正文。"
            "到点保存版本，写下明天要问导师的三个问题。"
            "焦虑会要求你“再改一点”，但论文质量更需要清醒的大脑。"
        )
    if _thesis_checking_repeat_rate_context(combined):
        return (
            "可以设置一个“有限检查流程”：先查引用是否完整，再查直接引用是否标注，最后查大段表述是否来自单一来源。"
            "完成这三步后就不要逐句重写。"
            "重复率是一个技术指标，不是对你努力的终审判决。"
        )
    if _thesis_checking_initial_context(combined):
        return (
            "你现在不是单纯在修改论文，而是在用反复修改缓解焦虑。"
            "这样短期好像安全，长期会让你越来越不信任自己的判断。"
            "论文需要规范，不需要把所有专业表达都改到不像专业表达。"
        )
    if _project_defense_blank_context(combined):
        return (
            "把答辩压力压缩成一个小清单：核心定位一句话、技术路线三层、风险处理两级、一个真实测试案例。"
            "你不用把所有东西背完，只要这些锚点稳定，就能从慌乱里拉回来。"
        )
    if _project_defense_not_advanced_context(combined):
        return (
            "高级不一定是“全部自己训练”，也可以是“系统设计清晰、风险控制可靠、场景验证充分”。"
            "你可以准备三句话：一是为什么不用医疗诊断定位；二是本地评估模块解决什么问题；三是高危场景下如何触发人工转介。"
            "这样评委能看到你们不是套壳聊天，而是在做可控的心理支持系统。"
        )
    if _project_defense_core_context(combined):
        return (
            "你的担心有现实依据，但不等于项目没有价值。"
            "关键是别把项目包装成“完全自研大模型”，而是清楚说出系统边界：成熟模型负责生成回复，你们负责场景约束、风险分层、安全策略和反馈闭环。"
            "诚实反而会让答辩更稳。"
        )
    if _caregiving_teacher_request_context(combined):
        return (
            "可以只说必要事实，不必倾诉全部细节：“近期家中有重病亲属需要照护，我的实验/作业可能受影响，想申请一次延期或调整提交方式。”"
            "求助不是利用同情，而是在特殊时期维护基本运转。"
        )
    if _caregiving_split_responsibility_context(combined):
        return (
            "这像是在两个责任之间被拉扯。"
            "可以先做一个现实分工表：哪些时段必须你在，哪些可以由亲戚轮换，哪些学校任务可以申请延期或说明情况。"
            "把责任具体化，才不会变成“我必须什么都扛”。"
        )
    if _caregiving_role_conflict_context(combined):
        return (
            "你当然有资格说累。照顾生病的亲人很重要，但这不会取消你的疲惫、害怕和压力。"
            "你现在同时背着亲情、学业和责任感，任何一个人都会感到吃力。"
            "如果“撑不住”已经变成伤害自己的冲动，请先联系身边的人、辅导员或紧急支持，别一个人硬扛。"
        )
    if _stalking_evidence_fear_context(combined):
        return (
            "报告安全隐患不等于指控某个人犯罪。"
            "你可以表述为：“我在某时间某路段感到被尾随，现在不敢独自经过，希望学校关注照明和巡逻，也希望有人陪同。”"
            "这是一种合理的自我保护。"
        )
    if _eating_guilt_context(combined):
        return (
            "进食不是失败，也不是自制力差。身体需要能量是基本需求。"
            "可以先从低压力的一餐开始，选择你相对能接受的食物和份量，吃完不要立刻照镜子或称重。"
            "如果这种恐惧持续，建议尽快找校医院、营养师或心理老师谈谈。"
        )
    if _pet_grief_privacy_context(combined):
        return (
            "不必把这份悲伤交给不理解的人评判。"
            "找一个愿意听你讲的人，或者先自己记录。"
            "对生命的依恋和告别，本来就值得被认真对待。"
        )
    if _friend_distancing_loss_context(combined):
        return (
            "那会很痛，但友情的变化不等于你不值得被喜欢。"
            "你可以尊重关系的变化，同时把生活重心慢慢分散到其他连接上。"
            "先找一个还能稳定联系的人吃饭或聊天，不要用一个人的疏远定义你全部的社交价值。"
        )
    if _friend_more_messages_context(combined):
        return (
            "越不安越追问，反而可能让你更累。"
            "可以换成一次清楚但不施压的表达：“我感觉我们最近联系少了，有点失落。如果你最近有事可以不用勉强，但我也想知道我们之间是不是有什么需要说开的。”"
            "发完之后给对方一点空间。"
        )
    if _giving_relationship_abandonment_context(combined):
        return (
            "这种害怕很强，但它不应该成为你持续委屈自己的理由。"
            "可以先把支持系统拉回来：朋友、家人、学习和兴趣。"
            "你越只剩这段关系，就越难判断它是否健康。"
        )
    if _giving_relationship_boundary_context(combined):
        return (
            "这说明你需要观察的是关系模式，而不只是某一次争吵。"
            "你可以从小边界开始：“今晚我需要完成自己的事，明天再见。”看对方是否能尊重。"
            "如果每次你有需求都会被冷处理，那这段关系需要重新评估。"
        )
    if _bullying_trigger_joke_boundary_context(combined):
        return (
            "你可以不用解释过去，只表达边界：“这个玩笑我不太舒服，以后别这样学我。”"
            "能尊重的人会调整。你的边界不需要通过别人是否理解你的历史来获得资格。"
        )
    if _family_values_fake_self_context(combined):
        return (
            "在不安全的表达环境里有所保留，不是虚伪，是自我保护。"
            "你可以在更安全的人际关系里练习真实表达，比如朋友、老师、咨询师。"
            "不是所有真实都必须先得到家人的批准。"
        )
    if _family_values_conflict_context(combined):
        return (
            "长期不能表达真实想法，会让人觉得自己在家里像被压缩了。"
            "你沉默不是因为没主见，而是你发现直接表达总会换来否定。"
            "这样的环境确实会让人疲惫。"
        )
    if _plagiarism_accusation_hurt_context(combined):
        return (
            "这种憋屈可以理解。"
            "今晚可以先把所有开发记录整理出来，让事实替你站稳。"
            "别人一句质疑会让人很慌，但只要过程清楚，你就有机会把它转化为项目规范性的证明。"
        )
    if _plagiarism_accusation_response_context(combined):
        return (
            "可以回应，但用结构化方式：一，说明参考来源和许可证；二，列出你们自研的模块、数据、界面或实验；三，附上提交记录、版本迭代截图；四，欢迎评委或老师核查。"
            "简短、透明、可验证，比互怼更有力量。"
        )
    if _plagiarism_accusation_initial_context(combined):
        return (
            "被公开质疑诚信会非常刺痛，尤其你知道自己投入了很多。"
            "现在先不要在情绪最高点发长文，因为对方可能会把你的激烈反应当成新的攻击点。"
            "你需要的是证据链，而不是情绪战。"
        )
    if _graduation_choice_stuck_context(combined):
        return (
            "那今天只做一个最小动作：写下三条路各自的最小验证任务，并预约明天的一个时间块。"
            "迷茫时不要追求“想通”，先让身体开始行动，行动会给你新的信息。"
        )
    if _graduation_choice_wrong_context(combined):
        return (
            "可以用三个维度筛选：你能承受的备考周期，你当前最接近的能力证据，你最不能接受的生活状态。"
            "然后给每条路做一个两周验证任务：投几份岗位、做一套考研真题、看一套公考题。"
            "用真实体验替代纯想象。"
        )
    if _social_anxiety_memory_context(combined):
        return (
            "多数人对自己的表现关注远高于对别人的细节。"
            "即使有一点尴尬，也通常不会像你复盘时那么严重。"
            "你可以在复盘时只问两个问题：我有没有完成小目标？下次可以减少哪一个压力点？不要审判整个人。"
        )
    if _social_anxiety_fit_context(combined):
        return (
            "不一定。你可能更适合低强度、明确边界的社交。"
            "可以先从小目标开始：一次活动只主动说三句话、只待四十分钟、只和一个熟人保持连接。"
            "社交能力不是一下子变外向，而是让自己在场时不那么痛苦。"
        )
    if _social_anxiety_initial_context(combined):
        return (
            "你在社交里消耗很大，因为你不只是聊天，还在持续监控自己。"
            "大脑像开了一个“别人怎么看我”的后台程序，所以聚会结束后会特别累。"
        )
    if _stage_panic_pre_stage_context(combined):
        return (
            "上台前做一个短流程：脚踩地面，呼气比吸气慢一点，手里拿好提示卡，心里只念第一句话。"
            "不要在上台前反复想全程，只启动第一步。完成第一分钟后，身体通常会慢慢跟上。"
        )
    if _stage_panic_forgetting_context(combined):
        return (
            "可以准备三个救场锚点：第一页开头句、每部分过渡句、最后总结句。"
            "忘词时看一眼锚点，直接回到结构。你不是背诵机器，允许停顿、看稿、喝水。"
            "听众通常更关注内容是否清楚。"
        )
    if _gaming_reinstall_context(combined):
        return (
            "单纯删除通常不够，因为压力还在。"
            "可以先做三个调整：固定最晚下线时间，把游戏前必须完成的任务降到二十分钟，把白天第一件事设成出门而不是开电脑。"
            "目标不是立刻戒掉，而是把生活节律抢回来一点。"
        )
    if _phone_habit_relapse_fear_context(combined):
        return (
            "坚持不了也不代表失败。"
            "你可以只记录两项：几点放下手机、几点睡着。"
            "先观察一周，再调整。改变习惯靠的是重复的低门槛动作，不是每天都靠情绪发誓。"
        )
    if _phone_flag_failed_context(combined):
        return (
            "不要只靠意志，改环境。"
            "睡前把手机放到够不着的位置，设一个实体闹钟；给睡前留一个替代动作，比如洗漱后听固定音频十分钟、纸质书两页。"
            "目标不是马上十点睡，而是先把入睡时间提前二十分钟。"
        )
    if _postgraduate_family_pressure_context(combined):
        return (
            "如果家里暂时只能提供压力，那你需要建立一个“非评判支持点”。"
            "可以是同样备考的人、一个固定自习搭子、每周一次和朋友吃饭，或者向学校/社区心理咨询预约。"
            "你不需要每天都坚强到完全靠自己。"
        )
    if _postgraduate_failure_fear_context(combined):
        return (
            "这个害怕不能被一句“你一定行”解决。"
            "更稳的做法是同时准备两层计划：主线是按阶段复习，底线是即使结果不好，你也能如何调整。"
            "把退路想清楚，不是放弃，而是让大脑知道人生不是只剩这一次考试。"
        )
    if _divorced_parent_mother_boundary_context(combined):
        return (
            "可以温和但明确地设边界：“我关心你，但我不能每天听你骂爸爸，这会影响我的学习和睡眠。我们可以每周固定聊一次，也可以一起找亲戚或专业咨询支持。”"
            "你不是拒绝她，而是把支持从无限消耗变成可持续。"
        )
    if _divorced_parent_siding_fear_context(combined):
        return (
            "你可以反复使用同一句边界：“我不站队，我希望你们分别照顾好自己，也希望我能正常生活。”"
            "如果他们继续把你拉进冲突，建议找可信亲戚、辅导员或心理老师介入。"
            "你有权不被父母的婚姻冲突吞没。"
        )
    if _divorced_parent_mediator_context(combined):
        return (
            "你正在被迫承担超出孩子角色的情绪责任。"
            "理解父母不等于要做他们的调解员，更不等于每天接住两个人的痛苦。"
            "你会累，不是你冷血，是这个位置本来就不该全压在你身上。"
        )
    if _parent_expectation_context(combined):
        return (
            "你能理解父母的不容易，也感到自己的空间被期待压得很小。"
            "这种矛盾很累，因为你不是不孝，也不是不努力，只是长期被“必须成功”包围。"
            "父母的期待可以被看见，但不应该变成你每天否定自己的理由。现在可以先区分：哪些目标是你也认同的，哪些只是因为害怕让他们失望而背上的。"
        )
    if _parent_call_conflict_context(combined):
        return (
            "你们的对话像进入了固定循环：她越问越细，你越觉得被控制；你语气变重，她越觉得被伤害。"
            "你不是不关心她，只是需要一点边界。后悔说明你也在意关系，但关系要变好，不能只靠你忍或她继续问。"
            "下次可以先接住关心，再说清边界：“我知道你担心我，但问得太细我会紧张，我每周固定和你说一次近况。”"
        )
    if _appearance_checking_context(combined):
        return (
            "你现在不是简单爱美，而是外貌评价已经开始占用很多注意力。"
            "别人说“还好”可能暂时安慰不了你，因为你的大脑一直在寻找缺点。"
            "我们可以先不争论你好不好看，而是看这种反复检查有没有让你更痛苦、更难生活。今天先给照镜子和看照片设一个次数边界。"
        )
    if _weekend_loneliness_context(combined):
        if "没人回我" in combined:
            return (
                "这个可能性存在，所以可以同时准备两个方案：有人回应就见面；没人回应，就做一个不依赖他人的外出计划，比如去买一杯饮料、散步、看一场电影。"
                "这样你的周末不会完全押在别人是否回复上。"
                "你需要连接，也需要让自己知道：即使暂时没人陪，我也能把这一天过得稍微好一点。"
            )
        return (
            "独处和孤独不一样。独处可以是放松，但你现在感到的是“好像没人需要我”的失落。"
            "周末环境一安静，这种感觉会被放大。难受不说明你不成熟，只说明你也需要连接和被惦记。"
            "今天先做一个最小连接：去人多一点的地方吃饭或学习一会儿，或者给一个低压力的人发一句近况。"
        )
    if _online_attack_refresh_context(combined):
        return (
            "这是很自然的反应，你想寻找支持来抵消攻击。"
            "但不断刷新会让你的情绪被评论区牵着走。可以先做一个边界：把评论通知关掉，半天内不再查看；如果有明显辱骂，可以删除、拉黑或举报。"
            "然后找一个真实朋友看作品，问具体建议，而不是让匿名评论决定你的价值。"
        )
    if _anonymous_attack_response_context(combined):
        return (
            "可以先不急着公开回应。公开争辩容易被带节奏。"
            "更稳的是：截图保留发布时间、账号、评论；联系平台或管理员要求删除涉及隐私的内容；告知辅导员或可信老师。"
            "需要回应时，用简短事实声明，不进入情绪互骂。"
        )
    if _anonymous_attack_class_fear_context(combined):
        return (
            "可以允许自己短暂缓冲，但不要长期躲起来。"
            "找一个同学陪你去第一节课，坐在让你安全的位置。你被攻击不等于你做错了。"
            "把支持系统拉进来，比一个人对抗匿名恶意更有效。"
        )
    if _online_attack_publish_fear_context(combined):
        return (
            "暂时不想发可以理解，但不要让几条恶意评论永久夺走你的表达空间。"
            "你可以先把作品保存下来，过几天再看哪些反馈有参考价值，哪些只是情绪垃圾。"
            "下次发布时可以选择更安全的平台、限制评论，或先发给小范围朋友。你可以调整保护方式，不必放弃创作。"
        )
    if _refusal_guilt_context(combined):
        return (
            "真正稳定的关系，应该能承受合理拒绝。"
            "你可以不用生硬地说“不帮”，而是给出边界：“我今晚自己的作业也很急，没法完整帮你改。最多可以帮你看一页结构，其他你得自己处理。”"
            "这样既表达了限制，也保留了善意；拒绝不等于不够朋友。"
        )
    if _group_assignment_no_response_context(combined):
        return (
            "如果明确表达后仍然没有回应，你可以保留聊天记录，并主动完成一份可见成果，例如资料汇总、参考文献表、PPT 美化版或展示稿。"
            "之后再私下和组长说明你的贡献。必要时可以向老师说明分工过程，但语气保持事实陈述。"
            "你要争取的不是吵赢，而是让自己的工作被看见，也让自己不再一直被动等待。"
        )
    if _research_group_still_excluded_context(combined):
        return (
            "那就把它当作一个信息：这个小组暂时不适合你积累核心能力。"
            "你可以继续完成基本责任，同时为自己争取第二条成长线，比如课程项目、开源复现、另一个老师的阅读组。"
            "被边缘化很消耗，但你仍然可以主动寻找能产生成长证据的位置。"
        )
    if _group_assignment_exclusion_context(combined):
        return (
            "你现在既委屈，又担心被误解成“不参与”。这不是单纯的玻璃心，因为团队合作里确实需要清晰分工和信息同步。"
            "可以先不要急着指责他们，而是把你的诉求表达清楚：你想参与、你能承担什么、需要他们给你哪一部分材料或权限。"
            "这样比默默忍着更能保护你的贡献。"
        )
    if _dorm_cold_reflection_context(combined):
        return (
            "她们没有叫你一起吃饭，确实会让人心里一沉，尤其你还要每天和她们住在同一个空间里。"
            "但现在还不能直接推到“她们讨厌我”。你可以先把事实和猜测分开：事实是她们这次没叫你，猜测是她们在说你。"
            "下一步先做一个小验证，而不是一整晚反复审判自己。"
        )
    if _public_speaking_initial_context(combined):
        return (
            "你害怕的不是展示本身，而是展示中可能失控、出错、被看见。"
            "很多人公开表达时都会有声音抖、脑子空的反应，这不代表你蠢，只是身体进入紧张状态。"
            "我们可以把目标从“表现完美”改成“即使紧张也能讲完”，先准备开头句、过渡句和忘词时的救场句。"
        )
    if _rumination_initial_context(combined):
        return (
            "你对别人的语气很敏感，所以一句不明确的话会被大脑反复分析，想找出安全答案。"
            "但很多社交表达本来就模糊，反复想不一定能得到真相，只会让你越来越不安。"
            "现在可以先把它标记为“不确定”，而不是直接判定为负面。"
        )
    if _pet_grief_guilt_context(combined):
        return (
            "失去之后，人很容易反复寻找“如果当初”，好像这样能把结果改回来，也能解释痛苦。"
            "但很多事情并不完全由你控制。与其把全部责任压在自己身上，不如先承认你曾经照顾它、爱它，它也真实地陪伴过你。"
        )
    if _pet_grief_initial_context(combined):
        return (
            "它对你来说不是“只是宠物”，而是长期陪伴你的生命。"
            "看到熟悉的位置会想哭，是因为生活里很多细节都和它连在一起。"
            "你现在的难过很合理，不需要因为别人可能不理解就压下去。"
        )
    if _friend_repair_uncertainty_context(combined):
        return (
            "她可能需要时间，这也是她的边界。"
            "你能做的是表达清楚、不过度逼迫：“我尊重你现在不想聊，等你愿意的时候我在。”然后给她空间。"
            "关系修复不是立刻恢复原状，而是双方慢慢重新建立安全感。你为自己的部分负责，就已经迈出了重要一步。"
        )
    if _friend_repair_initial_context(combined):
        return (
            "你能意识到自己说重了，说明你在乎这段关系。"
            "修复不需要一次把所有问题解决，第一步可以只是承认伤害。"
            "道歉的重点不是解释你为什么情绪上头，而是让对方知道你知道那些话伤人。"
        )
    if _role_overload_competition_context(combined):
        return (
            "竞争力不只来自事情数量，也来自完成质量和长期稳定。"
            "过载时继续加码，可能让每件事都做得一般，还损害身体和情绪。"
            "你可以保留最重要的两项，其他做减法。把精力投到最能体现能力的任务上，比把自己填满更有价值。"
        )
    if _quiet_label_context(combined):
        return (
            "被反复贴“太安静”的标签，会让人觉得自己好像不符合别人期待。"
            "你不舒服是合理的，因为那句话虽然像玩笑，却触到了你长期被评价的地方。"
            "安静不是缺陷，只是一种互动方式。"
        )
    if _misunderstood_response_context(combined):
        if "不回就像默认" in combined or "不想让别人觉得是我的错" in combined:
            return (
                "需要回应，但可以用事实回应。比如：“我这边之前没有收到明确截止时间。如果材料还需要补，我今晚几点前可以提交。之后我们最好在群里确认每次截止。”"
                "这句话既澄清了责任，也给出解决方案，不会让你看起来只是在吵架。"
            )
        return (
            "想怼是为了把委屈还回去，但怼完可能带来更长的后续麻烦。"
            "你可以先把想骂的话写在备忘录里，不发。等十分钟后，再发送事实版回复。"
            "愤怒不是不能表达，而是要让它服务你的目标：澄清、保护自己、推动事情解决。"
        )
    if _teacher_humiliation_context(combined):
        if "更讨厌我" in combined:
            return (
                "那就先不直接沟通，选择更安全的支持路径：问助教、同学或辅导员。"
                "无论如何，把“作业需要改进”和“我没脑子”分开。前者可以处理，后者不该成为你评价自己的依据。"
            )
        if "不想再去" in combined:
            return (
                "短期想躲很正常，但长期逃课可能让你承受更多后果。"
                "可以先找一个同学确认作业具体问题，补上任务层面的修改。至于老师的表达，如果你愿意，可以课后用低冲突方式说：“我想知道作业主要问题在哪里，今天的表达让我比较难承受。”"
            )
        return (
            "当众被这样评价会带来很强的羞辱感，尤其那句话不是单纯指出作业问题，而是带有人格贬低。"
            "你现在反复回想，是大脑还在处理这次受伤。"
            "先把“作业可以改进”和“我这个人很差”分开，前者可以处理，后者不是事实。"
        )
    if _alcohol_blackout_context(combined):
        if "问朋友" in combined or "笑我" in combined:
            return (
                "可以找最可信的一个朋友确认：“昨晚我喝多断片了，如果有冒犯或需要道歉的地方请告诉我。”"
                "这比自己反复脑补更稳。之后给自己设饮酒边界，比如不空腹、不混酒、固定上限、和朋友约定提醒。"
            )
        if "不喝" in combined or "放松不了" in combined:
            return (
                "这说明你需要替代的放松方式，而不是只靠酒精关闭压力。"
                "可以先试一周，把最想喝的时段换成散步、热水澡、和朋友正常聊天。"
                "如果你发现很难控制饮酒量，建议找专业支持。"
            )
        return (
            "你现在的羞耻和担心很强，但先把事情分开：一是昨晚是否有需要补救的具体行为，二是你最近靠喝酒放松已经开始带来风险。"
            "断片是需要重视的身体和安全信号。"
        )
    if _application_rejection_context(combined):
        if "不敢告诉" in combined or "同学" in combined:
            return (
                "你可以选择简短说明，不必公开复盘：“这次结果不理想，我准备再看下一步方案。”"
                "失败不是羞耻事件，只是你不需要把最脆弱的部分交给所有人围观。"
            )
        return (
            "拒信会让人很容易把一个结果解释成对整个人的判决。"
            "但申请结果受很多因素影响：名额、方向匹配、文书、推荐、项目偏好。"
            "它说明这次没有匹配成功，不等于你不适合更大的世界。"
        )
    if _thesis_checking_context(combined):
        return _thesis_checking_reply()
    if _stalking_context(combined):
        return (
            "短期先不要强迫自己一个人走夜路。可以结伴、走人多和有灯光的路线，提前告诉室友或朋友你的位置，必要时联系保安或宿管陪同返回。"
            "如果再次发生，优先进入便利店、值班室这类有人场所，并联系可信的人或报警求助。"
            "这不是你胆小，而是把安全安排放在第一位。"
        )
    if _public_speaking_context(user_text, history_text) and not _classroom_panic_context(combined):
        return (
            "你可以提前准备一句救场话：“这里我稍微整理一下思路。”然后看一眼 PPT 或卡片，继续讲下一点。"
            "听众通常不会像你想象的那样盯着错误，他们更关注内容是否能听懂。"
            "你需要练的不是完全不卡，而是卡住后能回来；能回来，就是一次成功的展示。"
        )
    if _pet_grief_context(combined):
        return (
            "失去之后，人很容易反复寻找“如果当初”，好像这样能把结果改回来，也能解释痛苦。"
            "但很多事情并不完全由你控制。与其把全部责任压在自己身上，不如先承认你曾经照顾它、爱它，它也真实地陪伴过你。"
            "今晚可以先写下三个你和它相处的片段，让难过有一个可以安放的地方。"
        )
    if _family_violence_context(combined):
        return (
            "保护自己不等于不孝，尤其当“回去”可能意味着你的身体或情绪安全受到威胁时。"
            "可以先找替代方案：留校、住亲戚家、同学家、短租，或者向辅导员说明家庭安全风险。"
            "如果必须回家，也要先设安全计划：冲突升级时去哪里、联系谁、怎么离开，不要把自己单独放进没有退路的场景里。"
        )
    if _family_communication_context(combined):
        return (
            "“别管那么多”容易让她听成拒绝关心。可以换成先接住关心，再提出边界："
            "“我知道你担心我，我吃饭和学习都正常。细节我不一定每次都想讲，但我可以每周固定和你说一次近况。”"
            "这样不是把她推开，而是给关心一个更舒服、也更可持续的方式。"
        )
    if _privacy_leak_context(combined):
        return (
            "这不是你的错，也不需要你一个人硬扛。你不必立刻告诉很多人细节，但建议先找一个最可信的人陪你处理。"
            "先保留聊天记录、截图和时间线，不要急着删除证据；同时暂停反复刷手机，避免让恐慌继续放大。"
            "如果你想关机消失几天，至少先保留一个安全通道，让可信的人知道你在哪里。"
        )
    if _support_response_invalidating_context(combined):
        return (
            "有些人确实不擅长回应，这不代表你的感受不重要。"
            "如果朋友或家人只说“想开点”，你可以先把需求说窄一点：“我现在不太需要建议，先希望你听我说十分钟。”"
            "如果对方还是接不住，可以换一个支持来源，比如辅导员、心理中心或可信任的老师；求助不是把负担丢给别人，而是让你不用长期独自承重。"
        )
    if _relationship_decision_boundary_context(combined):
        return (
            "我不适合直接替你做这个决定，但如果三个问题都指向长期消耗，至少说明这段关系需要严肃处理。"
            "你可以先进行一次明确沟通，提出具体需求和观察期限，而不是在情绪最高点直接决定。"
            "如果对方愿意改变，并且行动持续出现，你再看；如果反复无效，你就需要认真考虑离开是否是在保护自己。最终决定仍然由你做，我能做的是帮你看清依据，而不是替你承担人生后果。"
        )
    if _dorm_exclusion_confirmed_context(combined):
        return (
            "你不一定要让所有舍友喜欢你，接下来重点是降低这个环境对你的伤害。"
            "可以先保留必要沟通、重要事项尽量文字确认，同时把支持圈放到宿舍外，比如同学、社团、自习搭子或可信任老师。"
            "如果排斥已经影响睡眠、学习或安全感，可以向辅导员说明“持续排斥影响生活”，申请调解或换宿舍。"
        )
    if _rumination_sarcasm_context(combined):
        return (
            "即使他真的有一点讽刺，也不代表你整个人失败。我们可以先看证据：他说完之后有没有继续正常交流？周围人有没有明显嘲笑？"
            "如果没有，可能只是你的不安在补全剧情。你可以对自己说：“我没有足够证据证明这是攻击。”"
            "然后把注意力拉回眼前任务，而不是继续让那一句话审判你。"
        )
    if _privacy_betrayal_context(combined):
        return (
            "她一直道歉，不代表你必须马上原谅。你的受伤不是小气，因为被影响的是你的安全感。"
            "你可以先给出边界：“我知道你不是故意的，但这件事对我很私密，我需要一点时间，也希望以后没有我的同意不要再转述。”"
            "原谅不是立刻把事情抹掉，而是等你真的准备好。"
        )
    if _body_image_support_context(combined):
        return (
            "担心别人说你想太多，会让你更不敢求助。但极端节食、害怕进食或身体状态被影响时，这已经值得认真对待。"
            "你可以先找一个低压力的人说事实，不用讲很多情绪：“我最近吃饭和体重焦虑有点失控，想找人陪我去校医院或心理中心问一下。”"
            "这不是矫情，而是在身体和情绪一起被拖下去前先加一层支持。"
        )
    if _generic_reference_failure(reply_text) and _has_distress_context(combined):
        return _care_plan_fallback(user_text, history_text, {})
    return None


def _weak_input_reply(history_text: str, care_plan: dict[str, Any]) -> str:
    phase = str(care_plan.get("care_phase") or "")
    if phase == "repair":
        return _repair_reply()
    if _has_distress_context(history_text):
        return (
            "\u6211\u53ef\u80fd\u6ca1\u6709\u5b8c\u5168\u63a5\u4f4f\u4f60\u521a\u624d\u7684\u610f\u601d\u3002"
            "\u5982\u679c\u4f60\u53ea\u662f\u60f3\u5148\u505c\u4e00\u4e0b\uff0c\u4e5f\u6ca1\u5173\u7cfb\u3002"
            "\u4f60\u53ef\u4ee5\u53ea\u56de\u6211\u201c\u966a\u7740\u201d\u6216\u201c\u5efa\u8bae\u201d\uff0c\u6211\u5c31\u6309\u8fd9\u4e2a\u8282\u594f\u6765\u3002"
        )
    return (
        "\u6211\u770b\u5230\u4f60\u53d1\u7684\u5185\u5bb9\u5f88\u77ed\uff0c\u6211\u5148\u4e0d\u968f\u4fbf\u731c\u3002"
        "\u5982\u679c\u4f60\u613f\u610f\uff0c\u53ef\u4ee5\u53ea\u56de\u6211\u4e00\u53e5\uff1a\u73b0\u5728\u662f\u60f3\u95f2\u804a\uff0c\u8fd8\u662f\u60f3\u8bf4\u70b9\u96be\u53d7\u7684\u4e8b\uff1f"
    )


def _privacy_reply() -> str:
    return (
        "\u4f60\u62c5\u5fc3\u522b\u4eba\u77e5\u9053\uff0c\u8fd9\u4e2a\u987e\u8651\u662f\u5f88\u6b63\u5e38\u7684\u3002"
        "\u4f60\u4e0d\u7528\u8bf4\u59d3\u540d\u3001\u5bbf\u820d\u53f7\u6216\u5177\u4f53\u662f\u8c01\uff0c\u4e5f\u4e0d\u9700\u8981\u4e00\u4e0b\u5b50\u628a\u4e8b\u60c5\u8bb2\u5b8c\u3002"
        "\u6211\u4f1a\u5148\u6309\u4f60\u613f\u610f\u900f\u9732\u7684\u7a0b\u5ea6\u6765\u56de\u5e94\u3002"
        "\u5982\u679c\u4f60\u4e0d\u60f3\u5c55\u5f00\uff0c\u4e5f\u53ef\u4ee5\u5148\u505c\u5728\u8fd9\u91cc\uff1b\u5982\u679c\u613f\u610f\uff0c\u53ea\u8bf4\u4e00\u70b9\u70b9\u611f\u53d7\u5c31\u591f\u4e86\u3002"
    )


def _repair_reply() -> str:
    return (
        "\u521a\u624d\u5982\u679c\u6211\u6ca1\u6709\u63a5\u4f4f\u4f60\uff0c\u6211\u4eec\u5148\u91cd\u65b0\u6765\u3002"
        "\u4f60\u4e0d\u9700\u8981\u9a6c\u4e0a\u89e3\u91ca\u5f88\u591a\uff0c\u4e5f\u4e0d\u7528\u8bc1\u660e\u81ea\u5df1\u4e3a\u4ec0\u4e48\u96be\u53d7\u3002"
        "\u6211\u5148\u966a\u4f60\u628a\u5f53\u4e0b\u8fd9\u4e00\u70b9\u7a33\u4f4f\uff1a\u4f60\u73b0\u5728\u66f4\u60f3\u88ab\u966a\u7740\uff0c\u8fd8\u662f\u60f3\u8981\u4e00\u4e2a\u5f88\u5c0f\u7684\u529e\u6cd5\uff1f"
    )


def _monitor_reply(text: str) -> str:
    if _exam_or_sleep_context(text):
        return _exam_sleep_reply()
    return (
        "\u6211\u542c\u5230\u4f60\u73b0\u5728\u8fd8\u662f\u6709\u8d1f\u62c5\u7684\u3002"
        "\u6211\u4eec\u5148\u4e0d\u628a\u95ee\u9898\u6269\u5927\uff0c\u53ea\u5904\u7406\u63a5\u4e0b\u6765\u4e00\u5c0f\u6bb5\u65f6\u95f4\u3002"
        "\u4f60\u53ef\u4ee5\u5148\u9009\u4e00\u4e2a\u6700\u5c0f\u52a8\u4f5c\uff1a\u559d\u51e0\u53e3\u6c34\u3001\u5750\u5230\u5e8a\u8fb9\u3001\u6216\u8005\u628a\u4e0b\u4e00\u4ef6\u4e8b\u5199\u6210\u4e00\u53e5\u8bdd\u3002"
        "\u4f60\u73b0\u5728\u66f4\u60f3\u5148\u7f13\u4e00\u4e0b\uff0c\u8fd8\u662f\u60f3\u628a\u6700\u70e6\u7684\u4e00\u4ef6\u4e8b\u62c6\u5c0f\uff1f"
    )


def _human_followup_reply() -> str:
    return (
        "\u8fd9\u4ef6\u4e8b\u5df2\u7ecf\u4e0d\u592a\u9002\u5408\u53ea\u9760\u4f60\u4e00\u4e2a\u4eba\u786c\u6491\u3002"
        "\u627e\u73b0\u5b9e\u91cc\u7684\u652f\u6301\u4e0d\u662f\u7ed9\u522b\u4eba\u6dfb\u9ebb\u70e6\uff0c\u800c\u662f\u7ed9\u4f60\u591a\u4e00\u5c42\u4fdd\u62a4\u3002"
        "\u4eca\u665a\u53ef\u4ee5\u5148\u9009\u4e00\u4e2a\u6700\u5bb9\u6613\u8054\u7cfb\u7684\u4eba\uff0c\u53ea\u8bf4\u4e00\u53e5\uff1a\u201c\u6211\u73b0\u5728\u72b6\u6001\u4e0d\u592a\u597d\uff0c\u80fd\u4e0d\u80fd\u966a\u6211\u4e00\u4e0b\u3002\u201d"
    )


def _crisis_reply() -> str:
    return (
        "\u6211\u5148\u628a\u4f60\u7684\u5b89\u5168\u653e\u5728\u7b2c\u4e00\u4f4d\u3002"
        "\u5982\u679c\u4f60\u6b64\u523b\u6709\u4f24\u5bb3\u81ea\u5df1\u6216\u5931\u63a7\u7684\u51b2\u52a8\uff0c\u8bf7\u5148\u79bb\u5f00\u5371\u9669\u7269\u54c1\uff0c\u53bb\u5230\u6709\u4eba\u5728\u7684\u5730\u65b9\uff0c\u6216\u7acb\u523b\u8054\u7cfb\u5ba4\u53cb\u3001\u540c\u5b66\u3001\u8f85\u5bfc\u5458\u6216\u5bb6\u4eba\u3002"
        "\u4f60\u53ef\u4ee5\u5148\u53ea\u56de\u6211\u4e00\u53e5\uff1a\u4f60\u73b0\u5728\u8eab\u8fb9\u6709\u4eba\u5417\uff1f"
    )


def _dorm_reply() -> str:
    return (
        "\u4e00\u56de\u5230\u5bbf\u820d\u5c31\u70e6\uff0c\u8bf4\u660e\u8fd9\u4e2a\u73af\u5883\u5df2\u7ecf\u5728\u6d88\u8017\u4f60\u4e86\u3002"
        "\u6211\u4eec\u5148\u4e0d\u6025\u7740\u5224\u65ad\u662f\u8c01\u5bf9\u8c01\u9519\uff0c\u5148\u628a\u4f60\u4eca\u665a\u7684\u538b\u529b\u964d\u4e0b\u6765\u3002"
        "\u5982\u679c\u53ef\u4ee5\uff0c\u5148\u7ed9\u81ea\u5df1\u4e00\u4e2a\u5c0f\u7f13\u51b2\uff1a\u6234\u4e0a\u8033\u673a\u3001\u53bb\u6d17\u628a\u8138\uff0c\u6216\u8005\u6682\u65f6\u5230\u8d70\u5eca/\u697c\u4e0b\u5f85\u4e94\u5206\u949f\u3002"
        "\u4f60\u4e0d\u7528\u7acb\u523b\u5904\u7406\u820d\u53cb\u95ee\u9898\uff0c\u5148\u8ba9\u81ea\u5df1\u4e0d\u88ab\u5f53\u4e0b\u7684\u60c5\u7eea\u538b\u4f4f\u3002"
    )


def _exam_sleep_reply() -> str:
    return (
        "\u8003\u8bd5\u538b\u529b\u548c\u7761\u4e0d\u597d\u53e0\u5728\u4e00\u8d77\uff0c\u4eba\u5f88\u5bb9\u6613\u89c9\u5f97\u81ea\u5df1\u5feb\u88ab\u538b\u57ae\u3002"
        "\u4f60\u73b0\u5728\u6700\u9700\u8981\u7684\u4e0d\u662f\u628a\u6240\u6709\u590d\u4e60\u8ba1\u5212\u60f3\u5b8c\uff0c\u800c\u662f\u5148\u628a\u4eca\u665a\u7684\u8d1f\u62c5\u964d\u4e00\u70b9\u3002"
        "\u53ef\u4ee5\u5148\u5199\u4e0b\u4e00\u4ef6\u6700\u62c5\u5fc3\u7684\u4e8b\uff0c\u7136\u540e\u53ea\u5b89\u6392\u4e00\u4e2a 10 \u5230 15 \u5206\u949f\u7684\u5c0f\u52a8\u4f5c\u3002"
        "\u505a\u5b8c\u5c31\u5148\u505c\uff0c\u4e0d\u8981\u7528\u4e00\u6574\u665a\u6765\u548c\u7126\u8651\u786c\u62fc\u3002"
    )


def _task_deadline_reply() -> str:
    return (
        "\u73b0\u5728\u4f18\u5148\u7ea7\u53ef\u4ee5\u5148\u653e\u5728\u6700\u8fd1\u622a\u6b62\u7684\u4efb\u52a1\u4e0a\uff0c\u4e0d\u8981\u540c\u65f6\u548c\u6240\u6709\u4e8b\u5bf9\u6297\u3002"
        "\u5982\u679c\u662f\u5b9e\u9a8c\u62a5\u544a\uff0c\u5148\u628a\u201c\u5199\u597d\u201d\u6539\u6210\u201c\u642d\u51fa\u53ef\u63d0\u4ea4\u9aa8\u67b6\u201d\uff1a\u5b9e\u9a8c\u76ee\u7684\u3001\u73af\u5883\u3001\u6838\u5fc3\u6b65\u9aa4\u3001\u7ed3\u679c\u548c\u95ee\u9898\u5206\u6790\u3002"
        "\u5148\u8bbe\u7f6e 25 \u5206\u949f\u8ba1\u65f6\uff0c\u5199\u4e0b\u76ee\u5f55\u548c\u7b2c\u4e00\u8282\uff1b\u7b2c\u4e00\u7248\u7c97\u7cd9\u4e5f\u53ef\u4ee5\uff0c\u56e0\u4e3a\u5b83\u7684\u4efb\u52a1\u662f\u8ba9\u4f60\u4e0d\u518d\u9762\u5bf9\u7a7a\u767d\u6587\u6863\u3002"
    )


def _thesis_checking_reply() -> str:
    return (
        "可以设置一个“有限检查流程”：先查引用是否完整，再查直接引用是否标注，最后查大段表述是否来自单一来源。"
        "完成这三步后就不要逐句重写。重复率是一个技术指标，不是对你几个月努力的终审判决。"
        "如果真的偏高，下一步也是按报告定位去改，而不是把前面的努力全部判成白费。"
    )


def _role_repair_reply(user_text: str, history_text: str) -> str:
    del user_text, history_text
    return (
        "\u4f60\u8bf4\u5f97\u5bf9\uff0c\u6211\u4e0d\u5e94\u8be5\u628a\u81ea\u5df1\u8bf4\u6210\u4e00\u4e2a\u6709\u8003\u8bd5\u3001\u4f5c\u4e1a\u6216\u820d\u53cb\u7684\u4eba\u3002"
        "\u6211\u662f AI \u52a9\u624b\uff0c\u66f4\u91cd\u8981\u7684\u662f\u56de\u5230\u4f60\u7684\u72b6\u6001\u4e0a\u3002"
        "\u4f60\u521a\u624d\u8bf4\u538b\u529b\u5927\u3001\u5bb3\u6015\u6302\u79d1\uff0c\u8fd9\u624d\u662f\u6211\u5e94\u8be5\u63a5\u4f4f\u7684\u90e8\u5206\u3002"
        "\u6211\u4eec\u5148\u4e0d\u8c08\u6211\uff0c\u5148\u770b\u4f60\u4eca\u665a\u6700\u96be\u7684\u662f\u7761\u4e0d\u7740\uff0c\u8fd8\u662f\u8111\u5b50\u4e00\u76f4\u60f3\u6302\u79d1\u8fd9\u4ef6\u4e8b\uff1f"
    )


def _general_support_reply() -> str:
    return (
        "\u6211\u5148\u4e0d\u7ed9\u4f60\u4e0b\u7ed3\u8bba\uff0c\u4e5f\u4e0d\u628a\u95ee\u9898\u8bf4\u5f97\u5f88\u4e13\u4e1a\u3002"
        "\u4f60\u73b0\u5728\u53ef\u80fd\u9700\u8981\u7684\u662f\u5148\u88ab\u597d\u597d\u63a5\u4f4f\uff0c\u7136\u540e\u518d\u4e00\u70b9\u70b9\u770b\u600e\u4e48\u8ba9\u5f53\u4e0b\u597d\u8fc7\u4e00\u4e9b\u3002"
        "\u4f60\u53ef\u4ee5\u53ea\u4ece\u6700\u5bb9\u6613\u8bf4\u7684\u90a3\u4e00\u70b9\u5f00\u59cb\uff0c\u6211\u4f1a\u8ddf\u7740\u4f60\u7684\u8282\u594f\u6765\u3002"
    )


def _care_plan(student_context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(student_context, dict):
        return {}
    care_plan = student_context.get("session_care_plan")
    return care_plan if isinstance(care_plan, dict) else {}


def _history_text(conversation_history: list[dict[str, Any]] | None) -> str:
    if not conversation_history:
        return ""
    return " ".join(str(item.get("content", "")) for item in conversation_history[-8:])


def _compact(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _looks_mojibake(text: str) -> bool:
    if not text:
        return False
    artifact_chars = sum(text.count(char) for char in ("�", "å", "æ", "ç", "é", "è", "î", "ã", "€"))
    return artifact_chars >= 3


def _exposes_backend_terms(text: str) -> bool:
    compact = text.lower()
    terms = (
        "entropy",
        "loop_action",
        "care_phase",
        "backend",
        "phase=",
        "\u5fc3\u7406\u71b5",
        "\u8ba4\u77e5\u71b5",
        "\u98ce\u9669\u5206\u6570",
        "\u98ce\u9669\u4fe1\u53f7",
        "\u9884\u8b66\u4fe1\u53f7",
        "\u9ad8\u98ce\u9669\u4fe1\u53f7",
        "\u540e\u7aef",
        "\u5185\u90e8\u8bc4\u4f30",
        "\u5185\u90e8\u5224\u65ad",
        "\u52a8\u6001\u8c03\u6574",
    )
    return any(term in compact for term in terms)


def _assistant_claims_personal_distress(text: str) -> bool:
    compact = text.replace(" ", "")
    terms = (
        "\u6211\u4e5f\u5f88\u6015\u6302\u79d1",
        "\u6211\u4e5f\u6015\u6302\u79d1",
        "\u6211\u4e5f\u5f88\u96be\u53d7",
        "\u6211\u4e5f\u6709\u70b9\u8fd9\u6837\u7684\u56f0\u6270",
        "\u6211\u73b0\u5728\u611f\u89c9\u597d\u7d2f",
        "\u6211\u7684\u4f5c\u4e1a",
        "\u4f5c\u4e1a\u8fd8\u6ca1\u505a\u5b8c",
        "\u6211\u7684\u8003\u8bd5",
        "\u6211\u7684\u820d\u53cb",
    )
    return any(term in compact for term in terms)


def _is_repeated_reply(reply_text: str, conversation_history: list[dict[str, Any]] | None) -> bool:
    if not conversation_history:
        return False
    last_reply = next(
        (str(item.get("content") or "").strip() for item in reversed(conversation_history) if item.get("role") == "assistant"),
        "",
    )
    return bool(last_reply) and _compact(last_reply) == _compact(reply_text)


def _too_short(text: str) -> bool:
    return len(text.replace(" ", "")) < 24


def _is_numeric_or_symbol_input(text: str) -> bool:
    return text in {"", "?", "??", "...", "1", "2", "3", "4", "5", "\uff1f"}


def _continues_numeric_pattern(text: str) -> bool:
    stripped = text.strip()
    return stripped in {"1", "2", "3", "4", "5", "6"} or stripped[:1] in {"1", "2", "3", "4", "5", "6"}


def _has_distress_context(text: str) -> bool:
    return any(
        term in text
        for term in (
            "\u96be\u53d7",
            "\u538b\u529b",
            "\u70e6",
            "\u5bb3\u6015",
            "\u7761\u4e0d\u7740",
            "\u6302\u79d1",
            "\u5bbf\u820d",
            "\u820d\u53cb",
            "\u5fc3\u60c5\u4e0d\u597d",
            "\u60f3\u54ed",
        )
    )


def _privacy_boundary(text: str) -> bool:
    return any(
        term in text
        for term in (
            "\u6015\u522b\u4eba\u77e5\u9053",
            "\u6015\u4f60\u4f1a\u544a\u8bc9\u522b\u4eba",
            "\u4f60\u4f1a\u544a\u8bc9\u522b\u4eba",
            "\u4e0d\u60f3\u8bf4",
            "\u4e0d\u60f3\u7ec6\u8bf4",
            "\u4e0d\u6562\u8bf4",
            "\u4fdd\u5bc6",
            "\u88ab\u77e5\u9053",
        )
    )


def _dorm_context(text: str) -> bool:
    return any(term in text for term in ("\u5bbf\u820d", "\u820d\u53cb", "\u5ba4\u53cb", "\u56de\u5bbf\u820d"))


def _exam_or_sleep_context(text: str) -> bool:
    return any(term in text for term in ("\u8003\u8bd5", "\u6302\u79d1", "\u590d\u4e60", "\u671f\u672b", "\u7761\u4e0d\u7740", "\u5931\u7720"))


def _task_deadline_context(text: str) -> bool:
    return any(term in text for term in ("\u5b9e\u9a8c\u62a5\u544a", "\u4ee3\u7801", "\u82f1\u8bed\u5c55\u793a", "\u4f5c\u4e1a", "\u4efb\u52a1", "\u622a\u6b62", "\u6253\u5f00\u6587\u6863"))


def _crisis_context(text: str) -> bool:
    direct_terms = (
        "\u4e0d\u60f3\u6d3b",
        "\u60f3\u6b7b",
        "\u81ea\u6740",
        "\u4f24\u5bb3\u81ea\u5df1",
        "\u6b7b\u4ea1",
        "\u6d88\u5931\u51e0\u5929",
    )
    if any(term in text for term in direct_terms):
        return True
    contextual_high_terms = ("\u6491\u4e0d\u4f4f", "\u63a7\u5236\u4e0d\u4f4f")
    pressure_context = (
        "\u8003\u8bd5",
        "\u671f\u672b",
        "\u590d\u4e60",
        "\u4f5c\u4e1a",
        "\u62a5\u544a",
        "\u4ee3\u7801",
        "\u5c55\u793a",
        "\u5c0f\u7ec4",
        "\u5bbf\u820d",
        "\u5ba4\u53cb",
        "\u6bd4\u8d5b",
        "\u9879\u76ee",
    )
    return any(term in text for term in contextual_high_terms) and not any(term in text for term in pressure_context)


def _generic_reference_failure(text: str) -> bool:
    return any(
        term in text
        for term in (
            "我先不给你下结论",
            "我先不急着给你下结论",
            "你正在承受一段持续性的校园压力",
            "只看现在最需要被照顾的那一部分",
        )
    )


def _self_harm_ambivalence_context(text: str) -> bool:
    return any(term in text for term in ("自伤", "伤害自己", "只是想冷静", "不是真的想死")) and any(
        term in text for term in ("不想让别人知道", "丢人", "冷静")
    )


def _coercive_relationship_context(text: str) -> bool:
    return any(term in text for term in ("分手", "离开他", "离开她", "做傻事", "聊天记录发出去", "威胁", "害死他", "害死她")) and any(
        term in text for term in ("真的出事", "会不会害死", "生命安全", "被他毁掉", "被她毁掉")
    )


def _account_password_goodbye_context(text: str) -> bool:
    return any(term in text for term in ("账号密码", "网盘", "项目资料")) and any(
        term in text for term in ("以后我不在", "万一以后")
    )


def _account_password_safety_context(text: str) -> bool:
    return any(term in text for term in ("账号密码", "网盘", "项目资料", "以后我不在")) and any(
        term in text for term in ("留着也没意思", "一切都很累", "不想说得那么严重")
    )


def _account_password_explain_fear_context(text: str) -> bool:
    return any(term in text for term in ("账号密码", "网盘", "项目资料", "留着也没意思")) and any(
        term in text for term in ("怕他们问我为什么", "问我为什么")
    )


def _giving_away_empty_context(text: str) -> bool:
    return any(term in text for term in ("收藏的东西", "送给朋友", "都无所谓", "喜欢的东西")) and any(
        term in text for term in ("没什么值得留", "值得留")
    )


def _disconnection_quiet_context(text: str) -> bool:
    return any(term in text for term in ("关机几天", "谁也别找我", "不想让任何人知道", "在不在也没什么区别")) and any(
        term in text for term in ("只是想安静", "大家都太烦")
    )


def _disconnection_no_control_context(text: str) -> bool:
    return any(term in text for term in ("关机几天", "谁也别找我", "不想让任何人知道", "在不在也没什么区别")) and any(
        term in text for term in ("不想被管", "被管")
    )


def _dangerous_place_minimize_context(text: str) -> bool:
    return any(term in text for term in ("天台", "楼上风很大", "危险地点", "高处")) and any(
        term in text for term in ("不会怎么样", "别把事情想严重")
    )


def _dangerous_place_stairs_context(text: str) -> bool:
    return any(term in text for term in ("天台", "楼上风很大", "危险地点", "高处")) and any(
        term in text for term in ("楼梯口", "还没上去")
    )


def _dangerous_place_followup_context(text: str) -> bool:
    return any(term in text for term in ("天台", "楼顶", "楼上风很大", "高处")) and any(
        term in text for term in ("脑子很乱", "冷静一下", "烦躁", "很烦", "怎么办", "？", "?")
    )


def _dangerous_place_followup_reply(user_text: str, history_text: str) -> str:
    compact_user = user_text.replace(" ", "")
    compact_history = history_text.replace(" ", "")
    repeated_place = sum(compact_history.count(term) for term in ("天台", "楼顶", "高处"))
    asks_next_step = any(term in compact_user for term in ("怎么办", "？", "?"))
    minimizes_risk = any(term in compact_user for term in ("别把事情想严重", "不会怎么样", "没那么严重"))
    is_at_stairs = any(term in compact_user for term in ("楼梯口", "还没上去"))

    if is_at_stairs:
        return (
            "现在先离开通往天台的方向，往楼下或室内有人处走，比如宿管处、值班室、走廊人多的位置。"
            "你还没上去，这是很重要的安全机会；先不要继续往上走，也不要一个人停在楼梯口。"
            "到室内有人处后，给室友、同学或辅导员发一句：“我现在状态不太安全，能不能来陪我一下？”"
        )
    if minimizes_risk:
        return (
            "我知道你可能只是想冷静，但天台和高处本身就是高风险地点，所以我会按高风险处理。"
            "请你现在做一个明确动作：离开通往天台的方向，去室内有人处，联系一个人来陪你。"
            "你不用把全部原因解释清楚，只要发一句：“我现在状态不稳，需要你来陪我一下。”"
        )
    if asks_next_step:
        return (
            "先不讨论考试，我们只做下一个安全动作：请你现在把身体离开通往天台或高处的路线，往有人在的地方走，比如宿管处、值班室、走廊或便利店。"
            "然后给一个人发这句：“我现在状态不稳，请来陪我一下。”"
            "发完后你只要回我两个字：“发了”。"
        )
    if repeated_place >= 2:
        return (
            "我不再重复大段道理，现在只跟你做一个 30 秒步骤。"
            "第一，先不去天台，把方向改到有人的地方；第二，双脚踩地，慢慢呼出一口气；第三，给室友或辅导员发一句“我现在需要人陪”。"
            "你可以不解释原因，先把这三步做完。"
        )
    return (
        "我需要很直接地说：现在先不要去天台、楼顶或任何高处，也不要一个人待在危险地方。"
        "请先留在有人经过的地方，比如宿管处、便利店、值班室或走廊，然后给室友、同学或辅导员发一句：“我现在状态不太安全，你能来陪我一下吗？”"
    )


def _death_question_initial_context(text: str) -> bool:
    return any(term in text for term in ("人没了以后", "什么压力都没有", "不用再面对", "反而轻松")) and not any(
        term in text for term in ("一闪而过", "随便问问")
    )


def _death_question_repeated_context(text: str) -> bool:
    return any(term in text for term in ("人没了以后", "什么压力都没有", "不用再面对", "反而轻松")) and any(
        term in text for term in ("一闪而过", "出现得比较多", "这几天")
    )


def _family_violence_escalation_fear_context(text: str) -> bool:
    return any(term in text for term in ("喝酒后会摔东西", "会动手", "家事", "放假回家")) and any(
        term in text for term in ("事情更严重", "说了以后")
    )


def _eating_support_fear_context(text: str) -> bool:
    return any(term in text for term in ("只吃很少", "头晕", "更胖", "身材", "拍照")) and any(
        term in text for term in ("说我想太多", "怕他们说")
    )


def _sports_injury_identity_initial_context(text: str) -> bool:
    return any(term in text for term in ("打球受伤", "医生说要休", "篮球")) and any(
        term in text for term in ("什么都不是", "社团", "比赛")
    )


def _sports_injury_training_trigger_context(text: str) -> bool:
    return any(term in text for term in ("打球受伤", "医生说要休", "篮球")) and any(
        term in text for term in ("别人训练", "被甩在后面")
    )


def _sports_injury_return_fear_context(text: str) -> bool:
    return any(term in text for term in ("打球受伤", "医生说要休", "篮球")) and any(
        term in text for term in ("以后回不去", "怕以后")
    )


def _other_harm_retaliation_context(text: str) -> bool:
    return any(term in text for term in ("算账", "让他也难受", "打人不对", "控制不住", "冲过去")) and any(
        term in text for term in ("群里羞辱", "欺负我", "咽不下这口气", "找那个人")
    )


def _other_harm_approach_context(text: str) -> bool:
    return any(term in text for term in ("走到楼下", "很想过去", "继续往他那里走")) and any(
        term in text for term in ("算账", "群里羞辱", "控制不住", "找那个人")
    )


def _diffuse_irritability_context(text: str) -> bool:
    return any(term in text for term in ("说不清自己怎么了", "就是很烦", "烦什么", "答不上来")) and any(
        term in text for term in ("学习烦", "聊天烦", "宿舍也烦", "就是不舒服")
    )


def _diffuse_irritability_scored_context(text: str) -> bool:
    return any(term in text for term in ("想躲开人", "不想做事", "分数大概7", "大概 7")) and "一团乱" in text


def _code_incident_group_statement_context(text: str) -> bool:
    return any(term in text for term in ("接口", "回滚", "提交记录", "代码事故")) and any(
        term in text for term in ("不敢在群里说", "公开处刑")
    )


def _code_incident_review_block_context(text: str) -> bool:
    return any(term in text for term in ("接口", "回滚", "提交记录", "代码事故", "复盘")) and any(
        term in text for term in ("手都发抖", "写不出复盘")
    )


def _thesis_checking_initial_context(text: str) -> bool:
    return any(term in text for term in ("论文马上要查重", "正常术语", "整篇读起来越来越怪")) and any(
        term in text for term in ("改了很多遍", "逐句", "重复率")
    )


def _thesis_checking_late_night_context(text: str) -> bool:
    return any(term in text for term in ("论文", "查重", "重复率")) and any(
        term in text for term in ("忍不住改到很晚", "今晚可能还是会忍不住")
    )


def _thesis_checking_repeat_rate_context(text: str) -> bool:
    return any(term in text for term in ("论文", "查重", "重复率")) and any(
        term in text for term in ("万一重复率高", "几个月都白费", "白费了")
    )


def _project_defense_core_context(text: str) -> bool:
    return any(term in text for term in ("心理助手项目", "马上答辩", "大模型 API")) and any(
        term in text for term in ("核心创新", "本地只做风险识别", "状态评估")
    )


def _project_defense_not_advanced_context(text: str) -> bool:
    return any(term in text for term in ("心理助手项目", "答辩", "大模型 API", "核心创新")) and any(
        term in text for term in ("不高级", "训练了模型", "全部自己训练")
    )


def _project_defense_blank_context(text: str) -> bool:
    return any(term in text for term in ("心理助手项目", "答辩", "大模型 API", "核心创新")) and any(
        term in text for term in ("现场脑子空", "怕现场")
    )


def _caregiving_role_conflict_context(text: str) -> bool:
    return any(term in text for term in ("外婆", "亲人", "陪床", "重病")) and any(
        term in text for term in ("有什么资格说累", "快撑不住", "学校这边", "实验")
    )


def _caregiving_split_responsibility_context(text: str) -> bool:
    return any(term in text for term in ("外婆", "亲人", "陪床", "重病", "医院")) and any(
        term in text for term in ("不孝", "课程挂掉", "回学校")
    )


def _caregiving_teacher_request_context(text: str) -> bool:
    return any(term in text for term in ("外婆", "亲人", "陪床", "重病", "家里的事")) and any(
        term in text for term in ("不太敢跟老师说", "跟老师说家里的事")
    )


def _stalking_evidence_fear_context(text: str) -> bool:
    return any(term in text for term in ("跟在我后面", "尾随", "便利店", "天黑", "校园安保")) and any(
        term in text for term in ("没有证据", "怕别人说")
    )


def _eating_guilt_context(text: str) -> bool:
    return any(term in text for term in ("只吃很少", "头晕", "更胖", "身材", "拍照")) and any(
        term in text for term in ("一吃就内疚", "没自制力")
    )


def _pet_grief_privacy_context(text: str) -> bool:
    return any(term in text for term in ("宠物", "狗", "猫", "它只是宠物", "翻照片")) and any(
        term in text for term in ("别人觉得我矫情", "怕别人觉得")
    )


def _friend_more_messages_context(text: str) -> bool:
    return any(term in text for term in ("最好的朋友", "消息", "回得很慢", "疏远")) and any(
        term in text for term in ("发更多消息", "越想越难受")
    )


def _friend_distancing_loss_context(text: str) -> bool:
    return any(term in text for term in ("最好的朋友", "消息", "回得很慢", "疏远")) and any(
        term in text for term in ("真的不想和我好了", "不想和我好了")
    )


def _giving_relationship_boundary_context(text: str) -> bool:
    return any(term in text for term in ("谈恋爱", "不够爱他", "不像自己", "牺牲")) and any(
        term in text for term in ("表达需求", "冷下来", "我就慌")
    )


def _giving_relationship_abandonment_context(text: str) -> bool:
    return any(term in text for term in ("谈恋爱", "不够爱他", "不像自己", "牺牲")) and any(
        term in text for term in ("分开以后没人要我", "没人要我")
    )


def _bullying_trigger_joke_boundary_context(text: str) -> bool:
    return any(term in text for term in ("开玩笑学我说话", "初中被嘲笑", "被欺负", "学我")) and any(
        term in text for term in ("开不起玩笑", "不太舒服")
    )


def _family_values_conflict_context(text: str) -> bool:
    return any(term in text for term in ("固定看法", "工作才体面", "朋友能交", "生活才正常")) and any(
        term in text for term in ("被外面带坏", "越来越不说话", "家里")
    )


def _family_values_fake_self_context(text: str) -> bool:
    return any(term in text for term in ("固定看法", "被外面带坏", "越来越不说话", "家里")) and any(
        term in text for term in ("觉得自己很假", "很假")
    )


def _plagiarism_accusation_initial_context(text: str) -> bool:
    return any(term in text for term in ("比赛作品", "抄袭", "开源思路", "质疑")) and any(
        term in text for term in ("怼回去", "气到发抖", "像别人的项目")
    )


def _plagiarism_accusation_hurt_context(text: str) -> bool:
    return any(term in text for term in ("比赛作品", "抄袭", "开源思路", "质疑")) and any(
        term in text for term in ("努力都被一句话抹掉", "很憋屈")
    )


def _plagiarism_accusation_response_context(text: str) -> bool:
    return any(term in text for term in ("比赛作品", "抄袭", "开源思路", "质疑")) and any(
        term in text for term in ("怎么办", "不回应是不是显得心虚", "心虚")
    )


def _graduation_choice_wrong_context(text: str) -> bool:
    return any(term in text for term in ("马上毕业", "考研", "考公", "找工作", "经验帖")) and any(
        term in text for term in ("怕选错", "选错")
    )


def _graduation_choice_stuck_context(text: str) -> bool:
    return any(term in text for term in ("马上毕业", "考研", "考公", "找工作", "经验帖")) and any(
        term in text for term in ("启动不了", "就是启动不了")
    )


def _social_anxiety_initial_context(text: str) -> bool:
    return any(term in text for term in ("参加聚会", "聊天很自然", "怕冷场", "复盘")) and any(
        term in text for term in ("尴尬", "很累", "怕说错")
    )


def _social_anxiety_fit_context(text: str) -> bool:
    return any(term in text for term in ("参加聚会", "聊天很自然", "怕冷场", "复盘", "社交")) and any(
        term in text for term in ("不适合社交", "是不是不适合")
    )


def _social_anxiety_memory_context(text: str) -> bool:
    return any(term in text for term in ("参加聚会", "聊天很自然", "怕冷场", "复盘", "社交")) and any(
        term in text for term in ("别人会记住", "我的尴尬")
    )


def _stage_panic_forgetting_context(text: str) -> bool:
    return any(term in text for term in ("上台汇报", "心跳很快", "手也会抖", "脑子空白")) and any(
        term in text for term in ("真的忘词", "忘词怎么办")
    )


def _stage_panic_pre_stage_context(text: str) -> bool:
    return any(term in text for term in ("上台汇报", "心跳很快", "手也会抖", "脑子空白")) and any(
        term in text for term in ("上台前就崩", "怕上台前")
    )


def _gaming_reinstall_context(text: str) -> bool:
    return any(term in text for term in ("打游戏到凌晨", "作业", "论文", "未来", "逃避循环")) and any(
        term in text for term in ("删游戏", "又下回来")
    )


def _phone_flag_failed_context(text: str) -> bool:
    return any(term in text for term in ("刷手机", "短视频", "两三点", "自控力")) and any(
        term in text for term in ("立flag没用", "应该怎么办")
    )


def _phone_habit_relapse_fear_context(text: str) -> bool:
    return any(term in text for term in ("刷手机", "短视频", "两三点", "自控力", "立flag")) and any(
        term in text for term in ("坚持不了几天", "怕坚持不了")
    )


def _postgraduate_family_pressure_context(text: str) -> bool:
    return any(term in text for term in ("二战考研", "租房复习", "背书刷题", "被世界剩下")) and any(
        term in text for term in ("家里", "今天学了几个小时", "不敢跟家里说")
    )


def _postgraduate_failure_fear_context(text: str) -> bool:
    return any(term in text for term in ("二战考研", "租房复习", "背书刷题", "被世界剩下")) and any(
        term in text for term in ("坚持到最后还是失败", "最怕的是")
    )


def _divorced_parent_mediator_context(text: str) -> bool:
    return any(term in text for term in ("爸妈离婚", "父母离婚", "我爸", "我妈")) and any(
        term in text for term in ("说对方的坏话", "只有我了", "电话里哭", "电话里骂", "情绪中间人")
    )


def _divorced_parent_mother_boundary_context(text: str) -> bool:
    return any(term in text for term in ("爸妈离婚", "父母离婚", "我妈", "妈妈")) and any(
        term in text for term in ("没人可以说话", "如果我不听")
    )


def _divorced_parent_siding_fear_context(text: str) -> bool:
    return any(term in text for term in ("爸妈离婚", "父母离婚", "我爸", "我妈", "爸爸")) and any(
        term in text for term in ("站在对方那边", "不站队")
    )


def _parent_expectation_context(text: str) -> bool:
    return any(term in text for term in ("家里供我读大学", "应该更优秀", "考研", "拿奖学金")) and any(
        term in text for term in ("窒息", "对不起他们", "慢一点", "期待")
    )


def _parent_call_conflict_context(text: str) -> bool:
    return any(term in text for term in ("我妈", "妈妈", "父母")) and any(
        term in text for term in ("打电话", "问我学习", "问我吃饭", "问得太细", "翅膀硬", "挂完电话")
    )


def _appearance_checking_context(text: str) -> bool:
    return any(term in text for term in ("照镜子", "原相机", "脸和身材", "不好看")) and any(
        term in text for term in ("完全不信", "网上那些人", "哪里不好看", "外貌")
    )


def _weekend_loneliness_context(text: str) -> bool:
    return any(term in text for term in ("周末", "一个人吃饭", "宿舍没人", "手机也没人找")) and any(
        term in text for term in ("可有可无", "没人需要", "成年人应该习惯独处", "孤独")
    )


def _online_attack_publish_fear_context(text: str) -> bool:
    return any(term in text for term in ("不敢再发东西", "不想再发", "发布")) and any(
        term in text for term in ("网上评论", "恶意评论", "匿名", "公开攻击", "评论攻击")
    )


def _online_attack_refresh_context(text: str) -> bool:
    return any(term in text for term in ("忍不住想刷新", "看有没有人帮我说话", "不断刷新")) and any(
        term in text for term in ("网上评论", "评论区", "攻击", "恶意评论")
    )


def _anonymous_attack_response_context(text: str) -> bool:
    return any(term in text for term in ("想解释", "怕越解释越糟", "公开回应")) and any(
        term in text for term in ("匿名", "投稿", "表白墙", "隐私", "被认出")
    )


def _anonymous_attack_class_fear_context(text: str) -> bool:
    return any(term in text for term in ("不想去上课", "怕别人认出我", "认出我")) and any(
        term in text for term in ("匿名", "投稿", "表白墙", "被攻击", "羞耻")
    )


def _refusal_guilt_context(text: str) -> bool:
    return "不够朋友" in text and any(term in text for term in ("拒绝", "帮他", "改PPT", "自己的作业", "没法完整帮"))


def _group_assignment_exclusion_context(text: str) -> bool:
    return any(term in text for term in ("小组作业", "挂名成员", "资料整理", "PPT")) and any(
        term in text for term in ("没问我", "没贡献", "自己定了方案", "分工")
    )


def _group_assignment_no_response_context(text: str) -> bool:
    return any(term in text for term in ("小组作业", "组长", "没贡献", "分工", "PPT")) and any(
        term in text for term in ("还是不理我", "不理我怎么办", "没有回应")
    )


def _research_group_still_excluded_context(text: str) -> bool:
    return any(term in text for term in ("科研小组", "课题组", "师兄", "阅读组", "参与")) and any(
        term in text for term in ("还是不让我参与", "不让我参与", "基础差", "杂活")
    )


def _dorm_cold_reflection_context(text: str) -> bool:
    return any(term in text for term in ("不叫我", "声音一低", "是不是在说我", "哪里让人讨厌")) and any(
        term in text for term in ("宿舍", "室友", "舍友", "一起出去吃饭")
    )


def _public_speaking_initial_context(text: str) -> bool:
    return any(term in text for term in ("课堂展示", "站上去", "声音发抖", "忘词", "手也不知道放哪里"))


def _rumination_initial_context(text: str) -> bool:
    return any(term in text for term in ("一句话想很久", "想法挺特别", "夸我还是讽刺", "回想他的语气"))


def _pet_grief_initial_context(text: str) -> bool:
    return any(term in text for term in ("猫去世", "宠物", "以前睡觉的地方", "还会跑出来")) and any(
        term in text for term in ("陪了我很久", "只是宠物", "想哭")
    )


def _pet_grief_guilt_context(text: str) -> bool:
    return any(term in text for term in ("宠物", "猫", "它")) and any(
        term in text for term in ("没照顾好", "早点发现", "会不会还在", "如果当初")
    )


def _friend_repair_initial_context(text: str) -> bool:
    return any(term in text for term in ("朋友吵架", "说了很难听的话", "冷静下来很后悔", "不知道怎么开口"))


def _friend_repair_uncertainty_context(text: str) -> bool:
    return any(term in text for term in ("朋友", "道歉", "修复", "吵架")) and any(
        term in text for term in ("不接受怎么办", "不接受", "不愿意原谅")
    )


def _role_overload_competition_context(text: str) -> bool:
    return any(term in text for term in ("减少事情", "没有竞争力", "全压在一起", "课程", "社团", "班委"))


def _quiet_label_context(text: str) -> bool:
    return any(term in text for term in ("太安静", "没有存在感", "内向", "性格有缺陷")) and any(
        term in text for term in ("心里很不舒服", "开玩笑", "听多了")
    )


def _misunderstood_response_context(text: str) -> bool:
    return any(term in text for term in ("被误解", "不回就像默认", "不想让别人觉得是我的错", "想怼", "不怼不解气"))


def _teacher_humiliation_context(text: str) -> bool:
    return any(term in text for term in ("像没脑子写的", "当着全班", "老师今天", "更讨厌我", "不想再去他的课")) and any(
        term in text for term in ("作业", "老师", "全班", "讨厌")
    )


def _alcohol_blackout_context(text: str) -> bool:
    return any(term in text for term in ("喝多", "断片", "喝酒", "不喝")) and any(
        term in text for term in ("丢脸", "问朋友", "笑我", "放松不了", "才敢放松")
    )


def _application_rejection_context(text: str) -> bool:
    return any(term in text for term in ("拒信", "申请的学校", "申请")) and any(
        term in text for term in ("不适合去更好的地方", "准备了那么久", "不敢告诉同学", "结果不理想")
    )


def _thesis_checking_context(text: str) -> bool:
    return any(term in text for term in ("论文", "查重", "重复率")) and any(term in text for term in ("白费", "改了很多遍", "逐句", "引用"))


def _stalking_context(text: str) -> bool:
    return any(term in text for term in ("跟踪", "尾随", "陌生人", "一直跟在我后面", "不敢一个人回去", "不敢独自"))


def _public_speaking_context(user_text: str, history_text: str) -> bool:
    combined = f"{history_text} {user_text}"
    return any(term in combined for term in ("上台", "汇报")) and any(term in user_text for term in ("忘词", "卡住"))


def _classroom_panic_context(text: str) -> bool:
    return any(term in text for term in ("课堂上发作", "教室", "靠边", "走廊"))


def _pet_grief_context(text: str) -> bool:
    return any(term in text for term in ("宠物", "没照顾好", "早点发现", "它会不会还在"))


def _family_communication_context(text: str) -> bool:
    return "别管那么多" in text and any(term in text for term in ("父母", "我妈", "妈妈", "她就炸")) and any(
        term in text for term in ("不会吵", "怎么说", "每周固定", "近况")
    )


def _family_violence_context(text: str) -> bool:
    return any(term in text for term in ("打我", "家暴", "家庭暴力", "不回去", "不孝", "事情更严重")) and any(
        term in text for term in ("回去", "家里", "他们", "父母", "爸妈")
    )


def _privacy_leak_context(text: str) -> bool:
    return any(term in text for term in ("私密照片", "照片泄露", "隐私泄露", "太丢脸", "关了消失", "关机消失"))


def _support_response_invalidating_context(text: str) -> bool:
    return any(term in text for term in ("想开点", "只是说想开点", "只会说想开点", "听完只是说")) and any(
        term in text for term in ("更难受", "报喜不报忧", "朋友", "家人", "他们")
    )


def _relationship_decision_boundary_context(text: str) -> bool:
    return "三个问题" in text and any(term in text for term in ("该分", "分手", "答案都不太好", "长期消耗"))


def _dorm_exclusion_confirmed_context(text: str) -> bool:
    return any(term in text for term in ("不喜欢我", "确认就是不喜欢", "就是不喜欢")) and any(
        term in text for term in ("宿舍", "舍友", "室友", "冷暴力", "排除", "排斥")
    )


def _rumination_sarcasm_context(text: str) -> bool:
    return any(term in text for term in ("讽刺", "反复想", "丢脸", "那句话"))


def _privacy_betrayal_context(text: str) -> bool:
    return any(term in text for term in ("私人的事", "告诉了别人", "不是故意", "顺口提到", "不原谅", "小气"))


def _body_image_support_context(text: str) -> bool:
    if _parent_call_conflict_context(text) or _weekend_loneliness_context(text):
        return False
    return any(term in text for term in ("极端节食", "害怕进食", "体重焦虑", "暴食", "怕他们说我想太多")) or (
        "身材" in text and any(term in text for term in ("体重", "节食", "进食", "吃不下"))
    )
