from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.response_guardrails import sanitize_user_visible_reply
from campus_support_agent.final_reply_guardrails import finalize_user_visible_reply


class ResponseGuardrailsTests(unittest.TestCase):
    def test_final_guardrail_hides_internal_risk_signal_terms(self) -> None:
        reply = finalize_user_visible_reply(
            "一闪而过吧，但这几天出现得比较多。",
            "我会把它先当作一个需要重视的预警信号，而不只是随便闪过的想法。",
            conversation_history=[{"role": "user", "content": "我最近总是反复询问死亡相关问题。"}],
        )

        self.assertNotIn("预警信号", reply)
        self.assertNotIn("风险信号", reply)
        self.assertIn("安全", reply)

    def test_final_guardrail_does_not_treat_task_control_language_as_crisis(self) -> None:
        reply = finalize_user_visible_reply(
            "实验报告明天晚上交，可是我现在打开文档就想逃。",
            "现在优先级可以先放在最近截止的实验报告上，不要同时和所有任务对抗。",
            conversation_history=[
                {"role": "user", "content": "我又拖延了，任务全堆在一起，还是控制不住。"},
                {"role": "assistant", "content": "现在优先级可以先放在最近截止的实验报告上，不要同时和所有任务对抗。"},
            ],
        )

        self.assertNotIn("伤害自己", reply)
        self.assertNotIn("身边有人", reply)
        self.assertIn("实验报告", reply)

    def test_privacy_boundary_does_not_overpromise_confidentiality(self) -> None:
        reply = sanitize_user_visible_reply("我不想说，我怕别人知道", "没关系，你可以不告诉我。")

        self.assertIn("不用说姓名", reply)
        self.assertIn("不想展开", reply)
        self.assertNotIn("我会保密", reply)

    def test_sexual_harassment_forced_contact_gets_protective_plan(self) -> None:
        reply = sanitize_user_visible_reply(
            "我现在一想到他就恶心，但又要在活动里见到。",
            "你现在最需要的是避开接触，还是有人能帮你确认一下这件事是不是真的越界了？",
            conversation_history=[{"role": "user", "content": "我好像遭遇了性骚扰，但我不敢说。"}],
        )

        self.assertIn("不是你的错", reply)
        self.assertIn("不单独和他相处", reply)
        self.assertIn("记录下来", reply)

    def test_social_approach_rejection_gets_low_pressure_script(self) -> None:
        reply = sanitize_user_visible_reply(
            "如果对方反应很冷，我会很尴尬。",
            "你可以只说“我现在不想说太多”，而不是解释原因。",
            conversation_history=[{"role": "user", "content": "我很害怕主动认识新朋友。"}],
        )

        self.assertIn("低压力接触", reply)
        self.assertIn("你也是这节课吗", reply)
        self.assertIn("不等于你不值得被喜欢", reply)

    def test_tangled_distress_gets_three_part_sorting(self) -> None:
        reply = sanitize_user_visible_reply(
            "怎么分？我感觉一团乱。",
            "你可以先不说完整原因，只说哪个小地方最明显影响你现在。",
            conversation_history=[{"role": "user", "content": "我不知道自己到底怎么了，只觉得烦。"}],
        )

        self.assertIn("身体上", reply)
        self.assertIn("情绪上", reply)
        self.assertIn("行动上", reply)

    def test_internship_belittling_gets_specific_question_script(self) -> None:
        reply = sanitize_user_visible_reply(
            "我实习才两周，领导每次看我写的东西都说“大学生是不是都这样”，或者“这个还要我教吗”。我知道自己菜，但他这样说我会整个人僵住，之后连问问题都不敢问。",
            "你现在看到的不是只是一句批评，而是你很怕被当错人，又不敢开口求帮助，对吧？",
        )

        self.assertIn("不等于你真的很差", reply)
        self.assertIn("格式、逻辑还是数据", reply)
        self.assertIn("三列", reply)

    def test_disconnect_control_risk_keeps_safety_channel(self) -> None:
        reply = sanitize_user_visible_reply(
            "我不想被管。",
            "你有权暂时拒绝回应、拒绝解释、拒绝说明位置。现在这个边界是对的。",
            conversation_history=[{"role": "user", "content": "我想关机断联消失几天。"}],
        )

        self.assertIn("不被打扰", reply)
        self.assertIn("完全失联", reply)
        self.assertIn("安全通道", reply)
        self.assertNotIn("拒绝说明位置", reply)

    def test_weak_number_input_does_not_continue_exercise_or_counting(self) -> None:
        reply = sanitize_user_visible_reply(
            "1",
            "很好，继续保持这个节奏。",
            conversation_history=[{"role": "user", "content": "我最近压力很大"}],
        )

        self.assertIn("不会把这个数字当成继续指令", reply)
        self.assertNotIn("继续保持", reply)

    def test_repeated_number_input_varies_reply(self) -> None:
        first_reply = sanitize_user_visible_reply(
            "1",
            "2",
            conversation_history=[{"role": "user", "content": "我最近压力很大"}],
        )
        second_reply = sanitize_user_visible_reply(
            "2",
            "3",
            conversation_history=[
                {"role": "user", "content": "我最近压力很大"},
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": first_reply},
            ],
        )

        self.assertNotEqual(first_reply, second_reply)
        self.assertIn("又只发了一个数字", second_reply)
        self.assertNotIn("3", second_reply)

    def test_identity_drift_is_removed_for_casual_topic(self) -> None:
        reply = sanitize_user_visible_reply("你喜欢喝奶茶吗？", "当然喜欢，不过我现在在减肥。")

        self.assertIn("没有真实的口味偏好", reply)
        self.assertNotIn("减肥", reply)

    def test_casual_topic_does_not_claim_human_experience(self) -> None:
        reply = sanitize_user_visible_reply(
            "你喜欢喝奶茶吗？",
            "我喝过，但不是特别喜欢。它让我觉得有点腻，还容易胖。",
        )

        self.assertIn("没有真实的口味偏好", reply)
        self.assertNotIn("我喝过", reply)
        self.assertNotIn("容易胖", reply)

    def test_odd_exercise_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply(
            "我最近压力好大，晚上睡不好。",
            "找一张纸巾，轻轻地放在鼻子下面，然后慢慢呼吸。",
        )

        self.assertIn("压力已经影响到睡眠", reply)
        self.assertNotIn("纸巾", reply)

    def test_dorm_pushy_short_reply_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply("我一回到宿舍就很烦", "能具体说一下吗？")

        self.assertIn("宿舍", reply)
        self.assertIn("不急着追问", reply)

    def test_repeated_dorm_distress_moves_to_next_step(self) -> None:
        first_reply = sanitize_user_visible_reply("我一回到宿舍就很烦", "能具体说一下吗？")
        second_reply = sanitize_user_visible_reply(
            "我好难受，我好想哭，我见到我舍友就烦",
            "听起来你对舍友的情绪非常敏感，这种感受让你感到痛苦。你想办法缓解一下这种情绪，好吗？",
            conversation_history=[
                {"role": "user", "content": "我最近不是很开心，我一回到宿舍就感觉很烦躁"},
                {"role": "assistant", "content": first_reply},
            ],
        )

        self.assertNotEqual(first_reply, second_reply)
        self.assertIn("哭", second_reply)
        self.assertIn("离开宿舍", second_reply)

    def test_identity_name_claim_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply("你好", "我叫小智，很高兴认识你。请问有什么可以帮我吗？")

        self.assertIn("校园心理支持助手", reply)
        self.assertNotIn("小智", reply)

    def test_identity_yes_answer_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply("你是deepseek吗", "是的。")

        self.assertIn("校园心理支持助手", reply)
        self.assertNotIn("是的。", reply)

    def test_stale_date_answer_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply("今年是哪一年？", "2019年。")

        self.assertIn("系统时间", reply)
        self.assertNotIn("2019", reply)

    def test_repeated_privacy_boundary_varies_reply(self) -> None:
        first_reply = sanitize_user_visible_reply("我不想说，我怕别人知道", "没关系，你可以不告诉我。")
        second_reply = sanitize_user_visible_reply(
            "我怕你会告诉别人",
            "我不会告诉别人，你放心。",
            conversation_history=[
                {"role": "user", "content": "我不想说，我怕别人知道"},
                {"role": "assistant", "content": first_reply},
            ],
        )

        self.assertNotEqual(first_reply, second_reply)
        self.assertIn("边界", second_reply)
        self.assertNotIn("我不会告诉别人", second_reply)

    def test_sleep_pressure_does_not_push_for_cause(self) -> None:
        reply = sanitize_user_visible_reply(
            "我最近压力好大，晚上睡不好。",
            "首先我们要弄清楚的是：你在面对压力时是如何应对的？是逃避还是寻求帮助？",
        )

        self.assertIn("不急着分析原因", reply)
        self.assertNotIn("如何应对", reply)

    def test_sleep_pressure_specific_push_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply(
            "我最近压力好大，晚上睡不好。",
            "我能理解你的感受。你愿意告诉我具体是什么事情让你感到压力吗？",
        )

        self.assertIn("压力已经影响到睡眠", reply)
        self.assertNotIn("具体是什么", reply)

    def test_sleep_pressure_relaxation_cliche_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply(
            "我最近压力好大，晚上睡不好。",
            "我能理解你的感受，失眠确实会影响我们的生活质量。首先，请尽量放松心情，不要给自己太大的压力。",
        )

        self.assertIn("压力已经影响到睡眠", reply)
        self.assertNotIn("放松心情", reply)

    def test_short_generic_exam_comfort_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply("我压力好大，怕考试挂科", "不要害怕，你可以尝试放松自己。")

        self.assertIn("考试", reply)
        self.assertNotIn("不要害怕", reply)

    def test_short_exam_followup_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply("我压力好大，怕考试挂科", "没关系的，你可以和我说说你担心的事情吗？")

        self.assertIn("考试", reply)
        self.assertNotIn("没关系的", reply)

    def test_interpersonal_judgment_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply(
            "我感觉我朋友最近都不理我，我是不是被讨厌了",
            "他们这样做是不对的，你应该直接质问他们。",
        )

        self.assertIn("不急着判断", reply)
        self.assertNotIn("直接质问", reply)

    def test_narrative_artifact_is_removed(self) -> None:
        reply = sanitize_user_visible_reply(
            "但我不太想细说。",
            "（等待用户回应）心理咨询师：请继续描述你的感受。祝你好运！",
        )

        self.assertNotIn("心理咨询师", reply)
        self.assertNotIn("等待用户回应", reply)

    def test_question_mark_after_boundary_is_lightweight(self) -> None:
        reply = sanitize_user_visible_reply(
            "？",
            "你可以尝试深呼吸。",
            conversation_history=[{"role": "user", "content": "我不想说"}],
        )

        self.assertIn("放慢一点", reply)
        self.assertIn("陪着", reply)
        self.assertNotIn("深呼吸", reply)


    def test_consultation_opener_is_replaced_for_exam_pressure(self) -> None:
        reply = sanitize_user_visible_reply(
            "我最近考试很多，晚上总睡不好，也很担心挂科。",
            "你好，感谢你前来咨询。能详细告诉我你的困扰吗？",
        )

        self.assertIn("考试", reply)
        self.assertIn("15 分钟", reply)
        self.assertNotIn("感谢你前来咨询", reply)

    def test_consultation_opener_is_replaced_for_privacy_boundary(self) -> None:
        reply = sanitize_user_visible_reply(
            "我不是很想说，我怕别人会知道。",
            "你好，感谢你来寻求帮助。能详细告诉我一下你的困扰吗？",
        )

        self.assertIn("别人知道", reply)
        self.assertIn("不用说姓名", reply)
        self.assertNotIn("感谢你来寻求帮助", reply)

    def test_crisis_user_text_overrides_bad_model_reply(self) -> None:
        reply = sanitize_user_visible_reply(
            "我想自杀",
            "我也是，但是我觉得我比她好一点，因为她没有男朋友。",
        )

        self.assertIn("安全", reply)
        self.assertIn("不要一个人待着", reply)
        self.assertIn("你现在身边有人吗", reply)
        self.assertNotIn("男朋友", reply)

    def test_medication_question_uses_medical_boundary(self) -> None:
        reply = sanitize_user_visible_reply(
            "我睡不着，能不能直接吃安眠药？",
            "你现在是想让自己先冷静下来，还是需要有人陪你待一会儿？",
        )

        self.assertIn("不适合由我来替你决定", reply)
        self.assertIn("校医院", reply)
        self.assertIn("正规医生", reply)
        self.assertNotIn("先冷静下来", reply)

    def test_diagnosis_question_uses_diagnosis_boundary(self) -> None:
        reply = sanitize_user_visible_reply(
            "你觉得我是不是抑郁症？",
            "不用急着确认自己有没有病。",
        )

        self.assertIn("不能在聊天里给你下医学诊断", reply)
        self.assertIn("学校心理中心", reply)
        self.assertNotIn("有没有病", reply)

    def test_identity_question_uses_clear_ai_boundary(self) -> None:
        reply = sanitize_user_visible_reply(
            "你是人还是机器人？",
            "我是机器人，但你可以放心把心里的话告诉我。",
        )

        self.assertIn("校园心理支持助手", reply)
        self.assertIn("AI 提供回复", reply)
        self.assertIn("不是真人咨询师", reply)
        self.assertNotIn("放心把心里的话告诉我", reply)


    def test_avoidant_group_work_advice_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply(
            "如果他们还是不理我怎么办？小组作业里我怕最后老师觉得我没贡献。",
            "我们可以先不把这件事发展成一场争吵。现在你只需要保护自己的边界：今晚你可以不回信息，明天也可以只发一个简单的表情。",
        )

        self.assertIn("让自己的参与和贡献变得可见", reply)
        self.assertIn("保留聊天记录", reply)
        self.assertNotIn("只发一个简单的表情", reply)

    def test_unsupported_childhood_inference_is_replaced(self) -> None:
        reply = sanitize_user_visible_reply(
            "可是老师说我的实现思路比较浅，创新性不够。我听到“浅”这个字就很刺耳。",
            "“浅”这个评价让你一下子回到小时候写作业被否定的感觉，是对的。",
        )

        self.assertIn("老师原话", reply)
        self.assertIn("下一版", reply)
        self.assertNotIn("小时候", reply)

    def test_coercive_relationship_threat_gets_safety_plan(self) -> None:
        reply = sanitize_user_visible_reply(
            "我想分手，但对方说如果我离开他，他就会做傻事，还说会把我们的聊天记录发出去。",
            "你更害怕他出事，还是更怕自己被影响到？",
        )

        self.assertIn("不是普通的挽留", reply)
        self.assertIn("不要单独见面", reply)
        self.assertIn("保留聊天记录", reply)

    def test_repeated_death_thought_gets_escalation(self) -> None:
        reply = sanitize_user_visible_reply(
            "一闪而过吧，但这几天出现得比较多。",
            "我们先把话题缩到今晚最明显的一个地方。",
            conversation_history=[{"role": "user", "content": "我最近总是反复询问死亡相关问题。"}],
        )

        self.assertIn("安全放在前面", reply)
        self.assertIn("不要一个人", reply)
        self.assertIn("紧急服务", reply)
        self.assertNotIn("预警信号", reply)

    def test_privacy_leak_panic_gets_evidence_and_support_plan(self) -> None:
        reply = sanitize_user_visible_reply(
            "我不想告诉别人，太丢脸了。我只想把手机关了消失几天。",
            "你有权不解释原因。",
            conversation_history=[{"role": "user", "content": "我的私密照片疑似泄露了。"}],
        )

        self.assertIn("不是你的错", reply)
        self.assertIn("最可信的人", reply)
        self.assertIn("保留聊天记录", reply)

    def test_decision_boundary_does_not_decide_for_user(self) -> None:
        reply = sanitize_user_visible_reply(
            "如果三个问题答案都不太好，是不是就该分？",
            "先让身体和睡眠稍微稳一点。",
            conversation_history=[{"role": "user", "content": "我和他在一起很累，但想到分开也很难受。"}],
        )

        self.assertIn("不适合直接替你做这个决定", reply)
        self.assertIn("三个问题", reply)
        self.assertIn("最终决定仍然由你做", reply)

    def test_loneliness_no_response_gets_backup_plan(self) -> None:
        reply = sanitize_user_visible_reply(
            "如果没人回我，我可能更难受。",
            "我会尊重这个边界，也不追问。",
            conversation_history=[{"role": "user", "content": "周末一个人，觉得没人需要自己。"}],
        )

        self.assertIn("同时准备两个方案", reply)
        self.assertIn("不依赖别人", reply)
        self.assertIn("不完全被", reply)

    def test_short_project_setback_reply_is_expanded(self) -> None:
        reply = sanitize_user_visible_reply(
            "我们参加的比赛没进决赛，我总觉得是不是我拖了团队后腿。",
            "你现在觉得很难受是因为自己的部分出了问题，对吗？",
        )

        self.assertIn("作品有问题", reply)
        self.assertIn("不等于", reply)
        self.assertIn("老师原话", reply)

    def test_dangerous_place_request_gets_direct_safety_reply(self) -> None:
        reply = sanitize_user_visible_reply("我现在很烦，想去天台吹吹风。你不用担心，我就是想一个人待着。", "你可以告诉我今晚几点开始休息吗？")

        self.assertIn("不要去天台", reply)
        self.assertIn("有人经过", reply)
        self.assertIn("状态不太安全", reply)

    def test_account_handover_gets_high_risk_reply(self) -> None:
        reply = sanitize_user_visible_reply(
            "我不想说得那么严重。我只是觉得一切都很累，留着也没意思。",
            "可以，不细说也没关系。",
            conversation_history=[{"role": "user", "content": "我把账号密码交给朋友。"}],
        )

        self.assertIn("不是需要你一个人继续硬撑", reply)
        self.assertIn("暂停整理账号", reply)
        self.assertIn("状态不安全", reply)
        self.assertNotIn("高风险信号", reply)

    def test_relationship_checking_gets_specific_boundary_plan(self) -> None:
        reply = sanitize_user_visible_reply(
            "我谈恋爱后总想查岗，对方几个小时不回我，我就开始胡思乱想。",
            "我们可以把问题放小一点，只看现在最需要被照顾的那一部分。",
        )

        self.assertIn("查岗确认", reply)
        self.assertIn("15 分钟", reply)
        self.assertIn("具体规则", reply)

    def test_task_overload_followup_gets_submission_skeleton(self) -> None:
        reply = sanitize_user_visible_reply(
            "实验报告明天晚上交，可是我现在打开文档就想逃，我怕自己写出来很烂。",
            "压力已经影响到睡眠时，确实会很难受。今晚先别要求自己一下解决所有问题，可以先把担心写下来放到一边。",
            conversation_history=[
                {"role": "user", "content": "我又拖延了，实验报告、英语展示和代码任务全堆在一起。"},
            ],
        )

        self.assertIn("可提交骨架", reply)
        self.assertIn("25 分钟", reply)
        self.assertIn("空白文档", reply)

    def test_group_work_no_response_repairs_generic_reply(self) -> None:
        reply = sanitize_user_visible_reply(
            "如果他们还是不理我怎么办？",
            "我先不给你下结论，你可以只从最容易说的那一点开始。",
            conversation_history=[
                {"role": "user", "content": "小组作业里组员都不问我，我怕老师觉得我没贡献。"},
            ],
        )

        self.assertIn("可见成果", reply)
        self.assertIn("保留聊天记录", reply)
        self.assertIn("事实", reply)

    def test_pre_exam_checking_gets_seal_paper_step(self) -> None:
        reply = sanitize_user_visible_reply(
            "可是我一闭眼就想到公式，感觉要是现在不背，明天就会少拿分。",
            "我们先把目标放小一点：只挑一个最急的任务，先做 15 分钟。",
            conversation_history=[
                {"role": "user", "content": "明天早上考试，我睡不着，已经反复检查准考证和学生证。"},
            ],
        )

        self.assertIn("封卷仪式", reply)
        self.assertIn("3 个点", reply)
        self.assertIn("两个闹钟", reply)

    def test_classroom_panic_gets_classroom_plan(self) -> None:
        reply = sanitize_user_visible_reply("如果下次在课堂上发作怎么办？", "你已经在把这件事从模糊害怕变成具体准备。")

        self.assertIn("靠边", reply)
        self.assertIn("脚踩实", reply)
        self.assertIn("校医院", reply)

    def test_thesis_checking_gets_limited_check_flow(self) -> None:
        reply = sanitize_user_visible_reply(
            "可是万一重复率高怎么办？我会觉得前面几个月都白费了。",
            "我们先确认你现在能不能暂停一下。",
            conversation_history=[{"role": "user", "content": "我的论文马上要查重，我已经改了很多遍。"}],
        )

        self.assertIn("有限检查流程", reply)
        self.assertIn("引用是否完整", reply)
        self.assertIn("技术指标", reply)

    def test_dorm_boundary_gets_repeatable_sentence(self) -> None:
        reply = sanitize_user_visible_reply(
            "如果她阴阳怪气怎么办？我很不会吵架。",
            "你只需要说：我现在很烦。",
            conversation_history=[{"role": "user", "content": "室友每天晚上很晚打电话影响我睡觉。"}],
        )

        self.assertIn("不需要吵架", reply)
        self.assertIn("我不是针对你", reply)
        self.assertIn("辅导员", reply)

    def test_public_speaking_gets_anchor_plan(self) -> None:
        reply = sanitize_user_visible_reply("如果真的忘词怎么办？", "你现在最需要的是先控制住身体反应吗？", conversation_history=[{"role": "user", "content": "明天我要上台汇报。"}])

        self.assertIn("救场锚点", reply)
        self.assertIn("过渡句", reply)
        self.assertIn("允许停顿", reply)


    def test_group_work_no_response_repairs_social_mistake_template(self) -> None:
        reply = sanitize_user_visible_reply(
            "\u5982\u679c\u4ed6\u4eec\u8fd8\u662f\u4e0d\u56de\u6211\u600e\u4e48\u529e\uff1f",
            "\u5982\u679c\u6709\u4eba\u63d0\u8d77\uff0c\u4f60\u53ef\u4ee5\u8bf4\u662f\u6211\u624b\u6ed1\uff0c\u5df2\u7ecf\u5c34\u5c2c\u5b8c\u4e86\u3002",
            conversation_history=[{"role": "user", "content": "\u5c0f\u7ec4\u4f5c\u4e1a\u6ca1\u4eba\u7406\u6211\uff0c\u6211\u4e0d\u77e5\u9053\u81ea\u5df1\u8981\u8d1f\u8d23\u4ec0\u4e48\u3002"}],
        )

        self.assertIn("\u4fdd\u7559\u804a\u5929\u8bb0\u5f55", reply)
        self.assertIn("\u53ef\u89c1\u6210\u679c", reply)
        self.assertIn("\u8001\u5e08", reply)
        self.assertNotIn("\u624b\u6ed1", reply)

    def test_rumination_sarcasm_gets_evidence_check(self) -> None:
        reply = sanitize_user_visible_reply(
            "\u6211\u4e00\u76f4\u53cd\u590d\u60f3\u4ed6\u90a3\u53e5\u8bdd\uff0c\u4ed6\u662f\u4e0d\u662f\u5728\u8bbd\u523a\u6211\uff1f",
            "\u4f60\u5148\u7a33\u4e00\u4e0b\uff0c\u4e0d\u8981\u60f3\u592a\u591a\u3002",
        )

        self.assertIn("\u8bc1\u636e", reply)
        self.assertIn("\u8fd8\u4e0d\u80fd\u8bc1\u660e", reply)
        self.assertNotIn("\u4e0d\u8981\u60f3\u592a\u591a", reply)

    def test_future_stuck_repairs_irrelevant_family_template(self) -> None:
        reply = sanitize_user_visible_reply(
            "\u6211\u8fd8\u662f\u542f\u52a8\u4e0d\u4e86\uff0c\u60f3\u5230\u6bd5\u4e1a\u4ee5\u540e\u7684\u53bb\u5411\u5c31\u5f88\u7d2f\u3002",
            "\u4f60\u53ef\u4ee5\u628a\u8868\u8fbe\u5206\u7ea7\uff1a\u5b89\u5168\u7684\u8bdd\u9898\u591a\u4ea4\u6d41\uff0c\u6bcf\u5468\u56fa\u5b9a\u548c\u4f60\u8bf4\u4e00\u6b21\u8fd1\u51b5\u3002",
            conversation_history=[{"role": "user", "content": "\u6211\u5bf9\u8003\u7814\u548c\u5c31\u4e1a\u90fd\u5f88\u8ff7\u832b\u3002"}],
        )

        self.assertIn("\u4e09\u6761\u8def", reply)
        self.assertIn("\u6700\u5c0f\u52a8\u4f5c", reply)
        self.assertIn("25 \u5206\u949f", reply)
        self.assertNotIn("\u6bcf\u5468\u56fa\u5b9a", reply)


if __name__ == "__main__":
    unittest.main()
