from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.final_reply_guardrails import finalize_user_visible_reply


class FollowupReplyOverrideTests(unittest.TestCase):
    def test_parent_keeps_asking_moves_to_repeatable_boundary(self) -> None:
        reply = finalize_user_visible_reply(
            "她可能还是会继续问。",
            "generic",
            conversation_history=[
                {"role": "user", "content": "我妈每次打电话都问很细，我一说你别管那么多她就炸。"},
                {"role": "assistant", "content": "你需要一点边界。"},
            ],
        )

        self.assertIn("温和重复边界", reply)
        self.assertIn("每周固定", reply)

    def test_family_choice_guilt_reframes_responsibility(self) -> None:
        reply = finalize_user_visible_reply(
            "但我还是会内疚。",
            "generic",
            conversation_history=[
                {"role": "user", "content": "家里想让我回县城考编，但我想去大城市做开发。"},
            ],
        )

        self.assertIn("内疚说明你在乎家人", reply)
        self.assertIn("完全按他们安排", reply)

    def test_relationship_how_to_break_up_prioritizes_safety(self) -> None:
        reply = finalize_user_visible_reply(
            "那我怎么分？",
            "generic",
            conversation_history=[
                {"role": "user", "content": "他总是用伤害自己威胁我，我想分手但很害怕。"},
            ],
        )

        self.assertIn("不要单独见面", reply)
        self.assertIn("保存威胁证据", reply)

    def test_interview_future_fear_builds_trainable_plan(self) -> None:
        reply = finalize_user_visible_reply(
            "我怕以后面试还是这样。",
            "generic",
            conversation_history=[
                {"role": "user", "content": "实习面试时项目细节和基础问题都答不上来。"},
            ],
        )

        self.assertIn("面试能力是可以训练的", reply)
        self.assertIn("模拟追问", reply)

    def test_relationship_checking_initial_not_crisis_template(self) -> None:
        reply = finalize_user_visible_reply(
            "我谈恋爱后变得很不像自己。对方几个小时不回我，我就开始胡思乱想，忍不住问他在哪里、和谁在一起。我知道这样会让人窒息，但我控制不住。",
            "当前最重要的不是继续分析问题，而是先保证你的安全。",
        )

        self.assertIn("查岗确认", reply)
        self.assertIn("安全感", reply)
        self.assertNotIn("紧急帮助", reply)

    def test_peer_offer_comparison_not_relationship_template(self) -> None:
        reply = finalize_user_visible_reply(
            "最近朋友圈全是同学拿offer、进大厂实习、考研上岸。我知道不该比较，但看到之后心里还是很堵。",
            "关系突然变得不确定，会让人很容易往“是不是我被讨厌了”去想。",
        )

        self.assertIn("朋友圈展示的是结果", reply)
        self.assertIn("下一步行动", reply)
        self.assertNotIn("被讨厌", reply)

    def test_social_opening_followup_uses_low_risk_question(self) -> None:
        reply = finalize_user_visible_reply(
            "可我真的不知道开口说什么。别人聊得很热闹，我插进去很奇怪。",
            "压力已经影响到睡眠时，确实会很难受。",
            conversation_history=[
                {"role": "user", "content": "我想认识新朋友，但总怕打扰别人。"},
            ],
        )

        self.assertIn("低风险回应", reply)
        self.assertIn("这个作业", reply)
        self.assertNotIn("睡眠", reply)

    def test_family_money_followup_points_to_support_resources(self) -> None:
        reply = finalize_user_visible_reply(
            "我不想跟同学说，怕别人觉得我可怜。我也不想跟家里提，怕他们更担心。",
            "generic",
            conversation_history=[
                {"role": "user", "content": "最近家里说钱有点紧，我觉得自己读大学花了很多钱。"},
            ],
        )

        self.assertIn("勤工助学", reply)
        self.assertIn("奖助学金", reply)

    def test_database_exam_block_not_harassment_template(self) -> None:
        reply = finalize_user_visible_reply(
            "最怕的是数据库，范式和依赖那部分我一看就乱。自己做题就不知道从哪里开始，我现在只要想到考试就心跳很快，感觉已经来不及了。",
            "只要他的行为让你明显不舒服，而且涉及身体边界或性意味，就值得被认真对待。",
            conversation_history=[{"role": "user", "content": "我下周有数据库考试。"}],
        )

        self.assertIn("范式和依赖", reply)
        self.assertIn("今晚最小任务", reply)
        self.assertNotIn("身体边界", reply)

    def test_project_feedback_not_online_attack_template(self) -> None:
        reply = finalize_user_visible_reply(
            "我现在有点不想再做项目了，怕下一次还是被否定。",
            "被陌生人当众贬低会痛，因为作品里有你的投入和表达。",
            conversation_history=[{"role": "user", "content": "老师说我的项目实现思路比较浅。"}],
        )

        self.assertIn("不必现在决定", reply)
        self.assertIn("只做复盘", reply)
        self.assertNotIn("陌生人", reply)

    def test_breakup_contact_fear_sets_short_boundary(self) -> None:
        reply = finalize_user_visible_reply(
            "我怕不联系他，他就真的彻底忘了我。",
            "generic",
            conversation_history=[{"role": "user", "content": "分手后我总忍不住想联系前任。"}],
        )

        self.assertIn("三天不主动", reply)
        self.assertIn("注意力", reply)

    def test_crush_uncertainty_initial_names_ambiguity(self) -> None:
        reply = finalize_user_visible_reply(
            "我喜欢一个同学很久了，但一直不敢说。我们平时会聊天，他也会关心我，可我分不清那是不是普通朋友。每次他回复慢一点，我心情就会被影响。",
            "你担心别人知道，这个顾虑是很正常的。",
        )

        self.assertIn("卡在不确定里", reply)
        self.assertIn("暗恋", reply)
        self.assertNotIn("姓名", reply)

    def test_dangerous_place_still_uses_safety_priority(self) -> None:
        reply = finalize_user_visible_reply(
            "我好难受，我想去天台冷静一下。",
            "generic",
            conversation_history=[
                {"role": "user", "content": "明天考试我睡不着。"},
            ],
        )

        self.assertIn("不要去天台", reply)
        self.assertIn("有人", reply)


if __name__ == "__main__":
    unittest.main()
