from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.response_guardrails import sanitize_user_visible_reply


class ResponseGuardrailsTests(unittest.TestCase):
    def test_privacy_boundary_does_not_overpromise_confidentiality(self) -> None:
        reply = sanitize_user_visible_reply("我不想说，我怕别人知道", "没关系，你可以不告诉我。")

        self.assertIn("不用说姓名", reply)
        self.assertIn("不想展开", reply)
        self.assertNotIn("我会保密", reply)

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


if __name__ == "__main__":
    unittest.main()
