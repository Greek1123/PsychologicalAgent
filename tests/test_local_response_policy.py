from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.entropy import evaluate_psychological_entropy
from campus_support_agent.local_response_policy import maybe_build_local_support_plan
from campus_support_agent.safety import evaluate_text_risk


def _build_entropy(text: str):
    return evaluate_psychological_entropy(text, evaluate_text_risk(text))


class LocalResponsePolicyTests(unittest.TestCase):
    def test_privacy_concern_uses_local_policy(self) -> None:
        text = "我不是很想说，我害怕别人会知道。"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertEqual(result.info.policy_name, "privacy_concern")
        self.assertEqual(result.info.policy_stage, "rapport_boundary")
        self.assertIn("安全感", plan.summary)
        self.assertTrue(any("安全" in item or "细节" in item for item in plan.immediate_support))

    def test_exam_anxiety_uses_local_policy(self) -> None:
        text = "我最近很怕考试挂科，晚上总睡不好。"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertIn("睡眠", plan.summary)
        self.assertTrue(any("考试" in item or "复习" in item for item in plan.immediate_support))

    def test_late_night_distress_uses_local_policy(self) -> None:
        text = "现在凌晨了，我还是睡不着。"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertIn("睡不着", plan.summary)
        self.assertTrue(any("今晚" in item or "慢" in item for item in plan.self_regulation + plan.immediate_support))

    def test_dorm_conflict_uses_local_policy(self) -> None:
        text = "我今天本来还好，一回宿舍就烦。"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertIn("宿舍", plan.summary)

    def test_social_withdrawal_uses_boundary_policy(self) -> None:
        text = "我这几天不太想见人，也不太想说话。"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertTrue(any("不追问" in item or "轻的方式" in item for item in plan.immediate_support))

    def test_exam_anxiety_uses_local_reply(self) -> None:
        text = "我最近很怕考试挂科，晚上总睡不好。"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertTrue(any("考试" in item or "复习" in item or "挂科" in item for item in [plan.summary, *plan.immediate_support, *plan.follow_up]))

    def test_late_night_distress_uses_local_reply(self) -> None:
        text = "现在凌晨了，我还是睡不着。"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertTrue(any("今晚" in item or "睡不着" in item or "放慢" in item for item in [plan.summary, *plan.immediate_support, *plan.follow_up]))

    def test_social_isolation_uses_local_reply(self) -> None:
        text = "我觉得自己被孤立了，也越来越不想见人。"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertTrue(any("孤" in item or "排挤" in item or "外面" in item for item in [plan.summary, *plan.immediate_support, *plan.follow_up]))

    def test_exhaustion_withdrawal_uses_local_reply(self) -> None:
        text = "我连续几天都很累，什么都不想做，也不太想见人。"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertTrue(any("累" in item or "没电" in item or "不想做" in item for item in [plan.summary, *plan.immediate_support, *plan.follow_up]))

    def test_repetitive_help_seeking_uses_local_reply(self) -> None:
        text = "我又来了，还是很难受。"
        result = maybe_build_local_support_plan(
            text,
            entropy=_build_entropy(text),
            conversation_history=[{"role": "user", "content": "我昨天也在说这个事情，还是睡不着。"}],
        )
        self.assertIsNotNone(result)
        _, plan = result
        self.assertTrue(any("反复" in item or "上一次" in item or "变化" in item for item in [plan.summary, *plan.immediate_support, *plan.follow_up]))


    def test_authority_pressure_uses_local_reply(self) -> None:
        text = "\u8001\u5e08\u4e00\u76f4\u50ac\u6211\uff0c\u7238\u5988\u4e5f\u8bf4\u6211\u4e0d\u591f\u597d\uff0c\u6211\u771f\u7684\u5feb\u625b\u4e0d\u4f4f\u4e86\u3002"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertTrue(any("\u538b\u8feb" in item or "\u50ac" in item or "\u8981\u6c42" in item for item in [plan.summary, *plan.immediate_support, *plan.follow_up]))

    def test_future_panic_uses_local_reply(self) -> None:
        text = "\u6211\u4e00\u60f3\u5230\u6bd5\u4e1a\u4ee5\u540e\u8981\u600e\u4e48\u529e\uff0c\u5c31\u5bf9\u672a\u6765\u5f88\u614c\u3002"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertTrue(any("\u672a\u6765" in item or "\u614c" in item or "\u773c\u524d" in item for item in [plan.summary, *plan.immediate_support, *plan.follow_up]))

    def test_self_blame_uses_local_reply(self) -> None:
        text = "\u6211\u603b\u89c9\u5f97\u90fd\u662f\u6211\u7684\u9519\uff0c\u662f\u4e0d\u662f\u6211\u771f\u7684\u5f88\u5dee\u52b2\u3002"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertTrue(any("\u81ea\u8d23" in item or "\u90fd\u662f\u6211\u7684\u9519" in item or "\u602a\u81ea\u5df1" in item for item in [plan.summary, *plan.immediate_support, *plan.follow_up]))

    def test_sleep_appetite_drift_uses_local_reply(self) -> None:
        text = "\u6211\u8fd9\u51e0\u5929\u4e00\u76f4\u7761\u4e0d\u597d\uff0c\u4e5f\u5403\u4e0d\u4e0b\u4e1c\u897f\u3002"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertEqual(result.info.policy_name, "sleep_appetite_drift")
        self.assertEqual(result.info.policy_stage, "escalation_watch")
        self.assertTrue(any("\u7761" in item or "\u5403" in item or "\u8282\u5f8b" in item for item in [plan.summary, *plan.immediate_support, *plan.follow_up]))

    def test_helplessness_escalation_uses_local_reply(self) -> None:
        text = "\u6211\u771f\u7684\u4e0d\u77e5\u9053\u8fd8\u80fd\u600e\u4e48\u529e\uff0c\u611f\u89c9\u4ec0\u4e48\u90fd\u6ca1\u7528\u4e86\u3002"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        _, plan = result
        self.assertTrue(any("\u65e0\u52a9" in item or "\u6ca1\u6709\u7528" in item or "\u6491\u4e0d\u4f4f" in item for item in [plan.summary, *plan.immediate_support, *plan.follow_up]))

    def test_rising_emotional_spiral_uses_local_reply(self) -> None:
        text = "\u6211\u8fd9\u51e0\u5929\u8d8a\u6765\u8d8a\u6162\uff0c\u611f\u89c9\u6bd4\u524d\u4e24\u5929\u66f4\u96be\u53d7\u4e86\u3002"
        result = maybe_build_local_support_plan(
            text,
            entropy=_build_entropy(text),
            conversation_history=[{"role": "user", "content": "\u6211\u524d\u4e24\u5929\u5df2\u7ecf\u5f88\u6162\u4e86\uff0c\u800c\u4e14\u665a\u4e0a\u7761\u4e0d\u597d\u3002"}],
        )
        self.assertIsNotNone(result)
        _, plan = result
        self.assertTrue(any("\u52a0\u91cd" in item or "\u8d8a\u6765\u8d8a" in item or "\u53d8\u5316" in item for item in [plan.summary, *plan.immediate_support, *plan.follow_up]))

    def test_escalation_watch_policy_carries_hint(self) -> None:
        text = "\u6211\u8fd9\u51e0\u5929\u4e00\u76f4\u7761\u4e0d\u597d\uff0c\u4e5f\u5403\u4e0d\u4e0b\u4e1c\u897f\u3002"
        result = maybe_build_local_support_plan(text, entropy=_build_entropy(text), conversation_history=[])
        self.assertIsNotNone(result)
        self.assertEqual(result.info.policy_name, "sleep_appetite_drift")
        self.assertEqual(result.info.policy_stage, "escalation_watch")
        self.assertEqual(result.info.escalation_hint, "consider_sleep_and_health_followup")


if __name__ == "__main__":
    unittest.main()
