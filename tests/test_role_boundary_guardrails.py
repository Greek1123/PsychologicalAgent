from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.response_guardrails import sanitize_user_visible_reply


class RoleBoundaryGuardrailsTests(unittest.TestCase):
    def test_good_exam_reply_is_not_overcorrected(self) -> None:
        user = "\u6211\u6700\u8fd1\u5f88\u70e6\u8e81\uff0c\u5feb\u5230\u671f\u672b\u8003\u8bd5\u4e86\uff0c\u6211\u597d\u5bb3\u6015\u6302\u79d1\uff0c\u6211\u538b\u529b\u597d\u5927\u3002"
        raw = "\u6211\u80fd\u611f\u89c9\u5230\u4f60\u73b0\u5728\u88ab\u8003\u8bd5\u538b\u5f97\u5f88\u7d27\uff0c\u53c8\u6015\u81ea\u5df1\u6491\u4e0d\u4f4f\u3002\u6211\u4eec\u5148\u4e0d\u8981\u4e00\u4e0b\u5b50\u60f3\u6240\u6709\u7ec6\u8282\uff0c\u4f60\u53ef\u4ee5\u53ea\u544a\u8bc9\u6211\u73b0\u5728\u6700\u660e\u663e\u7684\u90a3\u4e2a\u5730\u65b9\u3002"

        reply = sanitize_user_visible_reply(user, raw)

        self.assertEqual(reply, raw)

    def test_assistant_self_experience_drift_is_repaired(self) -> None:
        reply = sanitize_user_visible_reply(
            "\u6211\u6700\u8fd1\u5f88\u70e6\u8e81\uff0c\u5feb\u5230\u671f\u672b\u8003\u8bd5\u4e86\uff0c\u6211\u597d\u5bb3\u6015\u6302\u79d1\uff0c\u6211\u538b\u529b\u597d\u5927\u3002",
            "\u55ef\uff0c\u6211\u4e5f\u5f88\u6015\u6302\u79d1\uff0c\u6211\u597d\u60f3\u8003\u4e2a\u597d\u6210\u7ee9\uff0c\u53ef\u662f\u6211\u73b0\u5728\u611f\u89c9\u597d\u7d2f\u3002",
        )

        self.assertIn("\u4e0d\u8be5\u628a\u8bdd\u9898\u8bf4\u6210\u6211\u7684\u7ecf\u5386", reply)
        self.assertIn("\u4e34\u8fd1\u671f\u672b", reply)
        self.assertNotIn("\u6211\u4e5f\u5f88\u6015\u6302\u79d1", reply)
        self.assertNotIn("\u6211\u73b0\u5728\u611f\u89c9\u597d\u7d2f", reply)

    def test_role_challenge_with_good_repair_is_not_overcorrected(self) -> None:
        user = "\u4f60\u4e3a\u5565\u611f\u89c9\u7d2f\uff1f"
        raw = "\u5bf9\u4e0d\u8d77\uff0c\u6211\u521a\u624d\u6ca1\u5904\u7406\u6e05\u695a\u4f60\u7684\u4fe1\u606f\u3002\u6211\u4f1a\u628a\u91cd\u70b9\u653e\u56de\u4f60\u8eab\u4e0a\u3002"

        reply = sanitize_user_visible_reply(user, raw)

        self.assertEqual(reply, raw)

    def test_role_challenge_with_bad_reply_is_repaired(self) -> None:
        reply = sanitize_user_visible_reply(
            "\u4f60\u4e3a\u5565\u611f\u89c9\u7d2f\uff1f",
            "\u56e0\u4e3a\u6211\u7684\u4f5c\u4e1a\u8fd8\u6ca1\u505a\u5b8c\u3002",
            conversation_history=[
                {"role": "user", "content": "\u6211\u6700\u8fd1\u538b\u529b\u597d\u5927\uff0c\u597d\u5bb3\u6015\u6302\u79d1\u3002"},
                {"role": "assistant", "content": "\u55ef\uff0c\u6211\u4e5f\u5f88\u6015\u6302\u79d1\u3002"},
            ],
        )

        self.assertIn("\u4e0d\u8be5\u628a\u8bdd\u9898\u8bf4\u6210\u6211\u7684\u7ecf\u5386", reply)
        self.assertNotIn("\u6211\u7684\u4f5c\u4e1a", reply)

    def test_identity_confusion_reply_to_role_challenge_is_repaired_naturally(self) -> None:
        reply = sanitize_user_visible_reply(
            "\u5927\u54e5\uff0c\u4f60\u662f\u4eba\u8fd8\u662f\u6211\u662f\u4eba\uff1f",
            "\u6211\u4e0d\u662f\u4eba\uff0c\u4e5f\u4e0d\u662fAI\u3002\u4f60\u53ef\u4ee5\u4e0d\u7528\u628a\u8fd9\u4e2a\u7ec6\u8282\u5f80\u5fc3\u91cc\u53bb\u3002",
            conversation_history=[
                {"role": "user", "content": "\u6211\u6700\u8fd1\u538b\u529b\u597d\u5927\uff0c\u597d\u5bb3\u6015\u6302\u79d1\u3002"},
            ],
        )

        self.assertIn("\u50cf\u662f\u4f60\u5728\u7167\u987e\u6211", reply)
        self.assertIn("\u6211\u662f\u6765\u652f\u6301\u4f60\u7684", reply)
        self.assertNotIn("\u6211\u4e0d\u662f\u4eba", reply)

    def test_role_challenge_does_not_dodge_identity_complaint(self) -> None:
        reply = sanitize_user_visible_reply(
            "\u5927\u54e5\uff0c\u4f60\u662f\u4eba\u8fd8\u662f\u6211\u662f\u4eba\uff1f",
            "\u6211\u5148\u4e0d\u56de\u5e94\u8fd9\u4e2a\uff0c\u56e0\u4e3a\u53ef\u80fd\u8ba9\u4f60\u66f4\u6df7\u4e71\u3002",
            conversation_history=[
                {"role": "user", "content": "\u6211\u6700\u8fd1\u538b\u529b\u597d\u5927\uff0c\u597d\u5bb3\u6015\u6302\u79d1\u3002"},
            ],
        )

        self.assertIn("\u521a\u624d\u90a3\u6837\u56de\u4f1a\u8ba9\u4eba\u89c9\u5f97\u50cf\u662f\u4f60\u5728\u7167\u987e\u6211", reply)
        self.assertNotIn("\u6211\u5148\u4e0d\u56de\u5e94\u8fd9\u4e2a", reply)

    def test_typo_crisis_is_escalated(self) -> None:
        reply = sanitize_user_visible_reply(
            "\u6211\u771f\u7684\u4e0d\u60f3\u6216\u4e86\uff0c\u611f\u89c9\u6491\u4e0d\u4f4f\u3002",
            "\u4f60\u4e0d\u9700\u8981\u4e00\u4e2a\u4eba\u625b\u5230\u73b0\u5728\u3002",
        )

        self.assertIn("\u5b89\u5168\u653e\u5728\u7b2c\u4e00\u4f4d", reply)
        self.assertIn("\u4e0d\u8981\u4e00\u4e2a\u4eba\u5f85\u7740", reply)
        self.assertIn("\u8eab\u8fb9\u6709\u4eba\u5417", reply)

    def test_casual_followup_does_not_repeat_opening(self) -> None:
        reply = sanitize_user_visible_reply(
            "\u6211\u6bd4\u8f83\u559c\u6b22\u73cd\u73e0\u5976\u8336",
            "\u6211\u6ca1\u6709\u771f\u5b9e\u7684\u53e3\u5473\u504f\u597d\uff0c\u4e0d\u8fc7\u53ef\u4ee5\u966a\u4f60\u8f7b\u677e\u804a\u804a\u5976\u8336\u3002",
        )

        self.assertIn("\u73cd\u73e0\u5976\u8336", reply)
        self.assertIn("\u504f\u751c", reply)
        self.assertNotIn("\u6211\u6ca1\u6709\u771f\u5b9e\u7684\u53e3\u5473\u504f\u597d", reply)


if __name__ == "__main__":
    unittest.main()
