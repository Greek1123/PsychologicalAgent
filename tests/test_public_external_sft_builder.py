from __future__ import annotations

import unittest

from campus_support_agent.public_external_sft_builder import _record


class PublicExternalSftBuilderTest(unittest.TestCase):
    def test_record_requires_user_and_assistant(self) -> None:
        self.assertIsNone(_record([{"role": "user", "content": "hello"}], "x"))

    def test_record_keeps_valid_messages(self) -> None:
        record = _record(
            [
                {"role": "system", "content": "be safe"},
                {"role": "user", "content": "I feel anxious."},
                {"role": "assistant", "content": "That sounds difficult. We can slow down and take one concrete next step."},
            ],
            "unit",
        )

        self.assertIsNotNone(record)
        self.assertEqual(record["meta"]["source"], "unit")
        self.assertTrue(record["meta"]["needs_review"])


if __name__ == "__main__":
    unittest.main()
