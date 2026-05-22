from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_support_agent.schemas import DynamicAdjustment, InterventionStrategy, StateProfile, SupportPlan
from campus_support_agent.strategy_execution import (
    apply_dynamic_adjustment_to_plan,
    apply_intervention_strategy_to_plan,
    apply_session_continuity_to_plan,
)


def _plan() -> SupportPlan:
    return SupportPlan(
        summary="generic",
        immediate_support=["old support"],
        campus_actions=[],
        self_regulation=[],
        follow_up=["old follow up"],
    )


def _profile(**kwargs) -> StateProfile:
    values = {
        "primary_state": "general_support",
        "intensity": 3,
        "confidence": 0.7,
        "stress_domains": [],
        "emotion_signals": [],
        "body_signals": [],
        "cognitive_signals": [],
        "social_signals": [],
        "boundary_flags": [],
        "risk_signals": [],
        "weak_input_detected": False,
        "noisy_input_detected": False,
        "recommended_focus": "supportive_listening",
        "evidence": [],
    }
    values.update(kwargs)
    return StateProfile(**values)


def _strategy(strategy_id: str) -> InterventionStrategy:
    return InterventionStrategy(
        strategy_id=strategy_id,
        priority="medium",
        response_mode="test",
        user_visible_goal="test",
        hidden_clinical_goal="test",
        should_ask_question=True,
        max_questions=1,
        suggested_opening="test",
        next_step="test",
    )


def _adjustment(action: str) -> DynamicAdjustment:
    return DynamicAdjustment(
        adjustment_id=f"test:{action}",
        stability_state="test",
        action=action,
        intensity_shift="increase",
        trend_direction="up",
        trend_delta=8,
        should_modify_strategy=True,
        should_refer=False,
        review_window_hours=24,
        next_focus="test",
        reasons=["test"],
    )


class StrategyExecutionTests(unittest.TestCase):
    def test_privacy_strategy_rewrites_visible_plan(self) -> None:
        plan = apply_intervention_strategy_to_plan(
            _plan(),
            strategy=_strategy("privacy_reassurance"),
            state_profile=_profile(),
        )

        self.assertIn("别人知道", plan.summary)
        self.assertIn("不会逼你", plan.immediate_support[0])
        self.assertIn("陪你", plan.follow_up[0])

    def test_sleep_strategy_prioritizes_sleep_stabilization(self) -> None:
        plan = apply_intervention_strategy_to_plan(
            _plan(),
            strategy=_strategy("sleep_stabilization"),
            state_profile=_profile(body_signals=["sleep_disruption"]),
        )

        self.assertIn("睡不着", plan.summary)
        self.assertIn("十分钟", plan.immediate_support[0])
        self.assertIn("考试压力", plan.follow_up[0])

    def test_grounding_strategy_uses_academic_context(self) -> None:
        plan = apply_intervention_strategy_to_plan(
            _plan(),
            strategy=_strategy("grounding_small_step"),
            state_profile=_profile(stress_domains=["academic"]),
        )

        self.assertIn("考试", plan.summary)
        self.assertIn("十到十五分钟", plan.immediate_support[0])

    def test_group_work_strategy_adds_visibility_actions(self) -> None:
        plan = apply_intervention_strategy_to_plan(
            _plan(),
            strategy=_strategy("group_work_visibility"),
            state_profile=_profile(primary_state="group_work_marginalization"),
        )

        self.assertIn("贡献可能被看不见", plan.summary)
        self.assertIn("具体消息", plan.immediate_support[0])
        self.assertIn("保留聊天记录", plan.follow_up[0])

    def test_game_avoidance_strategy_names_escape_loop(self) -> None:
        plan = apply_intervention_strategy_to_plan(
            _plan(),
            strategy=_strategy("escape_loop_interruption"),
            state_profile=_profile(primary_state="game_avoidance_loop"),
        )

        self.assertIn("避难所", plan.summary)
        self.assertIn("中断点", plan.immediate_support[0])
        self.assertIn("行为计划", plan.campus_actions[0])

    def test_performance_strategy_adds_anchor_plan(self) -> None:
        plan = apply_intervention_strategy_to_plan(
            _plan(),
            strategy=_strategy("performance_grounding"),
            state_profile=_profile(primary_state="public_speaking_panic"),
        )

        self.assertIn("汇报", plan.summary)
        self.assertIn("救场锚点", plan.immediate_support[0])
        self.assertIn("脚踩实地面", plan.self_regulation[0])

    def test_family_boundary_strategy_limits_middleman_role(self) -> None:
        plan = apply_intervention_strategy_to_plan(
            _plan(),
            strategy=_strategy("family_boundary_sustainability"),
            state_profile=_profile(primary_state="family_middleman_stress"),
        )

        self.assertIn("情绪中间人", plan.summary)
        self.assertIn("我不能每天听你骂爸爸", plan.immediate_support[0])
        self.assertIn("边界方案", plan.campus_actions[0])

    def test_grief_strategy_reduces_total_self_blame(self) -> None:
        plan = apply_intervention_strategy_to_plan(
            _plan(),
            strategy=_strategy("grief_without_self_blame"),
            state_profile=_profile(primary_state="pet_grief"),
        )

        self.assertIn("很多事情并不完全由你控制", plan.summary)
        self.assertIn("三个你和它在一起的片段", plan.immediate_support[0])
        self.assertIn("是不是全怪我", plan.follow_up[0])

    def test_safety_reporting_strategy_makes_reporting_practical(self) -> None:
        plan = apply_intervention_strategy_to_plan(
            _plan(),
            strategy=_strategy("safety_reporting_without_blame"),
            state_profile=_profile(primary_state="campus_safety_fear"),
        )

        self.assertIn("报告安全隐患", plan.summary)
        self.assertIn("时间某路段", plan.immediate_support[0])
        self.assertIn("不要独自走", plan.campus_actions[0])

    def test_dynamic_rising_watch_lowers_pressure_before_reply(self) -> None:
        plan = apply_dynamic_adjustment_to_plan(
            _plan(),
            dynamic_adjustment=_adjustment("soften_and_stabilize"),
        )

        self.assertNotEqual(plan.summary, "generic")
        self.assertNotEqual(plan.immediate_support[0], "old support")
        self.assertNotEqual(plan.follow_up[0], "old follow up")

    def test_dynamic_improving_keeps_strategy_light(self) -> None:
        plan = apply_dynamic_adjustment_to_plan(
            _plan(),
            dynamic_adjustment=_adjustment("maintain_and_consolidate"),
        )

        self.assertEqual(plan.summary, "generic")
        self.assertNotEqual(plan.follow_up[0], "old follow up")

    def test_continuity_boundary_stage_reduces_detail_pressure(self) -> None:
        plan = apply_session_continuity_to_plan(
            _plan(),
            continuity_summary={
                "dialogue_stage": "boundary_building",
                "avoid_next_turn": ["不要追问隐私细节"],
                "recommended_next_moves": ["先明确不会逼用户细说"],
            },
        )

        self.assertIn("尊重你的边界", plan.summary)
        self.assertIn("不用解释原因", plan.follow_up[0])

    def test_continuity_deteriorating_stage_prioritizes_stabilization(self) -> None:
        plan = apply_session_continuity_to_plan(
            _plan(),
            continuity_summary={
                "dialogue_stage": "deteriorating_watch",
                "recent_user_needs": ["需要先被接住情绪，而不是立刻被分析"],
            },
        )

        self.assertIn("不急着让你分析原因", plan.summary)
        self.assertIn("这一分钟", plan.immediate_support[0])


if __name__ == "__main__":
    unittest.main()
