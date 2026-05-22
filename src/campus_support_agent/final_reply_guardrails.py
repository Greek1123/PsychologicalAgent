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
