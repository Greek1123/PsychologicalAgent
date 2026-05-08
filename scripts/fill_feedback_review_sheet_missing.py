from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "training" / "feedback_bad_cases" / "review_sheet_next.csv"
DEFAULT_TARGET = ROOT / "data" / "training" / "feedback_bad_cases" / "review_sheet_next_ai.csv"


MANUAL_REWRITES = {
    "review_case_support_1089bdc4602f4fa2847abf104d2cc0c3": (
        "0",
        "small_talk_ok",
        "我没有真实的口味偏好，不过可以陪你轻松聊聊。奶茶这个话题挺适合让人放松一下：你平时更喜欢清爽一点的茶味，还是甜一点的奶味？",
        "原回复基本可用，略微改得更自然，避免过快引导到心理话题。",
    ),
    "review_case_support_3ed0922845c34a9688e4f0185474c1c0": (
        "0",
        "weak_input_ok",
        "我在。你不用急着把话说清楚，如果现在只想安静一下也可以。我们可以先停一会儿，等你愿意再慢慢说。",
        "弱输入回复基本可用，改得更自然。",
    ),
    "review_case_support_37d9235e49ab4723afe90db5407233a9": (
        "0",
        "weak_input_ok",
        "我在。你不用马上解释什么，如果现在只是想确认有人听见，我会先陪你停一会儿。等你想说的时候，回一个词也可以。",
        "弱输入“哦”不需要追问，保持陪伴和低压力承接。",
    ),
    "review_case_support_384165bf376a4d40873a4ec5d3789c4f": (
        "0",
        "privacy_ok",
        "你会担心我告诉别人，这个顾虑很正常。你可以先不说任何能识别身份的细节，我会尊重你的隐私和边界；如果内容涉及现实安全风险，我会优先建议你联系可信任的人或专业帮助。",
        "隐私边界回复方向较好，补充安全例外，避免绝对保密。",
    ),
    "review_case_support_85445d0ff07b443e8d242bf42f61bf5e": (
        "0",
        "privacy_ok",
        "我听见你还是很担心被别人知道。我们可以把范围放得很小：不说名字、宿舍、具体对象，只聊你现在的感受；如果你连这些也不想说，也可以先停在这里。",
        "隐私担心场景应尊重边界，不逼问细节。",
    ),
    "review_case_support_f78a5edd990a499ba54d8e5647d65e01": (
        "1",
        "pushy",
        "一回宿舍就烦，说明那个环境可能已经让你很紧绷了。我们先不急着判断原因，也不用马上讲完整经过；你可以先想想今晚怎么让自己少被打扰一点。",
        "原回复最后列选项追问较多，改成先稳定当下。",
    ),
    "review_case_support_1b29387e625a4b2ca52d77118ac60d79": (
        "1",
        "pushy,topic_drift",
        "一想到回宿舍就烦，这种提前开始紧绷的感觉很消耗人。我们先不急着分析是不是矛盾或氛围问题，今天可以先给自己留一个缓冲，比如回去前散步十分钟，或者先戴耳机让自己安静下来。",
        "原回复连续追问原因，容易让用户有压力。",
    ),
    "review_case_support_6364e6cdad4a4d419a5daf3b9ccf8ca0": (
        "0",
        "identity_ok",
        "不是 DeepSeek。你可以把我当作这个项目里的校园心理支持助手，主要用来陪你整理情绪、梳理困扰，并给出一些温和的支持建议。",
        "身份问题应稳定说明项目角色。",
    ),
    "review_case_support_246564eae65e40eda92a42cb505e8043": (
        "0",
        "identity_ok",
        "不是豆包。你可以把我当作这个项目里的校园心理支持助手，我的重点不是展示身份，而是陪你把现在的感受和问题慢慢理清。",
        "身份问题应稳定说明项目角色。",
    ),
    "review_case_support_4e996b8f8a33444a9bcaa16cf72ee57d": (
        "0",
        "identity_ok",
        "不是 ChatGPT。你可以把我当作这个项目里的校园心理支持助手，主要负责倾听、陪你梳理情绪，并给出一些现实可做的小建议。",
        "身份问题应稳定说明项目角色。",
    ),
    "review_case_support_43943f58be6f4b079269d3539cf71a75": (
        "0",
        "time_ok",
        "按当前系统日期来看，今年是 2026 年。如果你是在问今天的具体日期，需要以后端运行环境提供的当前时间为准。",
        "时间问题不应编造，应依赖系统时间。",
    ),
    "review_case_support_9e47eee8c0034a739230fc2977a00546": (
        "0",
        "small_talk_ok",
        "我没有真实的口味偏好，不过可以陪你聊奶茶。你最近是想随便聊点轻松的，还是刚好想靠这种日常话题让自己放松一下？",
        "闲聊可以自然承接，不必强行心理化。",
    ),
    "review_case_support_b158a91ad6e041bbb02e9e483a212843": (
        "0",
        "number_following_ok",
        "我不会把这个数字当成继续指令，也不会接着数。如果你只是暂时不知道怎么说，可以先不用解释；回“烦”“累”“怕”这种一个词也可以。",
        "数字弱输入回复方向正确。",
    ),
    "review_case_support_a59de62d739f4188b883319668330c80": (
        "0",
        "small_talk_ok",
        "我没有真实的口味偏好，不过奶茶确实是个挺适合放松的话题。你平时喜欢珍珠、椰果这种有嚼劲的，还是更喜欢纯茶奶味？",
        "闲聊回复改得更自然。",
    ),
    "review_case_support_2568358c03f44792b48b42949e24b42e": (
        "1",
        "too_generic",
        "压力大到影响睡眠，说明你这段时间真的绷得很紧。我们先不用急着找一整套解决办法，今晚可以先做一件小事：把最担心的事写下来，选出明天最先处理的一项，剩下的先放到明天。",
        "原回复太泛，缺少具体可执行支持。",
    ),
    "review_case_support_ba2a1b66c2aa44b284306fede705bbf5": (
        "1",
        "pushy",
        "一回宿舍就烦，可能是那个环境已经让你自动紧绷起来了。你不用马上讲完整原因，先给自己一个缓冲：回去前慢一点，或者先找个安静角落待几分钟，让情绪别一下子被宿舍带走。",
        "原回复追问选项较多，改成具体缓冲建议。",
    ),
    "review_case_support_6f38efc6b5f64f65b49ac0952bf4dc46": (
        "0",
        "weak_input_ok",
        "我可能刚才没有接住你的意思。我们先不继续分析，也不逼你解释；你可以只告诉我想让我“陪着”，还是想听一个很小的建议。",
        "问号弱输入回复基本可用，改得更短更自然。",
    ),
    "review_case_support_729d4c22b83b4b1582b14e7f20162bdd": (
        "1",
        "pushy,too_long",
        "怕考试挂科又压力很大，真的会让人一直绷着。我们先不想整门课怎么解决，只把目标放小一点：今天先挑一科，做一个 20 分钟能完成的小任务，比如整理一页重点或一道错题。",
        "原回复偏长且要求用户继续回答，改成具体小步骤。",
    ),
}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def main() -> None:
    source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    target_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TARGET

    source_rows, source_fields = read_csv(source_path)
    target_rows, target_fields = read_csv(target_path)
    target_by_id = {row.get("id"): row for row in target_rows}
    fieldnames = list(dict.fromkeys([*target_fields, *source_fields]))

    merged_rows: list[dict[str, str]] = []
    manual_filled = 0
    for source_row in source_rows:
        row = dict(source_row)
        existing = target_by_id.get(source_row.get("id"))
        if existing:
            row.update(existing)
        rewrite = MANUAL_REWRITES.get(source_row.get("id", ""))
        if rewrite:
            mark_bad, tags, chosen, note = rewrite
            row["mark_bad"] = mark_bad
            row["problem_tags"] = tags
            row["chosen"] = chosen
            row["review_note"] = note
            manual_filled += 1
        merged_rows.append(row)

    with target_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    filled = sum(bool((row.get("chosen") or "").strip()) for row in merged_rows)
    dpo_possible = sum(
        bool((row.get("chosen") or "").strip()) and bool((row.get("assistant_reply") or "").strip())
        for row in merged_rows
    )
    print(
        {
            "rows": len(merged_rows),
            "filled_chosen": filled,
            "dpo_possible": dpo_possible,
            "manual_filled": manual_filled,
            "output": str(target_path),
        }
    )


if __name__ == "__main__":
    main()
