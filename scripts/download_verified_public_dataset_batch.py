"""Download a verified, non-duplicate public dataset batch.

The goal of this script is not to collect every dataset on the internet. It
keeps a small, auditable list of real sources that map to the project training
manual, skips datasets that are already present locally, and records the reason
for every skipped restricted/empty source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "public_training_datasets"
DEFAULT_BATCH_DIR = DATASET_ROOT / "batches" / "2026-05-10_verified_batch_08"


@dataclass(frozen=True)
class DatasetSource:
    source_id: str
    display_name: str
    platform: str
    repo_id: str
    url: str
    destination_name: str
    manual_categories: tuple[str, ...]
    recommended_use: str
    license_note: str
    size_note: str
    default_download: bool
    caution: str = ""


@dataclass
class DownloadResult:
    source_id: str
    display_name: str
    status: str
    destination: str | None
    message: str


DOWNLOADABLE_SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        source_id="li2017dailydialog_daily_dialog",
        display_name="DailyDialog",
        platform="huggingface",
        repo_id="li2017dailydialog/daily_dialog",
        url="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        destination_name="DailyDialog",
        manual_categories=("多轮自然聊天", "日常闲聊"),
        recommended_use=(
            "英文日常多轮对话，可用于对话结构、话题承接和情绪标签辅助；不直接替代中文心理语料。"
        ),
        license_note="Non-commercial/research terms; cite DailyDialog paper.",
        size_note="约 13K 日常对话。",
        default_download=True,
    ),
    DatasetSource(
        source_id="thu_coai_kdconv",
        display_name="KdConv",
        platform="huggingface",
        repo_id="thu-coai/kdconv",
        url="https://huggingface.co/datasets/thu-coai/kdconv",
        destination_name="KdConv",
        manual_categories=("多轮自然聊天", "日常闲聊"),
        recommended_use=(
            "中文多领域知识驱动多轮对话，适合补上下文承接、长轮次话题转换和自然追问。"
        ),
        license_note="Dataset card cites ACL 2020 KdConv; follow upstream terms and citation.",
        size_note="约 4,500 个对话、86K utterances，HF 显示约 47.3 MB。",
        default_download=True,
    ),
    DatasetSource(
        source_id="xywang1_NaturalConv",
        display_name="NaturalConv",
        platform="huggingface",
        repo_id="xywang1/NaturalConv",
        url="https://huggingface.co/datasets/xywang1/NaturalConv",
        destination_name="NaturalConv",
        manual_categories=("多轮自然聊天", "日常闲聊"),
        recommended_use=(
            "补强中文自然多轮聊天、话题承接和上下文连贯性；不直接当心理咨询语料。"
        ),
        license_note="Tencent AI Lab NaturalConv terms; research/non-commercial usage.",
        size_note="约 19,919 个多轮对话，约 400K utterances，HF 显示约 52.6 MB。",
        default_download=True,
    ),
    DatasetSource(
        source_id="to_be_annomi_motivational_interviewing",
        display_name="AnnoMI Motivational Interviewing Therapy Conversations",
        platform="huggingface",
        repo_id="to-be/annomi-motivational-interviewing-therapy-conversations",
        url="https://huggingface.co/datasets/to-be/annomi-motivational-interviewing-therapy-conversations",
        destination_name="AnnoMI_Motivational_Interviewing",
        manual_categories=("校园心理支持对话", "支持方案数据", "坏回复对比数据"),
        recommended_use=(
            "真实治疗/动机式访谈转写，适合学习倾听、反映、开放式提问和高低质量回复对比。"
        ),
        license_note="OpenRAIL; original AnnoMI paper/citation required.",
        size_note="133 个长对话，HF 文件约 475 KB。",
        default_download=True,
    ),
    DatasetSource(
        source_id="Algorithmic_Human_Development_Group_Multilingual_Therapy_Dialogues",
        display_name="Multilingual Therapy Dialogues",
        platform="huggingface",
        repo_id="Algorithmic-Human-Development-Group/Multilingual-Therapy-Dialogues",
        url="https://huggingface.co/datasets/Algorithmic-Human-Development-Group/Multilingual-Therapy-Dialogues",
        destination_name="Multilingual_Therapy_Dialogues",
        manual_categories=("校园心理支持对话", "支持方案数据"),
        recommended_use=(
            "多语言治疗/支持对话，用于学习咨询式对话结构；需筛出适合中文项目的字段。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="HF therapy filter显示约 7.18K rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="th_nuernberg_OnCoCoV1",
        display_name="OnCoCoV1",
        platform="huggingface",
        repo_id="th-nuernberg/OnCoCoV1",
        url="https://huggingface.co/datasets/th-nuernberg/OnCoCoV1",
        destination_name="OnCoCoV1",
        manual_categories=("支持方案数据", "坏回复对比数据"),
        recommended_use=(
            "英文咨询/辅导类对话数据，可辅助抽取开放式提问、反映和建议结构。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="HF counseling filter显示约 5.56K rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="jkhedri_psychology_dataset",
        display_name="Psychology Preference Dataset",
        platform="huggingface",
        repo_id="jkhedri/psychology-dataset",
        url="https://huggingface.co/datasets/jkhedri/psychology-dataset",
        destination_name="Psychology_Preference_Dataset",
        manual_categories=("坏回复对比数据", "拒绝诊断", "支持方案数据"),
        recommended_use=(
            "含 question、response_j、response_k，可用于构造偏好数据，学习好回复和坏回复差异。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="约 9.85K rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="Ihssane123_Mental_Health_Dataset",
        display_name="Mental Health QA Dataset",
        platform="huggingface",
        repo_id="Ihssane123/Mental_Health_Dataset",
        url="https://huggingface.co/datasets/Ihssane123/Mental_Health_Dataset",
        destination_name="Ihssane_Mental_Health_Dataset",
        manual_categories=("支持方案数据", "拒绝诊断"),
        recommended_use=(
            "英文心理健康 FAQ/QA，可用于知识性回复参考；需去重，避免和已有 FAQ 类数据重复。"
        ),
        license_note="MIT.",
        size_note="约 3.68K rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="rjac_DepressionDetection",
        display_name="DepressionDetection",
        platform="huggingface",
        repo_id="rjac/DepressionDetection",
        url="https://huggingface.co/datasets/rjac/DepressionDetection",
        destination_name="DepressionDetection",
        manual_categories=("风险等级标注数据", "场景化压力数据"),
        recommended_use=(
            "抑郁相关文本分类辅助数据，只用于风险/状态识别层，不用于生成回复。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="约 5.41K train rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="Johnson8187_Chinese_Multi_Emotion_Dialogue_Dataset",
        display_name="Chinese Multi-Emotion Dialogue Dataset",
        platform="huggingface",
        repo_id="Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset",
        url="https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset",
        destination_name="Chinese_Multi-Emotion_Dialogue_Dataset",
        manual_categories=("场景化压力数据", "风险等级标注数据"),
        recommended_use=(
            "辅助训练/评估情绪识别和情绪标签，不建议单独用于生成心理支持回复。"
        ),
        license_note="MIT.",
        size_note="CSV 文件约 320 KB。",
        default_download=True,
    ),
    DatasetSource(
        source_id="BEncoderRT_User_Intent_Risk_Triage",
        display_name="Chinese Mental Health Risk Assessment Dataset",
        platform="huggingface",
        repo_id="BEncoderRT/User_Intent_Risk_Triage",
        url="https://huggingface.co/datasets/BEncoderRT/User_Intent_Risk_Triage",
        destination_name="User_Intent_Risk_Triage",
        manual_categories=("风险等级标注数据", "转介判断数据", "高风险危机场景"),
        recommended_use=(
            "辅助训练本地风险评估层输出 intent/risk/strategy JSON；不用于直接生成聊天回复。"
        ),
        license_note="CC-BY-4.0 / Apache-2.0 card metadata; use with attribution.",
        size_note="HF 数据页显示约 2,000 行，下载文件约 8.2 MB。",
        default_download=True,
        caution="需人工复核风险标签质量，不能替代真实危机评估规范。",
    ),
    DatasetSource(
        source_id="RAKS19_mental_health_dataset_llama",
        display_name="Mental Health Dataset LLaMA Format",
        platform="huggingface",
        repo_id="RAKS19/mental-health-dataset-llama",
        url="https://huggingface.co/datasets/RAKS19/mental-health-dataset-llama",
        destination_name="RAKS19_Mental_Health_Dataset_LLaMA",
        manual_categories=("支持方案数据", "校园心理支持对话"),
        recommended_use=(
            "英文心理咨询指令格式数据，适合做候选池；需和已有 CounselChat 类数据去重。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="约 7.5K rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="Phora68_dr_sage_dataset",
        display_name="Dr Sage Dataset",
        platform="huggingface",
        repo_id="Phora68/dr-sage-dataset",
        url="https://huggingface.co/datasets/Phora68/dr-sage-dataset",
        destination_name="Dr_Sage_Dataset",
        manual_categories=("支持方案数据", "拒绝诊断"),
        recommended_use=(
            "英文心理/健康咨询类对话候选池；需要抽样检查是否存在诊断化或过度承诺。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="HF therapy filter显示约 5.29K rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="SerizawaJ_Calira",
        display_name="Calira",
        platform="huggingface",
        repo_id="SerizawaJ/Calira",
        url="https://huggingface.co/datasets/SerizawaJ/Calira",
        destination_name="Calira",
        manual_categories=("支持方案数据", "坏回复对比数据"),
        recommended_use=(
            "小型咨询类数据，适合抽样看风格，不建议提高权重训练。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="HF therapy filter显示约 254 rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="nowsika_NOBLE_Counseling_Navigator",
        display_name="NOBLE Counseling Navigator",
        platform="huggingface",
        repo_id="nowsika/NOBLE_Counseling_Navigator_EN-v3.2.1",
        url="https://huggingface.co/datasets/nowsika/NOBLE_Counseling_Navigator_EN-v3.2.1",
        destination_name="NOBLE_Counseling_Navigator_EN",
        manual_categories=("支持方案数据", "转介判断数据"),
        recommended_use=(
            "小型咨询导航数据，用于支持方案流程参考，不适合单独训练。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="HF therapy filter显示约 100 rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="nowsika_NOBLE_Counseling_Gardener",
        display_name="NOBLE Counseling Gardener",
        platform="huggingface",
        repo_id="nowsika/NOBLE_Counseling_Gardener_EN-v3.2.1",
        url="https://huggingface.co/datasets/nowsika/NOBLE_Counseling_Gardener_EN-v3.2.1",
        destination_name="NOBLE_Counseling_Gardener_EN",
        manual_categories=("支持方案数据", "转介判断数据"),
        recommended_use=(
            "小型咨询策略数据，用于支持方案结构参考，不适合单独训练。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="HF therapy filter显示约 100 rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="moujar_MentalHealth_Darija",
        display_name="MentalHealth Darija",
        platform="huggingface",
        repo_id="moujar/MentalHealth-Darija",
        url="https://huggingface.co/datasets/moujar/MentalHealth-Darija",
        destination_name="MentalHealth_Darija",
        manual_categories=("风险等级标注数据", "场景化压力数据"),
        recommended_use=(
            "多类别心理健康风险文本分类辅助数据，含英文翻译字段；只用于评估/分类层。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="约 51,093 examples。",
        default_download=True,
    ),
    DatasetSource(
        source_id="yibba_moroccan_darija_therapy_conversations",
        display_name="Moroccan Darija Therapy Conversations",
        platform="huggingface",
        repo_id="yibba/moroccan-darija-therapy-conversations",
        url="https://huggingface.co/datasets/yibba/moroccan-darija-therapy-conversations",
        destination_name="Moroccan_Darija_Therapy_Conversations",
        manual_categories=("校园心理支持对话", "支持方案数据"),
        recommended_use=(
            "治疗/支持对话结构参考；语言不匹配，默认只做结构和安全策略参考。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="HF therapy filter显示约 16.4K rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="yibba_chat_darija_therapy",
        display_name="Chat Darija Therapy",
        platform="huggingface",
        repo_id="yibba/chat-darija-therapy",
        url="https://huggingface.co/datasets/yibba/chat-darija-therapy",
        destination_name="Chat_Darija_Therapy",
        manual_categories=("校园心理支持对话", "支持方案数据"),
        recommended_use=(
            "治疗/支持对话结构参考；语言不匹配，默认只做结构和安全策略参考。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="HF therapy filter显示约 22.6K rows。",
        default_download=True,
    ),
    DatasetSource(
        source_id="ironDong_Children_Counsel",
        display_name="Children Counsel",
        platform="huggingface",
        repo_id="ironDong/Children_Counsel",
        url="https://huggingface.co/datasets/ironDong/Children_Counsel",
        destination_name="Children_Counsel",
        manual_categories=("校园心理支持对话", "隐私与信任数据"),
        recommended_use=(
            "儿童/青少年咨询相关数据，需严格抽样审核和伦理风险检查，不能直接混入训练。"
        ),
        license_note="Check dataset card before publication or redistribution.",
        size_note="HF counseling filter显示约 1.1K rows。",
        default_download=False,
        caution="涉及未成年人咨询场景，默认不下载；需要 --include-large 后再人工复核。",
    ),
    DatasetSource(
        source_id="silver_lccc",
        display_name="LCCC",
        platform="huggingface",
        repo_id="silver/lccc",
        url="https://huggingface.co/datasets/silver/lccc",
        destination_name="LCCC",
        manual_categories=("多轮自然聊天", "日常闲聊"),
        recommended_use=(
            "大规模中文社交媒体对话，只建议抽样用于自然聊天预热；不适合直接做心理支持风格。"
        ),
        license_note="MIT on HF dataset card; cite LCCC paper.",
        size_note="HF 显示约 18.8M rows，总文件约 979 MB。",
        default_download=False,
        caution="大文件且社交语料噪声高；需要 --include-large，下载后必须采样和清洗。",
    ),
    DatasetSource(
        source_id="allenai_prosocial_dialog",
        display_name="ProsocialDialog",
        platform="huggingface",
        repo_id="allenai/prosocial-dialog",
        url="https://huggingface.co/datasets/allenai/prosocial-dialog",
        destination_name="ProsocialDialog",
        manual_categories=("拒绝诊断", "高风险危机场景", "坏回复对比数据"),
        recommended_use=(
            "英文社会安全/有害行为对话数据，适合训练或评估安全策略层，不适合直接混入中文聊天 SFT。"
        ),
        license_note="CC-BY-4.0.",
        size_note="约 166K rows。",
        default_download=False,
        caution="包含攻击性/有害内容，只能用于安全分类和拒绝策略；需要 --include-large。",
    ),
    DatasetSource(
        source_id="lmsys_toxic_chat",
        display_name="ToxicChat",
        platform="huggingface",
        repo_id="lmsys/toxic-chat",
        url="https://huggingface.co/datasets/lmsys/toxic-chat",
        destination_name="ToxicChat",
        manual_categories=("高风险危机场景", "拒绝诊断", "坏回复对比数据"),
        recommended_use=(
            "安全/有害内容识别辅助数据，只用于 guardrail 和风险层评估，不用于普通聊天生成。"
        ),
        license_note="CC-BY-4.0.",
        size_note="约 10K user prompts。",
        default_download=False,
        caution="含有害内容，需要 --include-large，并且只能进入安全评估池。",
    ),
    DatasetSource(
        source_id="Anthropic_hh_rlhf",
        display_name="HH-RLHF",
        platform="huggingface",
        repo_id="Anthropic/hh-rlhf",
        url="https://huggingface.co/datasets/Anthropic/hh-rlhf",
        destination_name="HH_RLHF",
        manual_categories=("坏回复对比数据", "拒绝诊断"),
        recommended_use=(
            "帮助/无害偏好数据，用于偏好训练参考；英文且非心理专用，不能直接替代项目反馈数据。"
        ),
        license_note="MIT.",
        size_note="偏好数据，体积较大。",
        default_download=False,
        caution="大文件且英文泛安全偏好，需要 --include-large。",
    ),
    DatasetSource(
        source_id="LooksJuicy_Chinese_Emotional_Intelligence",
        display_name="Chinese Emotional Intelligence",
        platform="huggingface",
        repo_id="LooksJuicy/Chinese-Emotional-Intelligence",
        url="https://huggingface.co/datasets/LooksJuicy/Chinese-Emotional-Intelligence",
        destination_name="Chinese_Emotional_Intelligence",
        manual_categories=("日常闲聊", "支持方案数据", "坏回复对比数据"),
        recommended_use=(
            "中文高情商问答风格参考；样本包含玩笑、粗口和不适合心理支持的表达，必须过滤。"
        ),
        license_note="Apache-2.0.",
        size_note="约 40.3K rows。",
        default_download=False,
        caution="风格不稳定，默认不下载；如果下载，只能进候选池，不能直接训练。",
    ),
    DatasetSource(
        source_id="YIRONGCHEN_SoulChatCorpus",
        display_name="SoulChatCorpus",
        platform="modelscope",
        repo_id="YIRONGCHEN/SoulChatCorpus",
        url="https://modelscope.cn/datasets/YIRONGCHEN/SoulChatCorpus",
        destination_name="SoulChatCorpus",
        manual_categories=("校园心理支持对话", "弱输入承接", "隐私与信任数据"),
        recommended_use=(
            "大规模中文多轮共情对话，可作为心理支持风格参考；需要抽样清洗，防止模板化。"
        ),
        license_note="ModelScope dataset page; use according to source terms.",
        size_note="公开页说明约 258,354 个多轮对话、1,517,344 轮；体积较大。",
        default_download=False,
        caution="大文件，且可能含模型生成/改写成分；默认不下载，需加 --include-large。",
    ),
)


RESTRICTED_OR_SKIPPED_SOURCES = (
    {
        "source_id": "Mxode_AuraDial",
        "display_name": "AuraDial",
        "url": "https://huggingface.co/datasets/Mxode/AuraDial",
        "reason": "论文存在且指向 HF，但当前 HF 仓库显示数据文件为空，暂不下载。",
        "recommended_action": "后续定期复查；数据真正开放后再纳入。",
    },
    {
        "source_id": "BAAI_EmotionTalk",
        "display_name": "EmotionTalk",
        "url": "https://huggingface.co/datasets/BAAI/Emotiontalk",
        "reason": "需要登录同意共享联系信息，且总文件约 36.1GB；当前阶段不适合直接下载。",
        "recommended_action": "后续做语音/多模态阶段时再申请访问，只抽取文本标注部分。",
    },
    {
        "source_id": "EmoCareAI_Psych8k",
        "display_name": "Psych8k",
        "url": "https://huggingface.co/datasets/EmoCareAI/Psych8k",
        "reason": "真实咨询录音转写来源，但 HF 为 gated dataset，需要登录同意条件，不能绕过授权。",
        "recommended_action": "如项目需要英文治疗真实转写，可用你的 HF 账号申请后再导入。",
    },
    {
        "source_id": "qiuhuachuan_PsyDial",
        "display_name": "PsyDial-D0_m/D1-D4",
        "url": "https://huggingface.co/datasets/qiuhuachuan/PsyDial-D0_m",
        "reason": "Hugging Face gated dataset，需要登录并接受条件，不能绕过授权下载。",
        "recommended_action": "如项目确需使用，先用你的 HF 账号申请访问。",
    },
    {
        "source_id": "chatopera_efaqa_corpus_zh",
        "display_name": "EFAQA / Emotional First Aid Dataset",
        "url": "https://github.com/chatopera/efaqa-corpus-zh",
        "reason": "代码开源，但 README 说明语料文件需要证书标识才能下载和使用。",
        "recommended_action": "如需使用，购买/申请证书后再按官方方式导入。",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional source_id list. Example: --only xywang1_NaturalConv",
    )
    return parser.parse_args()


def existing_dataset_names() -> set[str]:
    names: set[str] = set()
    if DATASET_ROOT.exists():
        names.update(
            path.name.lower()
            for path in DATASET_ROOT.iterdir()
            if path.is_dir() and has_dataset_payload(path)
        )
    batches = DATASET_ROOT / "batches"
    if batches.exists():
        for batch in batches.iterdir():
            if batch.is_dir():
                names.update(
                    path.name.lower()
                    for path in batch.iterdir()
                    if path.is_dir() and has_dataset_payload(path)
                )
    return names


def has_dataset_payload(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    ignored = {"README.md", "manifest.json", "download_log.json"}
    for child in path.rglob("*"):
        if child.is_file() and child.name not in ignored and child.stat().st_size > 0:
            return True
    return False


def should_consider(source: DatasetSource, args: argparse.Namespace) -> bool:
    if args.only is not None and source.source_id not in set(args.only):
        return False
    if not source.default_download and not args.include_large:
        return False
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_manifest(root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def download_huggingface(source: DatasetSource, destination: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Missing dependency: huggingface_hub") from exc

    snapshot_download(
        repo_id=source.repo_id,
        repo_type="dataset",
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )


def download_modelscope(source: DatasetSource, destination: Path) -> None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Missing dependency: modelscope") from exc

    snapshot_download(source.repo_id, local_dir=str(destination))


def download_source(source: DatasetSource, out_dir: Path) -> DownloadResult:
    destination = out_dir / source.destination_name
    if has_dataset_payload(destination):
        return DownloadResult(
            source_id=source.source_id,
            display_name=source.display_name,
            status="skipped_existing_destination",
            destination=str(destination),
            message="Destination already exists and is not empty.",
        )

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if source.platform == "huggingface":
        download_huggingface(source, destination)
    elif source.platform == "modelscope":
        download_modelscope(source, destination)
    else:  # pragma: no cover
        raise RuntimeError(f"Unsupported platform: {source.platform}")

    return DownloadResult(
        source_id=source.source_id,
        display_name=source.display_name,
        status="downloaded",
        destination=str(destination),
        message="Downloaded successfully.",
    )


def make_readme(
    out_dir: Path,
    considered: Iterable[DatasetSource],
    results: Iterable[DownloadResult],
) -> str:
    lines = [
        "# Verified Public Dataset Batch 08",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This batch follows `docs/心理支持Agent训练集收集手册.docx` and only includes real, traceable public sources that were not already present in the local dataset archive.",
        "",
        "## Downloaded / Attempted Sources",
        "",
        "| Source | Platform | Categories | Recommended Use | License / Terms | Status |",
        "|---|---|---|---|---|---|",
    ]
    result_by_id = {item.source_id: item for item in results}
    for source in considered:
        result = result_by_id.get(source.source_id)
        status = result.status if result else "planned"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"[{source.display_name}]({source.url})",
                    source.platform,
                    "、".join(source.manual_categories),
                    source.recommended_use,
                    source.license_note,
                    status,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Restricted / Skipped Sources",
            "",
            "| Source | Reason | Recommended Action |",
            "|---|---|---|",
        ]
    )
    for source in RESTRICTED_OR_SKIPPED_SOURCES:
        lines.append(
            f"| [{source['display_name']}]({source['url']}) | {source['reason']} | {source['recommended_action']} |"
        )
    lines.extend(
        [
            "",
            "## Usage Notes",
            "",
            "- Do not merge these datasets directly into SFT without cleaning and sampling.",
            "- Crisis/risk datasets should train or evaluate the safety/risk layer, not normal chat generation.",
            "- Large empathetic corpora must be deduplicated and sampled, otherwise the model may become template-like again.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_names = existing_dataset_names()
    considered: list[DatasetSource] = []
    results: list[DownloadResult] = []

    for source in DOWNLOADABLE_SOURCES:
        if not should_consider(source, args):
            results.append(
                DownloadResult(
                    source_id=source.source_id,
                    display_name=source.display_name,
                    status="skipped_not_selected",
                    destination=None,
                    message=source.caution or "Not selected by default.",
                )
            )
            continue

        considered.append(source)
        if source.destination_name.lower() in existing_names:
            results.append(
                DownloadResult(
                    source_id=source.source_id,
                    display_name=source.display_name,
                    status="skipped_duplicate",
                    destination=None,
                    message="A dataset directory with the same name already exists locally.",
                )
            )
            continue

        if args.dry_run:
            results.append(
                DownloadResult(
                    source_id=source.source_id,
                    display_name=source.display_name,
                    status="dry_run_planned",
                    destination=str(out_dir / source.destination_name),
                    message="Would download.",
                )
            )
            continue

        try:
            results.append(download_source(source, out_dir))
        except Exception as exc:  # pragma: no cover - network/local env dependent
            failed_destination = out_dir / source.destination_name
            if failed_destination.exists() and not has_dataset_payload(failed_destination):
                shutil.rmtree(failed_destination)
            results.append(
                DownloadResult(
                    source_id=source.source_id,
                    display_name=source.display_name,
                    status="failed",
                    destination=str(out_dir / source.destination_name),
                    message=str(exc),
                )
            )

    manifest = {
        "batch_dir": str(out_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": [asdict(source) for source in DOWNLOADABLE_SOURCES],
        "restricted_or_skipped_sources": RESTRICTED_OR_SKIPPED_SOURCES,
        "results": [asdict(result) for result in results],
        "files": collect_file_manifest(out_dir) if not args.dry_run else [],
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "download_log.json", [asdict(result) for result in results])
    (out_dir / "README.md").write_text(
        make_readme(out_dir, considered, results),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "downloaded": sum(1 for item in results if item.status == "downloaded"),
                "skipped": sum(1 for item in results if item.status.startswith("skipped")),
                "failed": sum(1 for item in results if item.status == "failed"),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
