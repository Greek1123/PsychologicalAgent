from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from campus_support_agent.ms_swift_style_builder import build_ms_swift_style_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ms-swift-ready datasets from the style-first pack.")
    parser.add_argument(
        "--style-pack",
        default=str(ROOT / "data" / "training" / "style_first_pack"),
        help="Style-first pack directory.",
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "data" / "training" / "ms_swift"),
        help="Output directory for ms-swift datasets.",
    )
    args = parser.parse_args()

    manifest = build_ms_swift_style_datasets(args.style_pack, args.outdir)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
