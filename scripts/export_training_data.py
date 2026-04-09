from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# 这个脚本只是一个轻量入口，方便直接从项目根目录导出训练数据。
from campus_support_agent.training_export import main


if __name__ == "__main__":
    main()
