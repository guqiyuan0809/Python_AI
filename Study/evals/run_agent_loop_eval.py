"""Day25 Agent Loop Harness 命令行入口。

示例：
D:\\Pythoncode\\.venv\\Scripts\\python.exe -B evals\\run_agent_loop_eval.py --agent-version v1 --sample-limit 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.database import SessionLocal
from day04_app.services.agent_loop_eval_result_service import save_agent_eval_report
from day04_app.services.agent_loop_eval_runner import run_agent_loop_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Day25 Agent Loop Harness")
    parser.add_argument("--agent-version", required=True, help="本次被评测 Agent 的版本标签，例如 v1 或 candidate-v2")
    parser.add_argument("--dataset-version", default="agent_loop_v1", help="评测数据集版本")
    parser.add_argument("--sample-limit", type=int, default=None, help="最多执行多少条样本，用于控制模型调用成本")
    args = parser.parse_args()

    report = run_agent_loop_eval(
        agent_version=args.agent_version,
        dataset_version=args.dataset_version,
        sample_limit=args.sample_limit,
    )
    db = SessionLocal()
    try:
        save_agent_eval_report(db, report)
    finally:
        db.close()
    print(json.dumps({"run_id": report["run_id"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
