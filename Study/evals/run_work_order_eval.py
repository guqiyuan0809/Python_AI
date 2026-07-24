"""
Day15 最小 Harness：批量评测工单结构化分析 prompt。

运行命令：
D:\\Pythoncode\\.venv\\Scripts\\python.exe evals\\run_work_order_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.database import SessionLocal
from day04_app.services.chat_service import (
    WORK_ORDER_ANALYSIS_PROMPT_NAME,
    WORK_ORDER_ANALYSIS_PROMPT_VERSION,
)
from day04_app.services.eval_result_service import save_eval_report
from day04_app.services.work_order_eval_runner import run_work_order_eval


REPORT_DIR = Path(__file__).parent / "reports"

PROMPT_NAME = WORK_ORDER_ANALYSIS_PROMPT_NAME
PROMPT_VERSION = WORK_ORDER_ANALYSIS_PROMPT_VERSION
DATASET_VERSION = "work_order_analysis_v1"


def main() -> None:
    report = run_work_order_eval(
        prompt_name=PROMPT_NAME,
        prompt_version=PROMPT_VERSION,
        dataset_version=DATASET_VERSION,
    )
    report_path = try_save_report_file(report)
    save_report_to_database(report)
    print("==== EVAL REPORT ====")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    if report_path:
        print(f"报告文件已生成：{report_path}")


def save_report(report: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{report['run_id']}.json"
    # ensure_ascii=False 保留中文样本原文，方便人工复盘失败案例。
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def try_save_report_file(report: dict[str, Any]) -> Path | None:
    try:
        return save_report(report)
    except PermissionError as exc:
        # 文件报告只是评测留痕的一种形式；写文件失败时不影响后续数据库留痕。
        print(f"文件报告保存失败：{exc}")
        return None


def save_report_to_database(report: dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        # harness 属于离线评测脚本，这里显式创建 Session，类似 Java 命令行任务里手动拿 Service 保存结果。
        save_eval_report(db, report)
        print(f"数据库评测记录已保存：{report['run_id']}")
    except Exception as exc:
        db.rollback()
        print(f"数据库评测记录保存失败：{type(exc).__name__}: {exc}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
