"""
Day15 最小 Harness：批量评测工单结构化分析 prompt。

运行命令：
D:\\Pythoncode\\.venv\\Scripts\\python.exe evals\\run_work_order_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.common.exceptions import ModelCallException
from day04_app.database import SessionLocal
from day04_app.services.eval_result_service import save_eval_report
from day04_app.services.chat_service import (
    WORK_ORDER_ANALYSIS_PROMPT_NAME,
    WORK_ORDER_ANALYSIS_PROMPT_VERSION,
    analyze_work_order_structured,
)


DATASET_PATH = Path(__file__).parent / "datasets" / "work_order_analysis_v1.jsonl"
REPORT_DIR = Path(__file__).parent / "reports"
CHINA_TZ = timezone(timedelta(hours=8))

# 这三个值先用常量表示；后续升级成 prompt_version 表时，会从数据库读取。
PROMPT_NAME = WORK_ORDER_ANALYSIS_PROMPT_NAME
PROMPT_VERSION = WORK_ORDER_ANALYSIS_PROMPT_VERSION
DATASET_VERSION = "work_order_analysis_v1"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(json.loads(line))
    return samples


def is_match(actual: dict[str, Any], expected: dict[str, Any], field_name: str) -> bool:
    return actual.get(field_name) == expected.get(field_name)


def run_eval() -> dict[str, Any]:
    run_id = f"wo_eval_{datetime.now(CHINA_TZ).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    samples = load_dataset(DATASET_PATH)
    rows: list[dict[str, Any]] = []

    for sample in samples:
        start_time = time.perf_counter()
        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "schema_valid": False,
            "category_match": False,
            "risk_level_match": False,
            "human_review_match": False,
            "total_tokens": 0,
            "cost_ms": 0,
            "error_type": None,
            "error_message": None,
        }
        try:
            result = analyze_work_order_structured(sample["input"])
            cost_ms = round((time.perf_counter() - start_time) * 1000)
            actual = result.analysis.model_dump()
            expected = sample["expected"]

            # Pydantic 能返回 result，说明 JSON 格式和硬性字段规则已经通过。
            row["schema_valid"] = True
            row["category_match"] = is_match(actual, expected, "category")
            row["risk_level_match"] = is_match(actual, expected, "risk_level")
            row["human_review_match"] = is_match(actual, expected, "need_human_review")
            row["total_tokens"] = result.total_tokens
            row["cost_ms"] = cost_ms
            row["actual"] = actual
            row["expected"] = expected
        except ModelCallException as exc:
            row["cost_ms"] = round((time.perf_counter() - start_time) * 1000)
            row["error_type"] = exc.error_type
            row["error_message"] = exc.message
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    sample_count = len(rows)
    schema_valid_count = sum(1 for row in rows if row["schema_valid"])
    matched_rows = [row for row in rows if row["schema_valid"]]

    metrics = {
        "sample_count": sample_count,
        "schema_valid_rate": round(schema_valid_count / sample_count, 4) if sample_count else 0,
        "category_accuracy": accuracy(matched_rows, "category_match"),
        "risk_level_accuracy": accuracy(matched_rows, "risk_level_match"),
        "human_review_accuracy": accuracy(matched_rows, "human_review_match"),
        "avg_total_tokens": round(mean([row["total_tokens"] for row in rows]), 2) if rows else 0,
        "avg_cost_ms": round(mean([row["cost_ms"] for row in rows]), 2) if rows else 0,
        "failed_sample_ids": [row["sample_id"] for row in rows if not row["schema_valid"]],
    }
    report = {
        "run_id": run_id,
        "run_at": datetime.now(CHINA_TZ).isoformat(),
        "prompt_name": PROMPT_NAME,
        "prompt_version": PROMPT_VERSION,
        "dataset_version": DATASET_VERSION,
        "dataset_path": str(DATASET_PATH),
        # metrics 是给管理端列表页看的汇总指标，rows 是进入详情页后看的逐条样本结果。
        "metrics": metrics,
        "rows": rows,
    }
    return report


def accuracy(rows: list[dict[str, Any]], field_name: str) -> float:
    if not rows:
        return 0
    return round(sum(1 for row in rows if row[field_name]) / len(rows), 4)


def main() -> None:
    report = run_eval()
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
