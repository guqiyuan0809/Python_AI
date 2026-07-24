"""
评测结果服务。

类似 Java 项目中的 EvalResultService，负责把 harness 的运行结果保存到数据库。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from day04_app.models import AiEvalCaseResult, AiEvalRun


def save_eval_report(db: Session, report: dict[str, Any]) -> AiEvalRun:
    """保存一次评测汇总和每条样本明细。"""
    metrics = report["metrics"]
    metrics_with_context = {
        **metrics,
        "prompt_id": report.get("prompt_id"),
        "dataset_source": report.get("dataset_source"),
        "prompt_snapshot": report.get("prompt_snapshot"),
    }
    eval_run = AiEvalRun(
        run_id=report["run_id"],
        prompt_name=report["prompt_name"],
        prompt_version=report["prompt_version"],
        dataset_version=report["dataset_version"],
        sample_count=metrics["sample_count"],
        schema_valid_rate=metrics["schema_valid_rate"],
        category_accuracy=metrics["category_accuracy"],
        risk_level_accuracy=metrics["risk_level_accuracy"],
        human_review_accuracy=metrics["human_review_accuracy"],
        avg_total_tokens=metrics["avg_total_tokens"],
        avg_cost_ms=metrics["avg_cost_ms"],
        # 汇总指标整体保存一份 JSON，后续新增指标时可做到向后兼容。
        metrics_json=json.dumps(metrics_with_context, ensure_ascii=False),
    )
    db.add(eval_run)

    for row in report["rows"]:
        case_result = AiEvalCaseResult(
            run_id=report["run_id"],
            sample_id=row["sample_id"],
            # MySQL tinyint/bool 在不同驱动中表现不完全一致；这里用 0/1 存储布尔命中结果。
            schema_valid=1 if row["schema_valid"] else 0,
            category_match=1 if row["category_match"] else 0,
            risk_level_match=1 if row["risk_level_match"] else 0,
            human_review_match=1 if row["human_review_match"] else 0,
            total_tokens=row["total_tokens"],
            cost_ms=row["cost_ms"],
            error_type=row["error_type"],
            error_message=row["error_message"],
            expected_json=json.dumps(row.get("expected"), ensure_ascii=False),
            actual_json=json.dumps(row.get("actual"), ensure_ascii=False),
            # 明细整行也保存一份，方便以后增加字段时不用立刻改表。
            row_json=json.dumps(row, ensure_ascii=False),
        )
        db.add(case_result)

    db.commit()
    db.refresh(eval_run)
    return eval_run
