"""Agent Loop Harness 结果持久化服务。"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from day04_app.models import AiAgentEvalCaseResult, AiAgentEvalRun


def save_agent_eval_report(db: Session, report: dict[str, Any]) -> AiAgentEvalRun:
    """一次事务保存 Agent 汇总报告和每条样本断言结果。"""
    metrics = report["metrics"]
    metrics_with_context = {
        **metrics,
        "run_at": report["run_at"],
        "agent_snapshot": report["agent_snapshot"],
    }
    eval_run = AiAgentEvalRun(
        run_id=report["run_id"],
        agent_name=report["agent_name"],
        agent_version=report["agent_version"],
        dataset_version=report["dataset_version"],
        agent_snapshot_hash=report["agent_snapshot_hash"],
        sample_count=metrics["sample_count"],
        status_match_rate=metrics["status_match_rate"],
        step_sequence_match_rate=metrics["step_sequence_match_rate"],
        tool_call_accuracy=metrics["tool_call_accuracy"],
        observation_status_accuracy=metrics["observation_status_accuracy"],
        safety_case_pass_rate=metrics["safety_case_pass_rate"],
        full_pass_rate=metrics["full_pass_rate"],
        avg_step_count=metrics["avg_step_count"],
        avg_total_tokens=metrics["avg_total_tokens"],
        avg_cost_ms=metrics["avg_cost_ms"],
        metrics_json=json.dumps(metrics_with_context, ensure_ascii=False),
    )
    db.add(eval_run)
    for row in report["rows"]:
        db.add(
            AiAgentEvalCaseResult(
                run_id=report["run_id"],
                sample_id=row["sample_id"],
                sample_type=row["sample_type"],
                status_match=1 if row["status_match"] else 0,
                step_sequence_match=1 if row["step_sequence_match"] else 0,
                tool_call_match=1 if row["tool_call_match"] else 0,
                observation_status_match=1 if row["observation_status_match"] else 0,
                answer_match=1 if row["answer_match"] else 0,
                case_pass=1 if row["case_pass"] else 0,
                actual_step_count=row["actual_step_count"],
                total_tokens=row["total_tokens"],
                cost_ms=row["cost_ms"],
                error_type=row["error_type"],
                error_message=row["error_message"],
                expected_json=json.dumps(row["expected"], ensure_ascii=False),
                actual_json=json.dumps(row["actual"], ensure_ascii=False),
                row_json=json.dumps(row, ensure_ascii=False),
            )
        )
    db.commit()
    db.refresh(eval_run)
    return eval_run
