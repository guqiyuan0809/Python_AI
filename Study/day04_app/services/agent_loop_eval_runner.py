"""Day25 Agent Loop Harness 执行器。

复用 Day16 的 ai_eval_dataset / ai_eval_sample 保存主数据，但 Agent 的断言和指标独立处理，
避免把工单分类准确率等字段误用到多步骤工具编排场景。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from day04_app.database import SessionLocal
from day04_app.models import AiEvalSample
from day04_app.schemas.chat_schema import AgentLoopResponse
from day04_app.services.agent_loop_service import (
    AGENT_DECISION_POLICY_VERSION,
    AGENT_LOOP_SYSTEM_PROMPT,
    run_agent_loop,
)
from day04_app.services.tool_calling_service import list_available_tools


CHINA_TZ = timezone(timedelta(hours=8))
AGENT_NAME = "controlled_agent_loop"


def _rate(rows: list[dict[str, Any]], field_name: str) -> float:
    if not rows:
        return 0
    return round(sum(1 for row in rows if row[field_name]) / len(rows), 4)


def _count_accuracy(rows: list[dict[str, Any]], matched_field: str, expected_field: str) -> float:
    expected_count = sum(row[expected_field] for row in rows)
    if expected_count == 0:
        return 0
    return round(sum(row[matched_field] for row in rows) / expected_count, 4)


def _build_agent_snapshot() -> tuple[dict[str, Any], str]:
    snapshot = {
        "system_prompt": AGENT_LOOP_SYSTEM_PROMPT,
        "decision_policy_version": AGENT_DECISION_POLICY_VERSION,
        "available_tools": [tool.model_dump() for tool in list_available_tools()],
    }
    snapshot_json = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    return snapshot, hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()


def _load_samples(db: Session, dataset_version: str, sample_limit: int | None) -> list[AiEvalSample]:
    statement = (
        select(AiEvalSample)
        .where(AiEvalSample.dataset_version == dataset_version)
        .where(AiEvalSample.status == "active")
        .order_by(AiEvalSample.sample_id.asc())
    )
    if sample_limit is not None:
        statement = statement.limit(sample_limit)
    samples = list(db.scalars(statement).all())
    if not samples:
        raise RuntimeError(f"未找到可用 Agent 评测样本：{dataset_version}")
    return samples


def _expected_steps(expected: dict[str, Any]) -> list[dict[str, Any]]:
    steps = expected.get("expected_steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Agent 评测样本必须提供非空 expected_steps")
    return steps


def _is_subset_match(actual: Any, expected: Any) -> bool:
    """期望参数只标注业务关键字段，实际值可包含框架补充字段。"""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _is_subset_match(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and actual == expected
    return actual == expected


def _evaluate_response(
    *,
    sample_id: str,
    sample_type: str,
    expected: dict[str, Any],
    response: AgentLoopResponse | None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """把 Agent 响应转换为可持久化的断言结果，不包含任何模型调用。"""
    expected_steps = _expected_steps(expected)
    actual = response.model_dump(mode="json") if response else None
    actual_steps = actual["steps"] if actual else []
    status_match = bool(actual and actual["status"] == expected.get("expected_status"))

    step_sequence_match = len(actual_steps) == len(expected_steps)
    tool_call_match = True
    observation_status_match = True
    expected_tool_call_count = 0
    matched_tool_call_count = 0
    expected_observation_status_count = 0
    matched_observation_status_count = 0
    for index, expected_step in enumerate(expected_steps):
        actual_step = actual_steps[index] if index < len(actual_steps) else None
        if not actual_step or actual_step.get("action") != expected_step.get("action"):
            step_sequence_match = False
            if expected_step.get("action") == "call_tool":
                expected_tool_call_count += 1
            if "observation_status" in expected_step:
                expected_observation_status_count += 1
            continue
        if expected_step.get("action") == "call_tool":
            expected_tool_call_count += 1
            is_tool_call_match = (
                actual_step.get("tool_name") == expected_step.get("tool_name")
                and _is_subset_match(
                    actual_step.get("arguments", {}), expected_step.get("arguments", {})
                )
            )
            if is_tool_call_match:
                matched_tool_call_count += 1
            else:
                step_sequence_match = False
                tool_call_match = False
        if "observation_status" in expected_step:
            expected_observation_status_count += 1
            actual_status = (actual_step.get("observation") or {}).get("status")
            if actual_status == expected_step["observation_status"]:
                matched_observation_status_count += 1
            else:
                step_sequence_match = False
                observation_status_match = False

    tool_call_match = tool_call_match and (
        matched_tool_call_count == expected_tool_call_count
    )
    observation_status_match = observation_status_match and (
        matched_observation_status_count == expected_observation_status_count
    )

    answer_keywords = expected.get("answer_contains", [])
    answer_match = bool(
        actual
        and all(keyword in actual["answer"] for keyword in answer_keywords)
    )
    case_pass = all(
        (
            status_match,
            step_sequence_match,
            tool_call_match,
            observation_status_match,
            answer_match,
            error_type is None,
        )
    )
    return {
        "sample_id": sample_id,
        "sample_type": sample_type,
        "status_match": status_match,
        "step_sequence_match": step_sequence_match,
        "tool_call_match": tool_call_match,
        "observation_status_match": observation_status_match,
        "expected_tool_call_count": expected_tool_call_count,
        "matched_tool_call_count": matched_tool_call_count,
        "expected_observation_status_count": expected_observation_status_count,
        "matched_observation_status_count": matched_observation_status_count,
        "answer_match": answer_match,
        "case_pass": case_pass,
        "actual_step_count": len(actual_steps),
        "total_tokens": actual.get("total_tokens") if actual else 0,
        "cost_ms": actual.get("cost_ms") if actual else 0,
        "error_type": error_type,
        "error_message": error_message,
        "expected": expected,
        "actual": actual,
    }


def _run_one_sample(
    sample: AiEvalSample,
    *,
    db_factory: Callable[[], Session],
    agent_runner: Callable[..., AgentLoopResponse],
) -> dict[str, Any]:
    expected = json.loads(sample.expected_json)
    db = db_factory()
    try:
        response = agent_runner(
            db,
            message=sample.input_text,
            max_steps=int(expected.get("max_steps", 3)),
            trace_id=f"agent_eval_{sample.sample_id}",
        )
        return _evaluate_response(
            sample_id=sample.sample_id,
            sample_type=sample.sample_type,
            expected=expected,
            response=response,
        )
    except Exception as exc:
        return _evaluate_response(
            sample_id=sample.sample_id,
            sample_type=sample.sample_type,
            expected=expected,
            response=None,
            error_type=type(exc).__name__,
            error_message=str(getattr(exc, "message", exc)),
        )
    finally:
        db.close()


def run_agent_loop_eval(
    *,
    agent_version: str,
    dataset_version: str,
    sample_limit: int | None = None,
    db_factory: Callable[[], Session] = SessionLocal,
    agent_runner: Callable[..., AgentLoopResponse] = run_agent_loop,
) -> dict[str, Any]:
    """执行 Agent Loop Harness；默认调用真实 Agent，调用成本由 sample_limit 控制。"""
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("sample_limit 必须大于等于 1")

    db = db_factory()
    try:
        samples = _load_samples(db, dataset_version, sample_limit)
    finally:
        db.close()

    agent_snapshot, agent_snapshot_hash = _build_agent_snapshot()
    rows = [
        _run_one_sample(sample, db_factory=db_factory, agent_runner=agent_runner)
        for sample in samples
    ]
    safety_rows = [row for row in rows if row["sample_type"] == "safety"]
    metrics = {
        "sample_count": len(rows),
        "status_match_rate": _rate(rows, "status_match"),
        "step_sequence_match_rate": _rate(rows, "step_sequence_match"),
        "tool_call_accuracy": _count_accuracy(
            rows,
            "matched_tool_call_count",
            "expected_tool_call_count",
        ),
        "observation_status_accuracy": _count_accuracy(
            rows,
            "matched_observation_status_count",
            "expected_observation_status_count",
        ),
        "safety_sample_count": len(safety_rows),
        "safety_case_pass_rate": _rate(safety_rows, "case_pass"),
        "full_pass_rate": _rate(rows, "case_pass"),
        "avg_step_count": round(mean(row["actual_step_count"] for row in rows), 2),
        "avg_total_tokens": round(mean(row["total_tokens"] or 0 for row in rows), 2),
        "avg_cost_ms": round(mean(row["cost_ms"] or 0 for row in rows), 2),
        "failed_sample_ids": [row["sample_id"] for row in rows if not row["case_pass"]],
    }
    return {
        "run_id": f"agent_eval_{datetime.now(CHINA_TZ).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
        "run_at": datetime.now(CHINA_TZ).isoformat(),
        "agent_name": AGENT_NAME,
        "agent_version": agent_version,
        "dataset_version": dataset_version,
        "agent_snapshot_hash": agent_snapshot_hash,
        "agent_snapshot": agent_snapshot,
        "metrics": metrics,
        "rows": rows,
    }
