"""Prompt Harness 评测准入门禁服务。"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import AiEvalCaseResult, AiEvalGateDecision, AiEvalRun
from day04_app.utils.snowflake_id import next_snowflake_id


# 当前是工单结构化能力的第一版门禁策略；记录到快照中，后续可升级为数据库可配置策略。
GATE_POLICY = {
    "policy_name": "work_order_analysis_release_gate",
    "policy_version": "v1",
    "min_schema_valid_rate": 0.99,
    "min_human_review_accuracy": 0.90,
    "max_risk_level_accuracy_drop": 0.02,
    "max_avg_total_tokens_increase_ratio": 0.20,
    "max_avg_cost_ms_increase_ratio": 0.20,
    "require_error_samples_all_pass": True,
}


def _get_eval_run(db: Session, run_id: str, label: str) -> AiEvalRun:
    eval_run = db.scalar(select(AiEvalRun).where(AiEvalRun.run_id == run_id))
    if eval_run is None:
        raise BusinessException(code=40021, message=f"{label}评测运行记录不存在")
    return eval_run


def _get_run_case_results(db: Session, run_id: str) -> list[AiEvalCaseResult]:
    return list(
        db.scalars(
            select(AiEvalCaseResult)
            .where(AiEvalCaseResult.run_id == run_id)
            .order_by(AiEvalCaseResult.id.asc())
        ).all()
    )


def _is_case_passed(case_result: AiEvalCaseResult) -> bool:
    return all(
        (
            case_result.schema_valid == 1,
            case_result.category_match == 1,
            case_result.risk_level_match == 1,
            case_result.human_review_match == 1,
        )
    )


def _load_sample_type(case_result: AiEvalCaseResult) -> str | None:
    try:
        row = json.loads(case_result.row_json)
    except json.JSONDecodeError:
        return None
    return row.get("sample_type")


def _metric_comparison(baseline: float | None, candidate: float | None) -> dict[str, float | None]:
    baseline_value = float(baseline or 0)
    candidate_value = float(candidate or 0)
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": round(candidate_value - baseline_value, 4),
    }


def _increase_ratio(baseline: float | None, candidate: float | None) -> float | None:
    baseline_value = float(baseline or 0)
    candidate_value = float(candidate or 0)
    if baseline_value <= 0:
        return None
    return round((candidate_value - baseline_value) / baseline_value, 4)


def _append_reason(
    reasons: list[dict[str, Any]],
    *,
    level: str,
    rule: str,
    message: str,
) -> None:
    reasons.append({"level": level, "rule": rule, "message": message})


def _build_gate_judgement(
    baseline_run: AiEvalRun,
    candidate_run: AiEvalRun,
    candidate_cases: list[AiEvalCaseResult],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """依据固定策略比较已有评测报告；这里不再次调用模型。"""
    schema = _metric_comparison(baseline_run.schema_valid_rate, candidate_run.schema_valid_rate)
    category = _metric_comparison(baseline_run.category_accuracy, candidate_run.category_accuracy)
    risk_level = _metric_comparison(baseline_run.risk_level_accuracy, candidate_run.risk_level_accuracy)
    human_review = _metric_comparison(
        baseline_run.human_review_accuracy,
        candidate_run.human_review_accuracy,
    )
    token_increase_ratio = _increase_ratio(baseline_run.avg_total_tokens, candidate_run.avg_total_tokens)
    cost_increase_ratio = _increase_ratio(baseline_run.avg_cost_ms, candidate_run.avg_cost_ms)
    failed_error_sample_ids = [
        case.sample_id
        for case in candidate_cases
        if _load_sample_type(case) == "error" and not _is_case_passed(case)
    ]

    comparison = {
        "schema_valid_rate": schema,
        "category_accuracy": category,
        "risk_level_accuracy": risk_level,
        "human_review_accuracy": human_review,
        "avg_total_tokens": {
            **_metric_comparison(baseline_run.avg_total_tokens, candidate_run.avg_total_tokens),
            "increase_ratio": token_increase_ratio,
        },
        "avg_cost_ms": {
            **_metric_comparison(baseline_run.avg_cost_ms, candidate_run.avg_cost_ms),
            "increase_ratio": cost_increase_ratio,
        },
        "error_regression": {
            "candidate_error_sample_count": sum(
                1 for case in candidate_cases if _load_sample_type(case) == "error"
            ),
            "failed_sample_ids": failed_error_sample_ids,
        },
    }
    reasons: list[dict[str, Any]] = []

    # 这些是不可被其他高分抵消的硬规则，命中任意一项就拒绝本次候选版本。
    if schema["candidate"] < GATE_POLICY["min_schema_valid_rate"]:
        _append_reason(
            reasons,
            level="reject",
            rule="min_schema_valid_rate",
            message=(
                f"候选格式通过率 {schema['candidate']:.2%} 低于最低要求 "
                f"{GATE_POLICY['min_schema_valid_rate']:.2%}"
            ),
        )
    if human_review["candidate"] < GATE_POLICY["min_human_review_accuracy"]:
        _append_reason(
            reasons,
            level="reject",
            rule="min_human_review_accuracy",
            message=(
                f"候选人工复核判断准确率 {human_review['candidate']:.2%} 低于最低要求 "
                f"{GATE_POLICY['min_human_review_accuracy']:.2%}"
            ),
        )
    if risk_level["delta"] < -GATE_POLICY["max_risk_level_accuracy_drop"]:
        _append_reason(
            reasons,
            level="reject",
            rule="max_risk_level_accuracy_drop",
            message=(
                f"候选风险等级准确率较基线下降 {abs(risk_level['delta']):.2%}，"
                f"超过允许回退 {GATE_POLICY['max_risk_level_accuracy_drop']:.2%}"
            ),
        )
    if GATE_POLICY["require_error_samples_all_pass"] and failed_error_sample_ids:
        _append_reason(
            reasons,
            level="reject",
            rule="require_error_samples_all_pass",
            message=f"{len(failed_error_sample_ids)} 条失败回归样本未通过：{', '.join(failed_error_sample_ids)}",
        )

    if any(reason["level"] == "reject" for reason in reasons):
        return "reject", comparison, reasons

    # 轻微质量回退或成本上升需要人工判断业务收益是否值得，不能直接自动发布。
    if risk_level["delta"] < 0:
        _append_reason(
            reasons,
            level="manual_review",
            rule="risk_level_accuracy_regression",
            message=f"候选风险等级准确率较基线轻微下降 {abs(risk_level['delta']):.2%}",
        )
    if category["delta"] < 0:
        _append_reason(
            reasons,
            level="manual_review",
            rule="category_accuracy_regression",
            message=f"候选分类准确率较基线下降 {abs(category['delta']):.2%}",
        )
    if token_increase_ratio is not None and token_increase_ratio > GATE_POLICY["max_avg_total_tokens_increase_ratio"]:
        _append_reason(
            reasons,
            level="manual_review",
            rule="max_avg_total_tokens_increase_ratio",
            message=(
                f"候选平均 Token 增加 {token_increase_ratio:.2%}，超过建议阈值 "
                f"{GATE_POLICY['max_avg_total_tokens_increase_ratio']:.2%}"
            ),
        )
    if cost_increase_ratio is not None and cost_increase_ratio > GATE_POLICY["max_avg_cost_ms_increase_ratio"]:
        _append_reason(
            reasons,
            level="manual_review",
            rule="max_avg_cost_ms_increase_ratio",
            message=(
                f"候选平均耗时增加 {cost_increase_ratio:.2%}，超过建议阈值 "
                f"{GATE_POLICY['max_avg_cost_ms_increase_ratio']:.2%}"
            ),
        )

    if reasons:
        return "manual_review", comparison, reasons

    _append_reason(
        reasons,
        level="pass",
        rule="all_rules_passed",
        message="候选版本满足硬规则，且未发现需要人工确认的质量或成本回退",
    )
    return "pass", comparison, reasons


def create_eval_gate_decision(
    db: Session,
    *,
    baseline_run_id: str,
    candidate_run_id: str,
) -> AiEvalGateDecision:
    """比较两次已完成的 Harness 评测，并持久化门禁结论。"""
    if baseline_run_id == candidate_run_id:
        raise BusinessException(code=40022, message="基线评测运行和候选评测运行不能是同一条记录")

    baseline_run = _get_eval_run(db, baseline_run_id, "基线")
    candidate_run = _get_eval_run(db, candidate_run_id, "候选")
    if baseline_run.prompt_name != candidate_run.prompt_name:
        raise BusinessException(
            code=40023,
            message=(
                "只能比较同一业务场景的不同 Prompt 版本；"
                "请保持 prompt_name 一致（例如 work_order_analysis），仅升级 prompt_version"
            ),
        )
    if baseline_run.dataset_version != candidate_run.dataset_version:
        raise BusinessException(code=40024, message="两次评测的数据集版本不同，不能比较")

    baseline_cases = _get_run_case_results(db, baseline_run_id)
    candidate_cases = _get_run_case_results(db, candidate_run_id)
    if {case.sample_id for case in baseline_cases} != {case.sample_id for case in candidate_cases}:
        raise BusinessException(
            code=40025,
            message="两次评测的样本集合不同，请基于当前数据集重新执行基线评测后再比较",
        )

    decision, comparison, reasons = _build_gate_judgement(
        baseline_run,
        candidate_run,
        candidate_cases,
    )
    gate_decision = AiEvalGateDecision(
        gate_id=next_snowflake_id(),
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        prompt_name=baseline_run.prompt_name,
        dataset_version=baseline_run.dataset_version,
        decision=decision,
        comparison_json=json.dumps(comparison, ensure_ascii=False),
        reason_json=json.dumps(reasons, ensure_ascii=False),
        rule_snapshot_json=json.dumps(GATE_POLICY, ensure_ascii=False),
    )
    db.add(gate_decision)
    db.commit()
    db.refresh(gate_decision)
    return gate_decision


def list_eval_gate_decisions(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    prompt_name: str | None = None,
    decision: str | None = None,
) -> tuple[list[AiEvalGateDecision], int]:
    filters = []
    if prompt_name:
        filters.append(AiEvalGateDecision.prompt_name == prompt_name)
    if decision:
        filters.append(AiEvalGateDecision.decision == decision)

    total = db.scalar(select(func.count()).select_from(AiEvalGateDecision).where(*filters)) or 0
    statement = (
        select(AiEvalGateDecision)
        .where(*filters)
        .order_by(AiEvalGateDecision.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total
