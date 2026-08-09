"""
AI 调用日志服务层

用于记录模型调用成本、耗时和异常。
"""

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.models import AiAsyncTask, AiCallLog
from day04_app.utils.snowflake_id import next_snowflake_id


@dataclass(frozen=True)
class TraceMetricAggregate:
    """将阶段事件累计指标与端到端指标分开，避免父子摘要重复计数。"""

    event_total_tokens: int
    event_total_cost_ms: int
    end_to_end_total_tokens: int | None
    end_to_end_cost_ms: int | None
    end_to_end_metric_source: str
    safety_interception_count: int
    guardrail_stop_count: int


def create_call_log(
    db: Session,
    call_type: str,
    trace_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
    model: str | None = None,
    stage: str | None = None,
    prompt_id: str | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
    prompt_template_hash: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_ms: int | None = None,
    status: str = "success",
    error_type: str | None = None,
    error_message: str | None = None,
    detail: dict[str, Any] | None = None,
    commit: bool = True,
) -> AiCallLog:
    # call_id 使用雪花 ID，后续可以跨 Java、Python、消息队列统一追踪。
    call_log = AiCallLog(
        call_id=next_snowflake_id(),
        trace_id=trace_id,
        session_id=session_id,
        message_id=message_id,
        task_id=task_id,
        run_id=run_id,
        call_type=call_type,
        stage=stage,
        model=model,
        prompt_tokens=prompt_tokens,
        prompt_id=prompt_id,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        prompt_template_hash=prompt_template_hash,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_ms=cost_ms,
        status=status,
        error_type=error_type,
        error_message=error_message,
        detail_json=json.dumps(detail, ensure_ascii=False, sort_keys=True) if detail else None,
    )
    db.add(call_log)
    if commit:
        db.commit()
        db.refresh(call_log)
    return call_log


def load_call_log_detail(call_log: AiCallLog) -> dict[str, Any] | None:
    """恢复可观测详情；历史记录没有 detail_json 时保持兼容。"""
    if not call_log.detail_json:
        return None
    try:
        payload = json.loads(call_log.detail_json)
    except (TypeError, json.JSONDecodeError):
        return {"detail_parse_error": True}
    return payload if isinstance(payload, dict) else {"detail_parse_error": True}


def list_call_logs(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    trace_id: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    call_type: str | None = None,
    stage: str | None = None,
    status: str | None = None,
    error_type: str | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
) -> tuple[list[AiCallLog], int]:
    filters = []
    if trace_id:
        filters.append(AiCallLog.trace_id == trace_id)
    if session_id:
        filters.append(AiCallLog.session_id == session_id)
    if task_id:
        filters.append(AiCallLog.task_id == task_id)
    if call_type:
        filters.append(AiCallLog.call_type == call_type)
    if stage:
        filters.append(AiCallLog.stage == stage)
    if status:
        filters.append(AiCallLog.status == status)
    if error_type:
        filters.append(AiCallLog.error_type == error_type)
    if prompt_name:
        filters.append(AiCallLog.prompt_name == prompt_name)
    if prompt_version:
        filters.append(AiCallLog.prompt_version == prompt_version)

    total_statement = select(func.count()).select_from(AiCallLog).where(*filters)
    total = db.scalar(total_statement) or 0

    statement = (
        select(AiCallLog)
        .where(*filters)
        .order_by(AiCallLog.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total


def get_trace_observability(
    db: Session,
    *,
    trace_id: str,
) -> tuple[list[AiCallLog], list[AiAsyncTask]]:
    """以 trace_id 聚合 AI 调用事件与异步任务，不把业务正文返回给运维查询接口。"""
    call_logs = list(
        db.scalars(
            select(AiCallLog)
            .where(AiCallLog.trace_id == trace_id)
            .order_by(AiCallLog.created_at.asc(), AiCallLog.id.asc())
        )
    )
    tasks = list(
        db.scalars(
            select(AiAsyncTask)
            .where(AiAsyncTask.trace_id == trace_id)
            .order_by(AiAsyncTask.created_at.asc(), AiAsyncTask.id.asc())
        )
    )
    return call_logs, tasks


def build_trace_metric_aggregate(
    call_logs: list[AiCallLog],
    tasks: list[AiAsyncTask],
) -> TraceMetricAggregate:
    """构建链路聚合指标；事件明细可累加，端到端指标只取一个根来源。"""
    event_total_tokens = sum(call_log.total_tokens or 0 for call_log in call_logs)
    event_total_cost_ms = sum(call_log.cost_ms or 0 for call_log in call_logs)

    safety_interception_count = 0
    guardrail_stop_count = 0
    for call_log in call_logs:
        observation_status = (load_call_log_detail(call_log) or {}).get(
            "observation_status"
        )
        if observation_status in {"blocked", "require_confirm"}:
            safety_interception_count += 1
        elif observation_status == "stopped_by_guardrail":
            guardrail_stop_count += 1

    # 异步任务表是 Worker 的最终落点，优先作为线上端到端指标来源。
    task_has_final_metrics = any(
        task.total_tokens is not None or task.cost_ms is not None for task in tasks
    )
    if task_has_final_metrics:
        return TraceMetricAggregate(
            event_total_tokens=event_total_tokens,
            event_total_cost_ms=event_total_cost_ms,
            end_to_end_total_tokens=sum(task.total_tokens or 0 for task in tasks),
            end_to_end_cost_ms=sum(task.cost_ms or 0 for task in tasks),
            end_to_end_metric_source="async_task",
            safety_interception_count=safety_interception_count,
            guardrail_stop_count=guardrail_stop_count,
        )

    rag_summaries = [
        call_log
        for call_log in call_logs
        if call_log.stage == "rag_request_summary"
        and (load_call_log_detail(call_log) or {}).get("summary_type") == "rag_request"
    ]
    if len(rag_summaries) == 1:
        summary = rag_summaries[0]
        return TraceMetricAggregate(
            event_total_tokens=event_total_tokens,
            event_total_cost_ms=event_total_cost_ms,
            end_to_end_total_tokens=summary.total_tokens,
            end_to_end_cost_ms=summary.cost_ms,
            end_to_end_metric_source="rag_request_summary",
            safety_interception_count=safety_interception_count,
            guardrail_stop_count=guardrail_stop_count,
        )

    # 同步 Agent 没有 ai_async_task 时，Loop 自身的最终摘要是唯一根指标。
    agent_summaries = [
        call_log
        for call_log in call_logs
        if call_log.stage == "agent_loop_summary"
        and (load_call_log_detail(call_log) or {}).get("summary") is True
    ]
    if len(agent_summaries) == 1:
        summary = agent_summaries[0]
        return TraceMetricAggregate(
            event_total_tokens=event_total_tokens,
            event_total_cost_ms=event_total_cost_ms,
            end_to_end_total_tokens=summary.total_tokens,
            end_to_end_cost_ms=summary.cost_ms,
            end_to_end_metric_source="agent_loop_summary",
            safety_interception_count=safety_interception_count,
            guardrail_stop_count=guardrail_stop_count,
        )

    # 未记录根摘要的同步多阶段链路，不能把阶段耗时冒充端到端耗时。
    return TraceMetricAggregate(
        event_total_tokens=event_total_tokens,
        event_total_cost_ms=event_total_cost_ms,
        end_to_end_total_tokens=None,
        end_to_end_cost_ms=None,
        end_to_end_metric_source="unavailable",
        safety_interception_count=safety_interception_count,
        guardrail_stop_count=guardrail_stop_count,
    )
