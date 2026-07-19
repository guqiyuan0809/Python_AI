"""
AI 调用日志服务层

用于记录模型调用成本、耗时和异常。
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.models import AiCallLog
from day04_app.utils.snowflake_id import next_snowflake_id


def create_call_log(
    db: Session,
    call_type: str,
    trace_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
    model: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_ms: int | None = None,
    status: str = "success",
    error_message: str | None = None,
) -> AiCallLog:
    # call_id 使用雪花 ID，后续可以跨 Java、Python、消息队列统一追踪。
    call_log = AiCallLog(
        call_id=next_snowflake_id(),
        trace_id=trace_id,
        session_id=session_id,
        message_id=message_id,
        call_type=call_type,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_ms=cost_ms,
        status=status,
        error_message=error_message,
    )
    db.add(call_log)
    db.commit()
    db.refresh(call_log)
    return call_log


def list_call_logs(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    trace_id: str | None = None,
    session_id: str | None = None,
    status: str | None = None,
) -> tuple[list[AiCallLog], int]:
    filters = []
    if trace_id:
        filters.append(AiCallLog.trace_id == trace_id)
    if session_id:
        filters.append(AiCallLog.session_id == session_id)
    if status:
        filters.append(AiCallLog.status == status)

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
