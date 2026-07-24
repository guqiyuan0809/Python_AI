"""
AI 失败样本服务层。

用于保存结构化输出失败时的原始模型输出、校验错误和 schema 版本。
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.models import AiFailureSample
from day04_app.utils.snowflake_id import next_snowflake_id


def create_failure_sample(
    db: Session,
    *,
    call_type: str,
    schema_type: str,
    schema_version: str,
    error_type: str,
    error_message: str,
    raw_text: str | None = None,
    validation_error: str | None = None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
    model: str | None = None,
) -> AiFailureSample:
    # 失败样本保存的是模型原始输出和校验错误，用于离线分析，不直接返回给普通用户。
    sample = AiFailureSample(
        sample_id=next_snowflake_id(),
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
        call_type=call_type,
        model=model,
        schema_type=schema_type,
        schema_version=schema_version,
        error_type=error_type,
        error_message=error_message,
        raw_text=raw_text,
        validation_error=validation_error,
    )
    db.add(sample)
    db.commit()
    db.refresh(sample)
    return sample


def list_failure_samples(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    trace_id: str | None = None,
    schema_type: str | None = None,
    error_type: str | None = None,
) -> tuple[list[AiFailureSample], int]:
    filters = []
    if trace_id:
        filters.append(AiFailureSample.trace_id == trace_id)
    if schema_type:
        filters.append(AiFailureSample.schema_type == schema_type)
    if error_type:
        filters.append(AiFailureSample.error_type == error_type)

    total_statement = select(func.count()).select_from(AiFailureSample).where(*filters)
    total = db.scalar(total_statement) or 0

    statement = (
        select(AiFailureSample)
        .where(*filters)
        .order_by(AiFailureSample.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total
