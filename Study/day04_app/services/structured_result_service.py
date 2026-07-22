"""
AI 结构化结果服务层。

类似 Java 里的 StructuredResultService，负责把模型生成的标准 JSON 结果持久化。
"""

import json
from typing import Any

from sqlalchemy.orm import Session
from sqlalchemy import select

from day04_app.models import AiStructuredResult
from day04_app.utils.snowflake_id import next_snowflake_id


def create_structured_result(
    db: Session,
    *,
    business_type: str,
    schema_type: str,
    result_data: dict[str, Any] | None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
    business_id: str | None = None,
    schema_version: str = "v1",
    status: str = "success",
    error_message: str | None = None,
) -> AiStructuredResult:
    # result_json 只保存通过 Pydantic 校验后的标准结果，避免把模型原始脏输出写进业务结果表。
    result_json = None
    if result_data is not None:
        result_json = json.dumps(result_data, ensure_ascii=False)

    structured_result = AiStructuredResult(
        result_id=next_snowflake_id(),
        task_id=task_id,
        trace_id=trace_id,
        session_id=session_id,
        message_id=message_id,
        business_type=business_type,
        business_id=business_id,
        schema_type=schema_type,
        schema_version=schema_version,
        result_json=result_json,
        status=status,
        error_message=error_message,
    )
    db.add(structured_result)
    db.commit()
    db.refresh(structured_result)
    return structured_result


def get_structured_result_by_task_id(
    db: Session,
    task_id: str,
) -> AiStructuredResult | None:
    statement = (
        select(AiStructuredResult)
        .where(AiStructuredResult.task_id == task_id)
        .order_by(AiStructuredResult.id.desc())
    )
    return db.scalars(statement).first()


def load_result_json(structured_result: AiStructuredResult | None) -> dict | None:
    if structured_result is None or structured_result.result_json is None:
        return None
    return json.loads(structured_result.result_json)
