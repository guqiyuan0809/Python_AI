"""
AI 失败样本服务层。

用于保存结构化输出失败时的原始模型输出、校验错误和 schema 版本。
"""

import json

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import AiEvalDataset, AiEvalSample, AiFailureSample
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


def get_failure_sample(db: Session, sample_id: str) -> AiFailureSample:
    statement = select(AiFailureSample).where(AiFailureSample.sample_id == sample_id)
    sample = db.scalars(statement).first()
    if sample is None:
        raise BusinessException(code=40011, message="失败样本不存在")
    return sample


def resolve_target_eval_dataset(
    db: Session,
    *,
    failure_sample: AiFailureSample,
    dataset_id: str | None,
    dataset_version: str | None,
) -> AiEvalDataset:
    """解析人工选择或失败样本默认对应的启用评测数据集。"""
    if bool(dataset_id) != bool(dataset_version):
        raise BusinessException(code=40014, message="dataset_id 和 dataset_version 必须同时传入或同时不传")

    if dataset_id and dataset_version:
        dataset = db.scalars(
            select(AiEvalDataset).where(
                AiEvalDataset.dataset_id == dataset_id,
                AiEvalDataset.dataset_version == dataset_version,
            )
        ).first()
        if dataset is None:
            raise BusinessException(code=40012, message="目标评测数据集不存在")
        return dataset

    # schema_type/schema_version 是失败样本所属的结构化能力版本，约定映射为数据集名和版本。
    default_version = f"{failure_sample.schema_type}_{failure_sample.schema_version}"
    dataset = db.scalars(
        select(AiEvalDataset).where(
            AiEvalDataset.dataset_name == failure_sample.schema_type,
            AiEvalDataset.dataset_version == default_version,
            AiEvalDataset.status == "active",
        )
    ).first()
    if dataset is None:
        raise BusinessException(
            code=40015,
            message="未找到失败样本对应的启用评测数据集，请先在管理端选择数据集",
        )
    return dataset


def convert_failure_sample_to_eval_sample(
    db: Session,
    *,
    failure_sample_id: str,
    dataset_id: str | None,
    dataset_version: str | None,
    sample_type: str,
    input_text: str,
    expected: dict,
) -> AiEvalSample:
    """把线上失败样本经人工标注后转入正式评测样本库。"""
    failure_sample = get_failure_sample(db, failure_sample_id)
    dataset = resolve_target_eval_dataset(
        db,
        failure_sample=failure_sample,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
    )

    existing_sample = db.scalars(
        select(AiEvalSample).where(
            AiEvalSample.source_type == "failure_sample",
            AiEvalSample.source_ref_id == failure_sample.sample_id,
            AiEvalSample.dataset_version == dataset.dataset_version,
        )
    ).first()
    if existing_sample:
        raise BusinessException(code=40013, message="该失败样本已转入当前评测数据集")

    eval_sample = AiEvalSample(
        sample_id=next_snowflake_id(),
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        sample_type=sample_type,
        input_text=input_text,
        expected_json=json.dumps(expected, ensure_ascii=False),
        source_type="failure_sample",
        source_ref_id=failure_sample.sample_id,
        status="active",
        created_by="manual_review",
    )
    dataset.sample_count += 1
    db.add(eval_sample)
    db.commit()
    db.refresh(eval_sample)
    return eval_sample
