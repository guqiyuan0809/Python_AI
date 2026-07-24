"""
评测主数据服务。

负责读取 prompt 版本、评测数据集和样本。类似 Java 中的 PromptVersionService / EvalSampleService。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.models import AiEvalDataset, AiEvalSample, AiPromptVersion


@dataclass
class EvalSampleDTO:
    sample_id: str
    input: str
    expected: dict[str, Any]
    sample_type: str


def get_active_prompt_version(
    db: Session,
    prompt_name: str,
    prompt_version: str,
) -> AiPromptVersion:
    prompt = db.scalar(
        select(AiPromptVersion)
        .where(AiPromptVersion.prompt_name == prompt_name)
        .where(AiPromptVersion.prompt_version == prompt_version)
        .where(AiPromptVersion.status == "active")
    )
    if not prompt:
        raise RuntimeError(f"未找到可用 Prompt：{prompt_name}:{prompt_version}")
    return prompt


def list_active_eval_samples(db: Session, dataset_version: str) -> list[EvalSampleDTO]:
    rows = db.scalars(
        select(AiEvalSample)
        .where(AiEvalSample.dataset_version == dataset_version)
        .where(AiEvalSample.status == "active")
        .order_by(AiEvalSample.sample_id.asc())
    ).all()
    if not rows:
        raise RuntimeError(f"未找到可用评测样本：{dataset_version}")

    samples: list[EvalSampleDTO] = []
    for row in rows:
        samples.append(
            EvalSampleDTO(
                sample_id=row.sample_id,
                input=row.input_text,
                expected=json.loads(row.expected_json),
                sample_type=row.sample_type,
            )
        )
    return samples


def list_prompt_versions(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    prompt_name: str | None = None,
    status: str | None = None,
) -> tuple[list[AiPromptVersion], int]:
    filters = []
    if prompt_name:
        filters.append(AiPromptVersion.prompt_name == prompt_name)
    if status:
        filters.append(AiPromptVersion.status == status)

    total_statement = select(func.count()).select_from(AiPromptVersion).where(*filters)
    total = db.scalar(total_statement) or 0

    statement = (
        select(AiPromptVersion)
        .where(*filters)
        .order_by(AiPromptVersion.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total


def list_eval_datasets(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    dataset_name: str | None = None,
    status: str | None = None,
) -> tuple[list[AiEvalDataset], int]:
    filters = []
    if dataset_name:
        filters.append(AiEvalDataset.dataset_name == dataset_name)
    if status:
        filters.append(AiEvalDataset.status == status)

    total_statement = select(func.count()).select_from(AiEvalDataset).where(*filters)
    total = db.scalar(total_statement) or 0

    statement = (
        select(AiEvalDataset)
        .where(*filters)
        .order_by(AiEvalDataset.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total


def list_eval_samples_page(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    dataset_version: str | None = None,
    sample_type: str | None = None,
    status: str | None = None,
) -> tuple[list[AiEvalSample], int]:
    filters = []
    if dataset_version:
        filters.append(AiEvalSample.dataset_version == dataset_version)
    if sample_type:
        filters.append(AiEvalSample.sample_type == sample_type)
    if status:
        filters.append(AiEvalSample.status == status)

    total_statement = select(func.count()).select_from(AiEvalSample).where(*filters)
    total = db.scalar(total_statement) or 0

    statement = (
        select(AiEvalSample)
        .where(*filters)
        .order_by(AiEvalSample.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total
