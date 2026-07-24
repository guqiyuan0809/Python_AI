"""
评测结果查询服务。

用于管理端查看 harness 历史运行和样本级评测明细。
"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.models import AiEvalCaseResult, AiEvalRun


def list_eval_runs(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
    dataset_version: str | None = None,
) -> tuple[list[AiEvalRun], int]:
    filters = []
    if prompt_name:
        filters.append(AiEvalRun.prompt_name == prompt_name)
    if prompt_version:
        filters.append(AiEvalRun.prompt_version == prompt_version)
    if dataset_version:
        filters.append(AiEvalRun.dataset_version == dataset_version)

    total_statement = select(func.count()).select_from(AiEvalRun).where(*filters)
    total = db.scalar(total_statement) or 0

    statement = (
        select(AiEvalRun)
        .where(*filters)
        .order_by(AiEvalRun.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total


def list_eval_case_results(
    db: Session,
    run_id: str,
    page: int = 1,
    page_size: int = 50,
    only_failed: bool = False,
) -> tuple[list[AiEvalCaseResult], int]:
    filters = [AiEvalCaseResult.run_id == run_id]
    if only_failed:
        # 只要 schema 或任一业务字段未命中，就算本条样本需要复盘。
        filters.append(
            (AiEvalCaseResult.schema_valid == 0)
            | (AiEvalCaseResult.category_match == 0)
            | (AiEvalCaseResult.risk_level_match == 0)
            | (AiEvalCaseResult.human_review_match == 0)
        )

    total_statement = select(func.count()).select_from(AiEvalCaseResult).where(*filters)
    total = db.scalar(total_statement) or 0

    statement = (
        select(AiEvalCaseResult)
        .where(*filters)
        .order_by(AiEvalCaseResult.id.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total
