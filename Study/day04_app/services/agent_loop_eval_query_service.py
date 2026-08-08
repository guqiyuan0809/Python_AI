"""Agent Loop Harness 历史结果查询服务。"""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.models import AiAgentEvalCaseResult, AiAgentEvalRun


def list_agent_eval_runs(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    agent_name: str | None = None,
    agent_version: str | None = None,
    dataset_version: str | None = None,
) -> tuple[list[AiAgentEvalRun], int]:
    filters = []
    if agent_name:
        filters.append(AiAgentEvalRun.agent_name == agent_name)
    if agent_version:
        filters.append(AiAgentEvalRun.agent_version == agent_version)
    if dataset_version:
        filters.append(AiAgentEvalRun.dataset_version == dataset_version)
    total = db.scalar(select(func.count()).select_from(AiAgentEvalRun).where(*filters)) or 0
    rows = db.scalars(
        select(AiAgentEvalRun)
        .where(*filters)
        .order_by(AiAgentEvalRun.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return list(rows), total


def list_agent_eval_case_results(
    db: Session,
    *,
    run_id: str,
    page: int = 1,
    page_size: int = 50,
    only_failed: bool = False,
) -> tuple[list[AiAgentEvalCaseResult], int]:
    filters = [AiAgentEvalCaseResult.run_id == run_id]
    if only_failed:
        filters.append(AiAgentEvalCaseResult.case_pass == 0)
    total = db.scalar(
        select(func.count()).select_from(AiAgentEvalCaseResult).where(*filters)
    ) or 0
    rows = db.scalars(
        select(AiAgentEvalCaseResult)
        .where(*filters)
        .order_by(AiAgentEvalCaseResult.id.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return list(rows), total
