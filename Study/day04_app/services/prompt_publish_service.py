"""Prompt 人工发布服务。"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import (
    AiEvalGateDecision,
    AiEvalRun,
    AiPromptPublishAudit,
    AiPromptRollbackAudit,
    AiPromptVersion,
)
from day04_app.utils.snowflake_id import next_snowflake_id


PUBLISHABLE_GATE_DECISIONS = {"pass", "manual_review"}


def _get_gate_decision_for_publish(db: Session, gate_id: str) -> AiEvalGateDecision:
    gate_decision = db.scalar(
        select(AiEvalGateDecision)
        .where(AiEvalGateDecision.gate_id == gate_id)
        .with_for_update()
    )
    if gate_decision is None:
        raise BusinessException(code=40031, message="评测门禁记录不存在")
    if gate_decision.decision not in PUBLISHABLE_GATE_DECISIONS:
        raise BusinessException(code=40032, message="当前 Gate 结论为 reject，禁止发布候选 Prompt")
    return gate_decision


def publish_prompt_version(
    db: Session,
    *,
    prompt_id: str,
    gate_id: str,
    approval_note: str,
    approved_by: str,
) -> AiPromptPublishAudit:
    """人工批准候选 Prompt：归档旧 active、启用候选 draft，并写入审计记录。"""
    try:
        # 锁住候选版本，避免两个发布请求同时把同一 Prompt 或不同候选设为 active。
        candidate = db.scalar(
            select(AiPromptVersion)
            .where(AiPromptVersion.prompt_id == prompt_id)
            .with_for_update()
        )
        if candidate is None:
            raise BusinessException(code=40033, message="候选 Prompt 不存在")
        if candidate.status != "draft":
            raise BusinessException(code=40034, message="只有 draft 状态的候选 Prompt 可以发布")

        gate_decision = _get_gate_decision_for_publish(db, gate_id)
        candidate_run = db.scalar(
            select(AiEvalRun)
            .where(AiEvalRun.run_id == gate_decision.candidate_run_id)
            .with_for_update()
        )
        if candidate_run is None:
            raise BusinessException(code=40035, message="Gate 关联的候选评测运行不存在")
        if (
            candidate_run.prompt_name != candidate.prompt_name
            or candidate_run.prompt_version != candidate.prompt_version
        ):
            raise BusinessException(code=40036, message="Gate 关联的候选评测与待发布 Prompt 版本不一致")

        existing_audit = db.scalar(
            select(AiPromptPublishAudit)
            .where(AiPromptPublishAudit.prompt_id == candidate.prompt_id)
            .where(AiPromptPublishAudit.gate_id == gate_id)
        )
        if existing_audit is not None:
            raise BusinessException(code=40037, message="该 Gate 已用于发布当前 Prompt，不能重复发布")

        # 同一业务能力只能保留一个线上版本；旧版本保留内容但不再参与线上调用。
        previous_active_prompts = list(
            db.scalars(
                select(AiPromptVersion)
                .where(AiPromptVersion.prompt_name == candidate.prompt_name)
                .where(AiPromptVersion.status == "active")
                .with_for_update()
            ).all()
        )
        previous_versions = [item.prompt_version for item in previous_active_prompts]
        for previous_prompt in previous_active_prompts:
            previous_prompt.status = "archived"
        candidate.status = "active"

        audit = AiPromptPublishAudit(
            publish_id=next_snowflake_id(),
            gate_id=gate_decision.gate_id,
            prompt_id=candidate.prompt_id,
            prompt_name=candidate.prompt_name,
            candidate_prompt_version=candidate.prompt_version,
            previous_prompt_version=",".join(previous_versions) if previous_versions else None,
            gate_decision=gate_decision.decision,
            approval_note=approval_note,
            approved_by=approved_by,
        )
        db.add(audit)
        db.commit()
        db.refresh(audit)
        return audit
    except Exception:
        db.rollback()
        raise


def list_prompt_publish_audits(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    prompt_name: str | None = None,
) -> tuple[list[AiPromptPublishAudit], int]:
    filters = []
    if prompt_name:
        filters.append(AiPromptPublishAudit.prompt_name == prompt_name)

    total = db.scalar(select(func.count()).select_from(AiPromptPublishAudit).where(*filters)) or 0
    statement = (
        select(AiPromptPublishAudit)
        .where(*filters)
        .order_by(AiPromptPublishAudit.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total


def rollback_prompt_version(
    db: Session,
    *,
    publish_id: str,
    rollback_reason: str,
    rolled_back_by: str,
) -> AiPromptRollbackAudit:
    """根据发布审计回滚到被归档的上一线上版本，并写入独立审计记录。"""
    try:
        publish_audit = db.scalar(
            select(AiPromptPublishAudit)
            .where(AiPromptPublishAudit.publish_id == publish_id)
            .with_for_update()
        )
        if publish_audit is None:
            raise BusinessException(code=40041, message="原发布审计记录不存在")
        if not publish_audit.previous_prompt_version:
            raise BusinessException(code=40042, message="首次发布没有可恢复的历史 Prompt 版本")

        existing_rollback = db.scalar(
            select(AiPromptRollbackAudit)
            .where(AiPromptRollbackAudit.publish_id == publish_id)
        )
        if existing_rollback is not None:
            raise BusinessException(code=40043, message="该发布记录已完成回滚，不能重复执行")

        # 锁住同一业务能力的所有版本，保证回滚期间不存在并发发布造成双 active。
        prompt_versions = list(
            db.scalars(
                select(AiPromptVersion)
                .where(AiPromptVersion.prompt_name == publish_audit.prompt_name)
                .with_for_update()
            ).all()
        )
        current_active_prompts = [item for item in prompt_versions if item.status == "active"]
        if len(current_active_prompts) != 1:
            raise BusinessException(code=40044, message="当前业务 Prompt 的 active 状态异常，无法安全回滚")
        current_active = current_active_prompts[0]
        if current_active.prompt_version != publish_audit.candidate_prompt_version:
            raise BusinessException(code=40045, message="当前线上版本已发生后续发布，不能按旧审计记录直接回滚")

        restore_prompt = next(
            (item for item in prompt_versions if item.prompt_version == publish_audit.previous_prompt_version),
            None,
        )
        if restore_prompt is None:
            raise BusinessException(code=40046, message="原线上 Prompt 版本不存在，无法回滚")
        if restore_prompt.status != "archived":
            raise BusinessException(code=40047, message="待恢复 Prompt 不是 archived 状态，拒绝覆盖当前状态")

        current_active.status = "archived"
        restore_prompt.status = "active"
        rollback_audit = AiPromptRollbackAudit(
            rollback_id=next_snowflake_id(),
            publish_id=publish_audit.publish_id,
            prompt_name=publish_audit.prompt_name,
            rolled_back_prompt_version=current_active.prompt_version,
            restored_prompt_version=restore_prompt.prompt_version,
            rollback_reason=rollback_reason,
            rolled_back_by=rolled_back_by,
        )
        db.add(rollback_audit)
        db.commit()
        db.refresh(rollback_audit)
        return rollback_audit
    except Exception:
        db.rollback()
        raise


def list_prompt_rollback_audits(
    db: Session,
    *,
    page: int = 1,
    page_size: int = 20,
    prompt_name: str | None = None,
) -> tuple[list[AiPromptRollbackAudit], int]:
    filters = []
    if prompt_name:
        filters.append(AiPromptRollbackAudit.prompt_name == prompt_name)

    total = db.scalar(select(func.count()).select_from(AiPromptRollbackAudit).where(*filters)) or 0
    statement = (
        select(AiPromptRollbackAudit)
        .where(*filters)
        .order_by(AiPromptRollbackAudit.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total
