"""授权审计写入与查询服务。"""

import json
import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.models import AiSecurityAuditLog
from day04_app.security.principal import SecurityPrincipal
from day04_app.utils.snowflake_id import next_snowflake_id


logger = logging.getLogger("day04_app.security")


def create_security_audit(
    db: Session,
    *,
    trace_id: str | None,
    principal: SecurityPrincipal | None,
    permission: str,
    http_method: str,
    request_path: str,
    decision: str,
    reason: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    credential_id: str | None = None,
) -> None:
    """安全审计是旁路事实；写入失败不能把 401 错误变成 500。"""
    try:
        db.add(
            AiSecurityAuditLog(
                audit_id=next_snowflake_id(),
                trace_id=trace_id,
                actor_id=principal.actor_id if principal else None,
                # 历史字段名保留兼容性：直连模式记录 API Key ID，代理模式记录服务 Key ID。
                api_key_id=principal.api_key_id if principal else credential_id,
                roles_json=json.dumps(list(principal.roles) if principal else []),
                permission=permission,
                http_method=http_method,
                request_path=request_path,
                decision=decision,
                reason=reason,
                resource_type=resource_type,
                resource_id=resource_id,
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("security audit persistence failed")


def list_security_audits(
    db: Session,
    *,
    page: int,
    page_size: int,
    trace_id: str | None = None,
    actor_id: str | None = None,
    permission: str | None = None,
    decision: str | None = None,
) -> tuple[list[AiSecurityAuditLog], int]:
    filters = []
    if trace_id:
        filters.append(AiSecurityAuditLog.trace_id == trace_id)
    if actor_id:
        filters.append(AiSecurityAuditLog.actor_id == actor_id)
    if permission:
        filters.append(AiSecurityAuditLog.permission == permission)
    if decision:
        filters.append(AiSecurityAuditLog.decision == decision)

    total = db.scalar(
        select(func.count()).select_from(AiSecurityAuditLog).where(*filters)
    ) or 0
    items = list(
        db.scalars(
            select(AiSecurityAuditLog)
            .where(*filters)
            .order_by(AiSecurityAuditLog.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).all()
    )
    return items, total
