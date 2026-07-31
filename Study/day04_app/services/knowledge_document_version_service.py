"""知识库文档版本的校验与激活切换服务。"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import (
    KnowledgeDocument,
    KnowledgeDocumentVersion,
    KnowledgeDocumentVersionActivationAudit,
)
from day04_app.services.milvus_vector_store_service import count_vectors_by_version


VERSION_STATUS_INDEXED = "indexed"
VERSION_STATUS_ACTIVE = "active"
VERSION_STATUS_RETIRED = "retired"


def activate_document_version(
    db: Session,
    version_id: str,
    activated_by: str,
    activation_note: str,
) -> tuple[KnowledgeDocument, KnowledgeDocumentVersion, str | None]:
    """确认向量数量后原子切换 MySQL active 指针，旧版本不立即删除以便回滚。"""
    candidate = db.scalar(
        select(KnowledgeDocumentVersion).where(KnowledgeDocumentVersion.version_id == version_id)
    )
    if candidate is None:
        raise BusinessException(code=40452, message="知识库文档版本不存在")
    if candidate.status != VERSION_STATUS_INDEXED:
        raise BusinessException(code=40958, message="只有 indexed 状态的版本可以切换为 active")
    if candidate.vector_count != candidate.chunk_count:
        raise BusinessException(code=40959, message="版本的 MySQL chunk 数与已记录向量数不一致，拒绝切换")

    # MySQL 状态可能陈旧，切换前必须再从 Milvus 查询，避免向量意外丢失后仍发布空索引。
    actual_vector_count = count_vectors_by_version(candidate.version_id)
    if actual_vector_count != candidate.chunk_count:
        raise BusinessException(
            code=40959,
            message=f"Milvus 向量数校验失败：期望 {candidate.chunk_count}，实际 {actual_vector_count}",
        )

    document = db.scalar(
        select(KnowledgeDocument).where(KnowledgeDocument.document_id == candidate.document_id)
    )
    if document is None:
        raise BusinessException(code=40451, message="知识库文档不存在")

    previous_version_id = document.active_version_id
    now = datetime.now()
    try:
        if previous_version_id:
            previous_version = db.scalar(
                select(KnowledgeDocumentVersion).where(
                    KnowledgeDocumentVersion.version_id == previous_version_id
                )
            )
            if previous_version and previous_version.version_id != candidate.version_id:
                previous_version.status = VERSION_STATUS_RETIRED

        candidate.status = VERSION_STATUS_ACTIVE
        candidate.activated_at = now
        candidate.error_message = None
        document.active_version_id = candidate.version_id
        # 审计记录与状态指针同事务提交，保证“实际发布”和“谁批准发布”不可分离。
        db.add(
            KnowledgeDocumentVersionActivationAudit(
                activation_id=uuid4().hex,
                document_id=document.document_id,
                activated_version_id=candidate.version_id,
                previous_version_id=previous_version_id,
                activated_by=activated_by,
                activation_note=activation_note,
                activated_at=now,
            )
        )
        db.commit()
        db.refresh(document)
        db.refresh(candidate)
        return document, candidate, previous_version_id
    except Exception:
        db.rollback()
        raise
