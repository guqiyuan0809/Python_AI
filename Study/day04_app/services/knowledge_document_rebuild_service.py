"""文档候选版本重建：当前原文不变，仅重切 chunk 和重建向量。"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import KnowledgeDocument, KnowledgeDocumentVersion


VERSION_STATUS_PARSED = "parsed"


def create_candidate_document_version(
    db: Session,
    *,
    document_id: str,
    change_note: str,
) -> KnowledgeDocumentVersion:
    """以当前已解析原文创建新候选版本，不能修改 active 指针。"""
    document = db.scalar(
        select(KnowledgeDocument).where(KnowledgeDocument.document_id == document_id)
    )
    if document is None:
        raise BusinessException(code=40451, message="知识库文档不存在")
    if document.status != VERSION_STATUS_PARSED or document.parsed_segment_count <= 0:
        raise BusinessException(code=40965, message="文档尚未完成原文解析，不能创建候选索引版本")

    latest_version_number = db.scalar(
        select(func.max(KnowledgeDocumentVersion.version_number)).where(
            KnowledgeDocumentVersion.document_id == document.document_id
        )
    )
    candidate = KnowledgeDocumentVersion(
        version_id=uuid4().hex,
        document_id=document.document_id,
        version_number=(latest_version_number or 0) + 1,
        status=VERSION_STATUS_PARSED,
        source_sha256=document.content_sha256,
        rebuild_note=change_note,
        parser_name=document.parser_name,
        segment_count=document.parsed_segment_count,
        # 其余构建快照由后续 version chunk / vector-index 阶段填充。
        chunk_count=0,
        vector_count=0,
    )
    try:
        db.add(candidate)
        db.commit()
        db.refresh(candidate)
        return candidate
    except Exception:
        db.rollback()
        raise
