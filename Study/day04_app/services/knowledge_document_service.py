"""知识库文档主记录服务。"""

from sqlalchemy.orm import Session
from uuid import uuid4

from day04_app.models import KnowledgeDocument, KnowledgeDocumentVersion
from day04_app.services.document_storage_service import StoredDocument


DOCUMENT_STATUS_UPLOADED = "uploaded"


def create_uploaded_document(
    db: Session,
    stored_document: StoredDocument,
    trace_id: str | None,
) -> KnowledgeDocument:
    """在文件通过校验后写入文档主记录，作为后续解析任务的唯一数据来源。"""
    document = KnowledgeDocument(
        document_id=stored_document.document_id,
        original_file_name=stored_document.original_file_name,
        file_type=stored_document.file_type,
        storage_key=stored_document.storage_key,
        file_size=stored_document.file_size,
        content_sha256=stored_document.content_sha256,
        trace_id=trace_id,
        status=DOCUMENT_STATUS_UPLOADED,
    )
    # 上传即创建 v1 候选版本；它还未解析、切块或写入 Milvus，因此绝不能直接标记为 active。
    initial_version = KnowledgeDocumentVersion(
        version_id=uuid4().hex,
        document_id=stored_document.document_id,
        version_number=1,
        status=DOCUMENT_STATUS_UPLOADED,
        source_sha256=stored_document.content_sha256,
    )
    try:
        db.add_all([document, initial_version])
        db.commit()
        db.refresh(document)
        return document
    except Exception:
        db.rollback()
        raise
