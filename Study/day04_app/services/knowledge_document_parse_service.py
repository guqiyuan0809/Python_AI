"""通过文档业务 ID 执行解析并持久化原始文本段。"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import (
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    KnowledgeDocumentSegment,
    KnowledgeDocumentVersion,
)
from day04_app.schemas.knowledge_schema import ParsedDocument
from day04_app.services.document_parser_service import document_parser_registry
from settings import settings


DOCUMENT_STATUS_PARSING = "parsing"
DOCUMENT_STATUS_PARSED = "parsed"
DOCUMENT_STATUS_ERROR = "error"


def _find_document(db: Session, document_id: str) -> KnowledgeDocument:
    document = db.scalar(
        select(KnowledgeDocument).where(KnowledgeDocument.document_id == document_id)
    )
    if document is None:
        raise BusinessException(code=40451, message="知识库文档不存在")
    return document


def _resolve_storage_path(document: KnowledgeDocument) -> Path:
    """由可信的 storage_key 推导路径，并再次确保路径没有逃出上传根目录。"""
    storage_root = settings.knowledge_upload_dir.resolve()
    file_path = (storage_root / document.storage_key).resolve()
    if storage_root not in file_path.parents:
        raise BusinessException(code=40060, message="文档存储键非法，拒绝读取文件")
    return file_path


def _mark_parse_error(db: Session, document_id: str, error_message: str) -> None:
    """解析异常使用独立提交记录 error，避免主解析事务回滚后丢失排查依据。"""
    document = _find_document(db, document_id)
    document.status = DOCUMENT_STATUS_ERROR
    document.error_message = error_message[:2000]
    db.commit()


def parse_document_by_id(db: Session, document_id: str) -> tuple[KnowledgeDocument, ParsedDocument]:
    """解析指定文档，并以一个数据库事务替换该文档的全部原始文本段。"""
    document = _find_document(db, document_id)
    if document.status == DOCUMENT_STATUS_PARSING:
        raise BusinessException(code=40951, message="文档正在解析中，请勿重复提交")
    if document.active_version_id:
        # 当前原始段尚未携带 version_id，直接重解析会删除 active 版本的 MySQL 来源数据。
        raise BusinessException(
            code=40968,
            message="文档已有 active 版本，当前仅支持创建候选版本重切块；原文件更新需走后续完整版本化流程",
        )

    document.status = DOCUMENT_STATUS_PARSING
    document.error_message = None
    db.commit()

    try:
        parsed_document = document_parser_registry.parse(
            document_id=document.document_id,
            file_path=_resolve_storage_path(document),
        )
        # 解析器只知道物理文件；对外展示的名称必须回到文档主记录中的原始文件名。
        parsed_document = parsed_document.model_copy(
            update={"file_name": document.original_file_name}
        )

        # 删除旧段、写入新段和更新解析状态处于同一事务，避免读取到半份解析结果。
        db.execute(
            delete(KnowledgeDocumentSegment).where(
                KnowledgeDocumentSegment.document_id == document.document_id
            )
        )
        # 原始段重新解析后，旧 chunk 已不再对应当前内容，必须在同一事务中失效。
        db.execute(
            delete(KnowledgeDocumentChunk).where(
                KnowledgeDocumentChunk.document_id == document.document_id
            )
        )
        db.add_all(
            [
                KnowledgeDocumentSegment(
                    document_id=document.document_id,
                    segment_index=segment.segment_index,
                    content=segment.text,
                    location=segment.location,
                    metadata_json=json.dumps(segment.metadata, ensure_ascii=False),
                )
                for segment in parsed_document.segments
            ]
        )
        document.status = DOCUMENT_STATUS_PARSED
        document.parser_name = parsed_document.parser_name
        document.parsed_segment_count = len(parsed_document.segments)
        document.chunk_status = "not_started"
        document.chunk_count = 0
        document.chunk_config_json = None
        document.chunk_error_message = None
        document.chunked_at = None
        document.error_message = None
        # 当前同步链路将解析结果写入最新候选版本；active 版本的切换仍必须在向量验证后单独执行。
        latest_version = db.scalar(
            select(KnowledgeDocumentVersion)
            .where(KnowledgeDocumentVersion.document_id == document.document_id)
            .order_by(KnowledgeDocumentVersion.version_number.desc())
        )
        if latest_version is None:
            raise BusinessException(code=40955, message="文档尚未创建索引版本，不能保存解析结果")
        latest_version.status = "parsed"
        latest_version.parser_name = parsed_document.parser_name
        latest_version.segment_count = len(parsed_document.segments)
        latest_version.chunk_config_json = None
        latest_version.chunk_count = 0
        latest_version.vector_count = 0
        latest_version.embedding_model = None
        latest_version.embedding_dimension = None
        latest_version.vector_collection = None
        latest_version.error_message = None
        db.commit()
        db.refresh(document)
        return document, parsed_document
    except BusinessException as exc:
        db.rollback()
        _mark_parse_error(db, document_id, exc.message)
        raise
    except Exception as exc:
        db.rollback()
        _mark_parse_error(db, document_id, "文档解析发生未预期错误")
        raise BusinessException(code=50051, message="文档解析失败，请查看服务日志") from exc
