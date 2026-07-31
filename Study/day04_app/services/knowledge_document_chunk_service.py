"""基于已持久化原始段生成并保存检索文本切块。"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import (
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    KnowledgeDocumentSegment,
    KnowledgeDocumentVersion,
)
from day04_app.schemas.knowledge_schema import KnowledgeTextChunk, ParsedDocumentSegment
from day04_app.services.text_chunker_service import ChunkingConfig, build_text_chunks


DOCUMENT_STATUS_PARSED = "parsed"
CHUNK_STATUS_NOT_STARTED = "not_started"
CHUNK_STATUS_CHUNKING = "chunking"
CHUNK_STATUS_CHUNKED = "chunked"
CHUNK_STATUS_ERROR = "error"


def _find_document(db: Session, document_id: str) -> KnowledgeDocument:
    document = db.scalar(
        select(KnowledgeDocument).where(KnowledgeDocument.document_id == document_id)
    )
    if document is None:
        raise BusinessException(code=40451, message="知识库文档不存在")
    return document


def _load_parsed_segments(db: Session, document_id: str) -> list[ParsedDocumentSegment]:
    segment_records = list(
        db.scalars(
            select(KnowledgeDocumentSegment)
            .where(KnowledgeDocumentSegment.document_id == document_id)
            .order_by(KnowledgeDocumentSegment.segment_index)
        )
    )
    if not segment_records:
        raise BusinessException(code=40067, message="文档尚未产生原始文本段，不能切块")
    return [
        ParsedDocumentSegment(
            segment_index=segment.segment_index,
            text=segment.content,
            location=segment.location,
        )
        for segment in segment_records
    ]


def _find_version(
    db: Session,
    *,
    document_id: str,
    version_id: str,
) -> KnowledgeDocumentVersion:
    """指定版本必须属于当前文档，不能由 URL 的 version_id 越权写入其他文档。"""
    version = db.scalar(
        select(KnowledgeDocumentVersion).where(
            KnowledgeDocumentVersion.document_id == document_id,
            KnowledgeDocumentVersion.version_id == version_id,
        )
    )
    if version is None:
        raise BusinessException(code=40452, message="知识库文档版本不存在或不属于当前文档")
    if version.status not in {DOCUMENT_STATUS_PARSED, CHUNK_STATUS_CHUNKED, CHUNK_STATUS_ERROR}:
        raise BusinessException(code=40966, message=f"当前版本状态为 {version.status}，不能执行切块")
    return version


def _mark_version_chunk_error(db: Session, version_id: str, error_message: str) -> None:
    """候选版本失败只更新自身状态，不能污染正在提供检索服务的 active 版本。"""
    version = db.scalar(
        select(KnowledgeDocumentVersion).where(KnowledgeDocumentVersion.version_id == version_id)
    )
    if version is not None:
        version.status = CHUNK_STATUS_ERROR
        version.error_message = error_message[:2000]
    db.commit()


def chunk_document_by_id(
    db: Session,
    document_id: str,
    config: ChunkingConfig,
) -> tuple[KnowledgeDocument, list[KnowledgeTextChunk]]:
    """兼容 Day19 接口：只允许无 active 版本的文档切块，避免覆盖线上版本。"""
    document = _find_document(db, document_id)
    if document.active_version_id:
        raise BusinessException(code=40967, message="文档已有 active 版本，请创建候选版本后按版本切块")
    version = db.scalar(
        select(KnowledgeDocumentVersion)
        .where(KnowledgeDocumentVersion.document_id == document_id)
        .order_by(KnowledgeDocumentVersion.version_number.desc())
    )
    if version is None:
        raise BusinessException(code=40955, message="文档尚未创建索引版本，不能执行切块")
    return chunk_document_version_by_id(
        db,
        document,
        version.version_id,
        config,
        update_document_snapshot=True,
    )


def chunk_document_version_by_id(
    db: Session,
    document: KnowledgeDocument,
    version_id: str,
    config: ChunkingConfig,
    update_document_snapshot: bool = False,
) -> tuple[KnowledgeDocument, list[KnowledgeTextChunk]]:
    """只替换指定候选版本的 chunk，旧 active 版本的 MySQL 原文继续可回填。"""
    document_id = document.document_id
    if document.status != DOCUMENT_STATUS_PARSED:
        raise BusinessException(code=40952, message="文档尚未解析成功，不能执行切块")
    version = _find_version(db, document_id=document.document_id, version_id=version_id)
    version.status = CHUNK_STATUS_CHUNKING
    version.error_message = None
    db.commit()

    try:
        chunks = build_text_chunks(
            document_id=document.document_id,
            segments=_load_parsed_segments(db, document.document_id),
            config=config,
        )
        chunk_config_json = json.dumps(
            {
                "max_characters": config.max_characters,
                "overlap_characters": config.overlap_characters,
                "boundary_search_characters": config.boundary_search_characters,
            },
            ensure_ascii=False,
        )

        # 仅删除当前候选版本旧 chunk，绝不能按 document_id 删除 active 版本的回填原文。
        db.execute(
            delete(KnowledgeDocumentChunk).where(
                KnowledgeDocumentChunk.version_id == version.version_id
            )
        )
        db.add_all(
            [
                KnowledgeDocumentChunk(
                    chunk_id=uuid4().hex,
                    version_id=version.version_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    char_count=chunk.char_count,
                    source_references_json=json.dumps(
                        [reference.model_dump() for reference in chunk.source_references],
                        ensure_ascii=False,
                    ),
                )
                for chunk in chunks
            ]
        )
        version.status = CHUNK_STATUS_CHUNKED
        version.chunk_config_json = chunk_config_json
        version.chunk_count = len(chunks)
        version.vector_count = 0
        version.embedding_model = None
        version.embedding_dimension = None
        version.vector_collection = None
        version.error_message = None
        if update_document_snapshot:
            # Day19 初始构建仍维护主表快照；候选版本构建只看 version 状态，不能影响线上元数据。
            document.chunk_status = CHUNK_STATUS_CHUNKED
            document.chunk_count = len(chunks)
            document.chunk_config_json = chunk_config_json
            document.chunk_error_message = None
            document.chunked_at = datetime.now()
        db.commit()
        db.refresh(document)
        return document, chunks
    except BusinessException as exc:
        db.rollback()
        _mark_version_chunk_error(db, version_id, exc.message)
        raise
    except Exception as exc:
        db.rollback()
        _mark_version_chunk_error(db, version_id, "文档切块发生未预期错误")
        raise BusinessException(code=50052, message="文档切块失败，请查看服务日志") from exc
