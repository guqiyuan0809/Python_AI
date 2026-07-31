"""Day19 教学用内存向量检索；Day20 会由 Qdrant 替代全量扫描。"""

from __future__ import annotations

import json

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import KnowledgeDocument, KnowledgeDocumentChunk
from day04_app.schemas.knowledge_schema import (
    ChunkSourceReference,
    InMemoryChunkSearchItem,
)
from day04_app.services.knowledge_embedding_service import (
    calculate_cosine_similarity,
    generate_text_embeddings,
)


MAX_IN_MEMORY_CANDIDATES = 50


def _load_source_references(source_references_json: str) -> list[ChunkSourceReference]:
    """将 MySQL 中的来源 JSON 恢复为 DTO，防止接口直接泄露未校验的字符串。"""
    try:
        raw_references = json.loads(source_references_json)
        return [ChunkSourceReference.model_validate(item) for item in raw_references]
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise BusinessException(code=50053, message="知识库 chunk 来源数据损坏") from exc


def search_document_chunks_in_memory(
    db: Session,
    document_id: str,
    question: str,
    top_k: int,
) -> tuple[str, int, int, list[InMemoryChunkSearchItem]]:
    """为问题和当前文档所有候选块临时生成向量，逐个算分后返回 Top-K。"""
    document = db.scalar(
        select(KnowledgeDocument).where(KnowledgeDocument.document_id == document_id)
    )
    if document is None:
        raise BusinessException(code=40451, message="知识库文档不存在")
    if document.chunk_status != "chunked":
        raise BusinessException(code=40954, message="文档尚未完成切块，不能执行检索演示")

    chunk_records = list(
        db.scalars(
            select(KnowledgeDocumentChunk)
            .where(KnowledgeDocumentChunk.document_id == document_id)
            .order_by(KnowledgeDocumentChunk.chunk_index)
        )
    )
    if not chunk_records:
        raise BusinessException(code=40068, message="文档没有可检索的 chunk")
    if len(chunk_records) > MAX_IN_MEMORY_CANDIDATES:
        raise BusinessException(
            code=40069,
            message=f"Day19 内存检索最多支持 {MAX_IN_MEMORY_CANDIDATES} 个 chunk，请在 Day20 使用向量数据库",
        )

    # 第 0 条是用户问题，后续向量按 chunk_records 的数据库顺序一一对应。
    model, vectors = generate_text_embeddings(
        [question, *[chunk.content for chunk in chunk_records]]
    )
    question_vector = vectors[0]
    ranked_items = [
        InMemoryChunkSearchItem(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            score=calculate_cosine_similarity(question_vector, chunk_vector),
            content=chunk.content,
            source_references=_load_source_references(chunk.source_references_json),
        )
        for chunk, chunk_vector in zip(chunk_records, vectors[1:], strict=True)
    ]
    ranked_items.sort(key=lambda item: item.score, reverse=True)
    return model, len(question_vector), len(chunk_records), ranked_items[:top_k]
