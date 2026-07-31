"""Day20 真实 Milvus 检索：Milvus 召回 ID，MySQL 回填原文与引用来源。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import KnowledgeDocument, KnowledgeDocumentChunk, KnowledgeDocumentVersion
from day04_app.schemas.knowledge_schema import ChunkSourceReference, MilvusChunkSearchItem
from day04_app.services.knowledge_embedding_service import generate_text_embeddings
from day04_app.services.milvus_vector_store_service import (
    EMBEDDING_DIMENSION,
    search_chunk_vectors,
)
from settings import settings


VERSION_STATUS_ACTIVE = "active"


def _load_source_references(source_references_json: str) -> list[ChunkSourceReference]:
    """将 MySQL 中的 JSON 来源信息恢复为已校验 DTO。"""
    try:
        raw_references = json.loads(source_references_json)
        return [ChunkSourceReference.model_validate(item) for item in raw_references]
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise BusinessException(code=50053, message="知识库 chunk 来源数据损坏") from exc


def _get_hit_field(hit: dict[str, Any], field_name: str) -> Any:
    """兼容 PyMilvus 不同版本的扁平和 entity 嵌套命中结果。"""
    entity = hit.get("entity")
    if isinstance(entity, dict) and field_name in entity:
        return entity[field_name]
    if field_name in hit:
        return hit[field_name]
    if field_name == "chunk_id":
        return hit.get("id")
    return None


def search_active_document_chunks(
    db: Session,
    *,
    document_id: str,
    question: str,
    top_k: int,
) -> tuple[str, int, str, list[MilvusChunkSearchItem]]:
    """仅检索文档当前 active 版本，避免候选或已 retired 版本意外进入线上回答。"""
    document = db.scalar(
        select(KnowledgeDocument).where(KnowledgeDocument.document_id == document_id)
    )
    if document is None:
        raise BusinessException(code=40451, message="知识库文档不存在")
    if not document.active_version_id:
        raise BusinessException(code=40960, message="文档尚未激活向量索引版本，不能执行真实检索")

    active_version = db.scalar(
        select(KnowledgeDocumentVersion).where(
            KnowledgeDocumentVersion.version_id == document.active_version_id
        )
    )
    if active_version is None or active_version.status != VERSION_STATUS_ACTIVE:
        raise BusinessException(code=40961, message="文档 active 版本状态异常，拒绝执行检索")
    if active_version.embedding_model != settings.dashscope_embedding_model:
        raise BusinessException(code=40962, message="当前查询 Embedding 模型与 active 索引版本不一致")
    if active_version.embedding_dimension != EMBEDDING_DIMENSION:
        raise BusinessException(code=40963, message="active 索引版本的向量维度与当前服务契约不一致")
    if active_version.vector_collection != settings.milvus_collection_name:
        raise BusinessException(code=40964, message="active 索引版本不属于当前 Milvus Collection")

    # 查询文本只生成一次向量；候选 chunk 的向量早已离线写入 Milvus。
    model, vectors = generate_text_embeddings([question])
    question_vector = vectors[0]
    if len(question_vector) != EMBEDDING_DIMENSION:
        raise BusinessException(code=50055, message="查询 Embedding 维度与 Milvus Collection 契约不一致")

    try:
        hits = search_chunk_vectors(
            version_id=active_version.version_id,
            question_vector=question_vector,
            top_k=top_k,
        )
    except Exception as exc:
        raise BusinessException(code=50056, message="Milvus 向量检索失败，请查看服务日志") from exc

    hit_ids = [str(_get_hit_field(hit, "chunk_id")) for hit in hits if _get_hit_field(hit, "chunk_id")]
    if not hit_ids:
        return model, len(question_vector), active_version.version_id, []

    chunks = list(
        db.scalars(
            select(KnowledgeDocumentChunk).where(
                KnowledgeDocumentChunk.chunk_id.in_(hit_ids),
                KnowledgeDocumentChunk.document_id == document.document_id,
                KnowledgeDocumentChunk.version_id == active_version.version_id,
            )
        )
    )
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    # 按 Milvus 分数顺序回填，不能使用MySQL  默认返回顺序，否则 Top-K 排名会丢失。
    items: list[MilvusChunkSearchItem] = []
    for hit in hits:
        chunk_id = _get_hit_field(hit, "chunk_id")
        chunk = chunk_by_id.get(str(chunk_id)) if chunk_id else None
        if chunk is None:
            # 向量库与 MySQL 出现短暂不一致时宁可跳过，不能向用户返回无来源的幻影命中。
            continue
        score = hit.get("distance", hit.get("score"))
        if score is None:
            raise BusinessException(code=50056, message="Milvus 检索结果缺少相似度分数")
        items.append(
            MilvusChunkSearchItem(
                document_id=chunk.document_id,
                version_id=chunk.version_id,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                score=max(-1.0, min(1.0, float(score))),
                content=chunk.content,
                source_references=_load_source_references(chunk.source_references_json),
            )
        )
    return model, len(question_vector), active_version.version_id, items
