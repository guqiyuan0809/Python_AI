"""Day20 真实 Milvus 检索：Milvus 召回 ID，MySQL 回填原文与引用来源。"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import (
    KnowledgeDocument,
    KnowledgeDocumentChunk,
    KnowledgeDocumentParentChunk,
    KnowledgeDocumentVersion,
)
from day04_app.schemas.knowledge_schema import ChunkSourceReference, MilvusChunkSearchItem
from day04_app.services.call_log_service import create_call_log
from day04_app.services.knowledge_embedding_service import generate_text_embeddings
from day04_app.services.milvus_vector_store_service import (
    EMBEDDING_DIMENSION,
    search_chunk_vectors,
    search_chunk_vectors_for_versions,
)
from day04_app.services.knowledge_reranker_service import rerank_chunks
from settings import settings


VERSION_STATUS_ACTIVE = "active"
VERSION_STATUS_INDEXED = "indexed"
_QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}")


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


def _build_rerank_document(item: MilvusChunkSearchItem, chunk: KnowledgeDocumentChunk) -> str:
    """Reranker 可以读取检索背景，但回答和引用仍只能使用真实原文 content。"""
    if chunk.contextual_summary:
        return (
            f"检索背景：{chunk.contextual_summary.strip()}\n\n"
            f"原文子块：{item.content.strip()}"
        )
    return item.content


def _keyword_coverage_score(question: str, document: str) -> float:
    """用很轻的词面覆盖度补偿向量/Reranker 对短定义块的低估。"""
    tokens: set[str] = set()
    for token in _QUERY_TOKEN_PATTERN.findall(question):
        normalized = token.lower().strip()
        if len(normalized) < 2:
            continue
        tokens.add(normalized)
        if re.fullmatch(r"[\u4e00-\u9fff]+", normalized):
            tokens.update(normalized[index : index + 2] for index in range(len(normalized) - 1))
    if not tokens:
        return 0.0
    normalized_document = document.lower()
    matched = sum(1 for token in tokens if token in normalized_document)
    return matched / len(tokens)


def _intent_match_score(question: str, document: str) -> float:
    """补充定义类/作用类问题的结构信号，避免短定义块被泛化总览块压低。"""
    normalized_question = question.lower()
    normalized_document = document.lower()
    score = 0.0
    if any(keyword in normalized_question for keyword in ("作用", "是什么", "什么是", "概念", "定义")):
        if any(marker in normalized_document for marker in ("作用：", "核心作用", "概念：", "定义：")):
            score += 0.7
        if "存储对象" in normalized_document or ("对象" in normalized_document and "成员变量" in normalized_document):
            score += 0.3
    return min(score, 1.0)


def _search_document_version_chunks(
    db: Session,
    *,
    document_id: str,
    version: KnowledgeDocumentVersion,
    question: str,
    top_k: int,
    use_reranker: bool = False,
    rerank_top_n: int | None = None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
) -> tuple[str, int, str, list[MilvusChunkSearchItem]]:
    """按已确认的单一版本检索并回填原文，避免两个版本的结果混在同一 Top-K。"""
    if version.embedding_model != settings.dashscope_embedding_model:
        raise BusinessException(code=40962, message="当前查询 Embedding 模型与 active 索引版本不一致")
    if version.embedding_dimension != EMBEDDING_DIMENSION:
        raise BusinessException(code=40963, message="active 索引版本的向量维度与当前服务契约不一致")
    if version.vector_collection != settings.milvus_collection_name:
        raise BusinessException(code=40964, message="active 索引版本不属于当前 Milvus Collection")

    # 查询文本只生成一次向量；候选 chunk 的向量早已离线写入 Milvus。
    embedding_start_time = time.perf_counter()
    model, vectors = generate_text_embeddings([question])
    question_vector = vectors[0]
    if len(question_vector) != EMBEDDING_DIMENSION:
        raise BusinessException(code=50055, message="查询 Embedding 维度与 Milvus Collection 契约不一致")
    create_call_log(
        db,
        call_type="session_rag",
        stage="rag_query_embedding",
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
        model=model,
        cost_ms=round((time.perf_counter() - embedding_start_time) * 1000),
        detail={"input_count": 1, "embedding_dimension": len(question_vector)},
        commit=False,
    )

    try:
        vector_search_start_time = time.perf_counter()
        hits = search_chunk_vectors(
            version_id=version.version_id,
            question_vector=question_vector,
            top_k=rerank_top_n if use_reranker and rerank_top_n else top_k,
        )
    except Exception as exc:
        raise BusinessException(code=50056, message="Milvus 向量检索失败，请查看服务日志") from exc
    create_call_log(
        db,
        call_type="session_rag",
        stage="rag_vector_search",
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
        cost_ms=round((time.perf_counter() - vector_search_start_time) * 1000),
        detail={
            "version_id": version.version_id,
            "requested_top_k": rerank_top_n if use_reranker and rerank_top_n else top_k,
            "hit_count": len(hits),
        },
        commit=False,
    )

    hit_ids = [str(_get_hit_field(hit, "chunk_id")) for hit in hits if _get_hit_field(hit, "chunk_id")]
    if not hit_ids:
        return model, len(question_vector), version.version_id, []

    chunks = list(
        db.scalars(
            select(KnowledgeDocumentChunk).where(
                KnowledgeDocumentChunk.chunk_id.in_(hit_ids),
                KnowledgeDocumentChunk.document_id == document_id,
                KnowledgeDocumentChunk.version_id == version.version_id,
            )
        )
    )
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    parent_by_id: dict[str, KnowledgeDocumentParentChunk] = {}
    parent_ids = {chunk.parent_chunk_id for chunk in chunks if chunk.parent_chunk_id}
    if parent_ids:
        parents = list(
            db.scalars(
                select(KnowledgeDocumentParentChunk).where(
                    KnowledgeDocumentParentChunk.parent_chunk_id.in_(parent_ids),
                    KnowledgeDocumentParentChunk.version_id == version.version_id,
                )
            )
        )
        parent_by_id = {parent.parent_chunk_id: parent for parent in parents}

    # 按 Milvus 分数顺序回填，不能使用 MySQL 默认返回顺序，否则 Top-K 排名会丢失。
    items: list[MilvusChunkSearchItem] = []
    rerank_documents: list[str] = []
    for hit in hits:
        chunk_id = _get_hit_field(hit, "chunk_id")
        chunk = chunk_by_id.get(str(chunk_id)) if chunk_id else None
        if chunk is None:
            # 向量库与 MySQL 出现短暂不一致时宁可跳过，不能向用户返回无来源的幻影命中。
            continue
        score = hit.get("distance", hit.get("score"))
        if score is None:
            raise BusinessException(code=50056, message="Milvus 检索结果缺少相似度分数")
        parent = parent_by_id.get(chunk.parent_chunk_id) if chunk.parent_chunk_id else None
        item = MilvusChunkSearchItem(
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            parent_chunk_id=chunk.parent_chunk_id,
            score=max(-1.0, min(1.0, float(score))),
            vector_score=max(-1.0, min(1.0, float(score))),
            content=chunk.content,
            source_references=_load_source_references(chunk.source_references_json),
            parent_content=parent.content if parent else None,
            parent_source_references=(
                _load_source_references(parent.source_references_json) if parent else []
            ),
        )
        items.append(item)
        rerank_documents.append(_build_rerank_document(item, chunk))
    if use_reranker and items:
        rerank_start_time = time.perf_counter()
        reranked = rerank_chunks(question, rerank_documents, len(rerank_documents))
        create_call_log(
            db,
            call_type="session_rag",
            stage="rag_rerank",
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=message_id,
            model=settings.dashscope_rerank_model,
            cost_ms=round((time.perf_counter() - rerank_start_time) * 1000),
            detail={"candidate_count": len(rerank_documents), "result_count": len(reranked)},
            commit=False,
        )
        reranked_items = [
            (
                result,
                items[result.index],
                rerank_documents[result.index],
            )
            for result in reranked
        ]
        ranked_items = sorted(
            reranked_items,
            key=lambda item: (
                item[0].relevance_score * 0.65
                + (item[1].vector_score or 0.0) * 0.15
                + _keyword_coverage_score(question, item[2]) * 0.05
                + _intent_match_score(question, item[2]) * 0.15
            ),
            reverse=True,
        )
        items = [
            item.model_copy(
                update={
                    "score": result.relevance_score,
                    "rerank_score": result.relevance_score,
                }
            )
            for result, item, _ in ranked_items[:top_k]
        ]
    else:
        items = items[:top_k]
    return model, len(question_vector), version.version_id, items


def search_active_document_chunks(
    db: Session,
    *,
    document_id: str,
    question: str,
    top_k: int,
    use_reranker: bool = False,
    rerank_top_n: int | None = None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
) -> tuple[str, int, str, list[MilvusChunkSearchItem]]:
    """线上入口只检索当前 active 版本，不能由普通 RAG 请求指定候选版本。"""
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
    return _search_document_version_chunks(
        db,
        document_id=document_id,
        version=active_version,
        question=question,
        top_k=top_k,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
    )


def search_active_documents_chunks(
    db: Session,
    *,
    document_ids: list[str],
    question: str,
    top_k: int,
    use_reranker: bool = False,
    rerank_top_n: int | None = None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
) -> tuple[str, int, dict[str, str], list[MilvusChunkSearchItem]]:
    """在多个文档的 active 版本中进行一次全局 Top-K 检索并回填可审计来源。

    这是多文档知识库的底层业务检索，而不是对单文档函数做 N 次循环：查询 Embedding
    只生成一次，Milvus 对所有准入版本执行全局排序，Reranker 也只对全局候选执行一次。
    参数中的 document_ids 应来自路由/权限层允许访问的知识域；本函数仍会逐个确认
    ``active_version_id`` 与 Collection 契约，拒绝候选或失效版本进入线上回答。
    """

    normalized_document_ids = list(dict.fromkeys(item.strip() for item in document_ids if item.strip()))
    if not normalized_document_ids:
        raise BusinessException(code=40074, message="多文档检索至少需要一个 document_id")

    documents = list(
        db.scalars(
            select(KnowledgeDocument).where(KnowledgeDocument.document_id.in_(normalized_document_ids))
        )
    )
    document_by_id = {document.document_id: document for document in documents}
    missing_document_ids = [document_id for document_id in normalized_document_ids if document_id not in document_by_id]
    if missing_document_ids:
        raise BusinessException(code=40451, message="知识库文档不存在")

    active_version_ids: list[str] = []
    active_version_by_document_id: dict[str, str] = {}
    for document_id in normalized_document_ids:
        active_version_id = document_by_id[document_id].active_version_id
        if not active_version_id:
            raise BusinessException(code=40960, message="存在尚未激活向量索引版本的知识库文档")
        active_version_ids.append(active_version_id)
        active_version_by_document_id[document_id] = active_version_id

    versions = list(
        db.scalars(
            select(KnowledgeDocumentVersion).where(
                KnowledgeDocumentVersion.version_id.in_(active_version_ids)
            )
        )
    )
    version_by_id = {version.version_id: version for version in versions}
    for document_id, active_version_id in active_version_by_document_id.items():
        version = version_by_id.get(active_version_id)
        if version is None or version.document_id != document_id or version.status != VERSION_STATUS_ACTIVE:
            raise BusinessException(code=40961, message="知识库文档 active 版本状态异常，拒绝执行检索")
        if version.embedding_model != settings.dashscope_embedding_model:
            raise BusinessException(code=40962, message="当前查询 Embedding 模型与 active 索引版本不一致")
        if version.embedding_dimension != EMBEDDING_DIMENSION:
            raise BusinessException(code=40963, message="active 索引版本的向量维度与当前服务契约不一致")
        if version.vector_collection != settings.milvus_collection_name:
            raise BusinessException(code=40964, message="active 索引版本不属于当前 Milvus Collection")

    embedding_start_time = time.perf_counter()
    model, vectors = generate_text_embeddings([question])
    question_vector = vectors[0]
    if len(question_vector) != EMBEDDING_DIMENSION:
        raise BusinessException(code=50055, message="查询 Embedding 维度与 Milvus Collection 契约不一致")
    create_call_log(
        db,
        call_type="multi_document_rag",
        stage="rag_query_embedding",
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
        model=model,
        cost_ms=round((time.perf_counter() - embedding_start_time) * 1000),
        detail={
            "input_count": 1,
            "embedding_dimension": len(question_vector),
            "document_count": len(normalized_document_ids),
        },
        commit=False,
    )

    requested_top_k = rerank_top_n if use_reranker and rerank_top_n else top_k
    try:
        vector_search_start_time = time.perf_counter()
        hits = search_chunk_vectors_for_versions(
            version_ids=active_version_ids,
            question_vector=question_vector,
            top_k=requested_top_k,
        )
    except Exception as exc:
        raise BusinessException(code=50056, message="Milvus 多文档向量检索失败，请查看服务日志") from exc
    create_call_log(
        db,
        call_type="multi_document_rag",
        stage="rag_vector_search",
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
        cost_ms=round((time.perf_counter() - vector_search_start_time) * 1000),
        detail={
            "version_ids": active_version_ids,
            "requested_top_k": requested_top_k,
            "hit_count": len(hits),
            "global_ranking": True,
        },
        commit=False,
    )
    hit_ids = [str(_get_hit_field(hit, "chunk_id")) for hit in hits if _get_hit_field(hit, "chunk_id")]
    if not hit_ids:
        return model, len(question_vector), active_version_by_document_id, []

    chunks = list(
        db.scalars(
            select(KnowledgeDocumentChunk).where(
                KnowledgeDocumentChunk.chunk_id.in_(hit_ids),
                KnowledgeDocumentChunk.document_id.in_(normalized_document_ids),
                KnowledgeDocumentChunk.version_id.in_(active_version_ids),
            )
        )
    )
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    parent_ids = {chunk.parent_chunk_id for chunk in chunks if chunk.parent_chunk_id}
    parents = (
        list(
            db.scalars(
                select(KnowledgeDocumentParentChunk).where(
                    KnowledgeDocumentParentChunk.parent_chunk_id.in_(parent_ids),
                    KnowledgeDocumentParentChunk.version_id.in_(active_version_ids),
                )
            )
        )
        if parent_ids
        else []
    )
    parent_by_id = {parent.parent_chunk_id: parent for parent in parents}

    items: list[MilvusChunkSearchItem] = []
    rerank_documents: list[str] = []
    for hit in hits:
        chunk_id = _get_hit_field(hit, "chunk_id")
        chunk = chunk_by_id.get(str(chunk_id)) if chunk_id else None
        if chunk is None:
            # MySQL/Milvus 短暂不一致时不能返回没有来源或跨版本的命中。
            continue
        score = hit.get("distance", hit.get("score"))
        if score is None:
            raise BusinessException(code=50056, message="Milvus 检索结果缺少相似度分数")
        parent = parent_by_id.get(chunk.parent_chunk_id) if chunk.parent_chunk_id else None
        item = MilvusChunkSearchItem(
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            parent_chunk_id=chunk.parent_chunk_id,
            score=max(-1.0, min(1.0, float(score))),
            vector_score=max(-1.0, min(1.0, float(score))),
            content=chunk.content,
            source_references=_load_source_references(chunk.source_references_json),
            parent_content=parent.content if parent else None,
            parent_source_references=(
                _load_source_references(parent.source_references_json) if parent else []
            ),
        )
        items.append(item)
        rerank_documents.append(_build_rerank_document(item, chunk))

    if use_reranker and items:
        rerank_start_time = time.perf_counter()
        reranked = rerank_chunks(question, rerank_documents, len(rerank_documents))
        create_call_log(
            db,
            call_type="multi_document_rag",
            stage="rag_rerank",
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=message_id,
            model=settings.dashscope_rerank_model,
            cost_ms=round((time.perf_counter() - rerank_start_time) * 1000),
            detail={"candidate_count": len(rerank_documents), "result_count": len(reranked), "global_ranking": True},
            commit=False,
        )
        ranked_items = sorted(
            [(result, items[result.index], rerank_documents[result.index]) for result in reranked],
            key=lambda candidate: (
                candidate[0].relevance_score * 0.65
                + (candidate[1].vector_score or 0.0) * 0.15
                + _keyword_coverage_score(question, candidate[2]) * 0.05
                + _intent_match_score(question, candidate[2]) * 0.15
            ),
            reverse=True,
        )
        items = [
            item.model_copy(update={"score": result.relevance_score, "rerank_score": result.relevance_score})
            for result, item, _ in ranked_items[:top_k]
        ]
    else:
        items = items[:top_k]
    return model, len(question_vector), active_version_by_document_id, items


def search_document_version_chunks_for_validation(
    db: Session,
    *,
    document_id: str,
    version_id: str,
    question: str,
    top_k: int,
    use_reranker: bool = False,
    rerank_top_n: int | None = None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
) -> tuple[str, int, str, list[MilvusChunkSearchItem]]:
    """仅供发布前验证：允许检索 indexed/active 版本，但绝不改动线上 active 指针。"""
    version = db.scalar(
        select(KnowledgeDocumentVersion).where(
            KnowledgeDocumentVersion.document_id == document_id,
            KnowledgeDocumentVersion.version_id == version_id,
        )
    )
    if version is None:
        raise BusinessException(code=40452, message="知识库文档版本不存在或不属于当前文档")
    if version.status not in {VERSION_STATUS_INDEXED, VERSION_STATUS_ACTIVE}:
        raise BusinessException(code=40968, message="只有 indexed 或 active 版本可以执行验证检索")
    return _search_document_version_chunks(
        db,
        document_id=document_id,
        version=version,
        question=question,
        top_k=top_k,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
    )
