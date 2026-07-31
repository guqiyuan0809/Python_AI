"""将某个文档版本的 MySQL chunk 构建为 Milvus 向量索引。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import KnowledgeDocument, KnowledgeDocumentChunk, KnowledgeDocumentVersion
from day04_app.services.knowledge_embedding_service import generate_text_embeddings
from day04_app.services.milvus_vector_store_service import (
    EMBEDDING_DIMENSION,
    count_vectors_by_version,
    ensure_knowledge_chunk_collection,
    upsert_chunk_vectors,
)
from settings import settings


VERSION_STATUS_CHUNKED = "chunked"
VERSION_STATUS_INDEXING = "indexing"
VERSION_STATUS_INDEXED = "indexed"
VERSION_STATUS_ACTIVE = "active"
VERSION_STATUS_ERROR = "error"
VECTOR_UPSERT_BATCH_SIZE = 10


def _find_version(db: Session, version_id: str) -> KnowledgeDocumentVersion:
    version = db.scalar(
        select(KnowledgeDocumentVersion).where(KnowledgeDocumentVersion.version_id == version_id)
    )
    if version is None:
        raise BusinessException(code=40452, message="知识库文档版本不存在")
    return version


def _mark_index_error(db: Session, version_id: str, error_message: str) -> None:
    """Milvus 或 Embedding 失败时独立提交错误状态，保留重试和排查依据。"""
    version = _find_version(db, version_id)
    version.status = VERSION_STATUS_ERROR
    version.error_message = error_message[:2000]
    db.commit()


def build_version_vector_index(
    db: Session,
    version_id: str,
) -> KnowledgeDocumentVersion:
    """批量生成指定版本所有 chunk 的 Embedding 并写入 Milvus，不切换 active 指针。"""
    version = _find_version(db, version_id)
    if version.status not in {VERSION_STATUS_CHUNKED, VERSION_STATUS_INDEXED, VERSION_STATUS_ERROR}:
        raise BusinessException(
            code=40956,
            message=f"当前版本状态为 {version.status}，不能构建向量索引",
        )
    if version.status == VERSION_STATUS_INDEXING:
        raise BusinessException(code=40957, message="版本正在构建向量索引，请勿重复提交")

    chunks = list(
        db.scalars(
            select(KnowledgeDocumentChunk)
            .where(KnowledgeDocumentChunk.version_id == version.version_id)
            .order_by(KnowledgeDocumentChunk.chunk_index)
        )
    )
    if not chunks:
        raise BusinessException(code=40070, message="当前版本没有 chunk，不能生成向量索引")

    version.status = VERSION_STATUS_INDEXING
    version.error_message = None
    db.commit()

    try:
        # 先校验 Collection 契约，避免调完模型才发现维度或字段错误。
        ensure_knowledge_chunk_collection()
        model, vectors = generate_text_embeddings(
            [chunk.content for chunk in chunks],
            batch_size=VECTOR_UPSERT_BATCH_SIZE,
        )
        if len(vectors) != len(chunks):
            raise ValueError("Embedding 向量数量与 chunk 数量不一致")
        if any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
            raise ValueError(f"Embedding 向量维度不是 {EMBEDDING_DIMENSION}")

        records = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "version_id": chunk.version_id,
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        upsert_chunk_vectors(records=records)
        indexed_vector_count = count_vectors_by_version(version.version_id)
        if indexed_vector_count != len(chunks):
            raise ValueError(
                f"Milvus 向量数量校验失败：期望 {len(chunks)}，实际 {indexed_vector_count}"
            )

        version.status = VERSION_STATUS_INDEXED
        version.embedding_model = model
        version.embedding_dimension = EMBEDDING_DIMENSION
        version.vector_collection = settings.milvus_collection_name
        version.vector_count = indexed_vector_count
        version.error_message = None
        db.commit()
        db.refresh(version)
        return version
    except BusinessException as exc:
        db.rollback()
        _mark_index_error(db, version_id, exc.message)
        raise
    except Exception as exc:
        db.rollback()
        _mark_index_error(db, version_id, "文档版本向量索引构建失败")
        raise BusinessException(code=50054, message="文档版本向量索引构建失败，请查看服务日志") from exc
