"""Day22 add parent-child chunks, contextual embeddings and reranker settings."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260803_002"
down_revision: str | None = "20260803_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table_columns(table_name: str) -> set[str]:
    return {column["name"] for column in inspect(op.get_bind()).get_columns(table_name)}


def _index_names(table_name: str) -> set[str]:
    return {index["name"] for index in inspect(op.get_bind()).get_indexes(table_name)}


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    tables = set(inspector.get_table_names())

    if "knowledge_document_parent_chunk" not in tables:
        op.create_table(
            "knowledge_document_parent_chunk",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="父块记录自增主键"),
            sa.Column("parent_chunk_id", sa.String(length=64), nullable=False, comment="父块业务唯一 ID，被子块引用"),
            sa.Column("version_id", sa.String(length=64), nullable=False, comment="所属文档候选版本业务 ID"),
            sa.Column("document_id", sa.String(length=64), nullable=False, comment="所属知识库文档业务 ID"),
            sa.Column("parent_index", sa.Integer(), nullable=False, comment="父块在文档版本内的顺序，从 0 开始"),
            sa.Column("content", sa.Text(), nullable=False, comment="由原始段落拼接出的完整父块原文"),
            sa.Column("char_count", sa.Integer(), nullable=False, comment="父块原文字符数"),
            sa.Column("source_references_json", sa.Text(), nullable=False, comment="父块覆盖的原始文档来源 JSON"),
            sa.Column("created_at", sa.DateTime(), nullable=False, comment="父块创建时间"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("parent_chunk_id", name="uk_kdpc_parent_id"),
            sa.UniqueConstraint("version_id", "parent_index", name="uk_kdpc_ver_idx"),
        )
        op.create_index(
            "ix_kdpc_doc_ver",
            "knowledge_document_parent_chunk",
            ["document_id", "version_id"],
        )
    else:
        indexes = _index_names("knowledge_document_parent_chunk")
        if "ix_kdpc_doc_ver" not in indexes:
            op.create_index(
                "ix_kdpc_doc_ver",
                "knowledge_document_parent_chunk",
                ["document_id", "version_id"],
            )

    chunk_columns = _table_columns("knowledge_document_chunk")
    if "parent_chunk_id" not in chunk_columns:
        op.add_column(
            "knowledge_document_chunk",
            sa.Column("parent_chunk_id", sa.String(length=64), nullable=True, comment="父子切块模式下的父块业务 ID"),
        )
    if "contextual_summary" not in chunk_columns:
        op.add_column(
            "knowledge_document_chunk",
            sa.Column("contextual_summary", sa.Text(), nullable=True, comment="模型生成的检索背景说明，仅用于召回"),
        )
    if "embedding_text" not in chunk_columns:
        op.add_column(
            "knowledge_document_chunk",
            sa.Column("embedding_text", sa.Text(), nullable=True, comment="背景说明与原文拼接后的向量化输入"),
        )
    chunk_indexes = _index_names("knowledge_document_chunk")
    if "ix_kdc_parent" not in chunk_indexes:
        op.create_index("ix_kdc_parent", "knowledge_document_chunk", ["parent_chunk_id"])

    run_columns = _table_columns("knowledge_retrieval_eval_run")
    if "use_reranker" not in run_columns:
        op.add_column(
            "knowledge_retrieval_eval_run",
            sa.Column("use_reranker", sa.Integer(), nullable=False, server_default="0", comment="是否启用 Reranker 精排，1 是 0 否"),
        )
    if "rerank_top_n" not in run_columns:
        op.add_column(
            "knowledge_retrieval_eval_run",
            sa.Column("rerank_top_n", sa.Integer(), nullable=True, comment="Reranker 使用的粗排候选数量"),
        )
    if "reranker_model" not in run_columns:
        op.add_column(
            "knowledge_retrieval_eval_run",
            sa.Column("reranker_model", sa.String(length=128), nullable=True, comment="本次评测使用的 Reranker 模型"),
        )


def downgrade() -> None:
    run_columns = _table_columns("knowledge_retrieval_eval_run")
    if "reranker_model" in run_columns:
        op.drop_column("knowledge_retrieval_eval_run", "reranker_model")
    if "rerank_top_n" in run_columns:
        op.drop_column("knowledge_retrieval_eval_run", "rerank_top_n")
    if "use_reranker" in run_columns:
        op.drop_column("knowledge_retrieval_eval_run", "use_reranker")

    chunk_columns = _table_columns("knowledge_document_chunk")
    if "embedding_text" in chunk_columns:
        op.drop_column("knowledge_document_chunk", "embedding_text")
    if "contextual_summary" in chunk_columns:
        op.drop_column("knowledge_document_chunk", "contextual_summary")
    if "parent_chunk_id" in chunk_columns:
        op.drop_index("ix_kdc_parent", table_name="knowledge_document_chunk")
        op.drop_column("knowledge_document_chunk", "parent_chunk_id")

    if "knowledge_document_parent_chunk" in set(inspect(op.get_bind()).get_table_names()):
        op.drop_index("ix_kdpc_doc_ver", table_name="knowledge_document_parent_chunk")
        op.drop_table("knowledge_document_parent_chunk")
