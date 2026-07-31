"""Day19 add knowledge document chunks.

Revision ID: 20260729_001
Revises: 20260728_002
Create Date: 2026-07-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260729_001"
down_revision: str | None = "20260728_002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _add_missing_document_columns() -> None:
    existing_columns = {
        column["name"]
        for column in inspect(op.get_bind()).get_columns("knowledge_document")
    }
    columns = [
        sa.Column("chunk_status", sa.String(length=32), nullable=False, server_default="not_started", comment="切块生命周期状态：not_started/chunking/chunked/error"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0", comment="最近一次成功切块得到的检索块数量"),
        sa.Column("chunk_config_json", sa.Text(), nullable=True, comment="最近一次成功切块使用的参数快照 JSON"),
        sa.Column("chunk_error_message", sa.Text(), nullable=True, comment="切块失败时记录的错误原因"),
        sa.Column("chunked_at", sa.DateTime(), nullable=True, comment="最近一次成功完成切块的时间"),
    ]
    for column in columns:
        if column.name not in existing_columns:
            op.add_column("knowledge_document", column)


def _create_chunk_table() -> None:
    op.create_table(
        "knowledge_document_chunk",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
        sa.Column("document_id", sa.String(length=64), nullable=False, comment="所属知识库文档业务 ID"),
        sa.Column("chunk_index", sa.Integer(), nullable=False, comment="检索块在文档内的从 0 开始顺序"),
        sa.Column("content", sa.Text(), nullable=False, comment="将参与 Embedding 与语义检索的文本内容"),
        sa.Column("char_count", sa.Integer(), nullable=False, comment="切块文本字符数，用于控制上下文与成本"),
        sa.Column("source_references_json", sa.Text(), nullable=False, comment="切块覆盖的原始文档段来源 JSON"),
        sa.Column("created_at", sa.DateTime(), nullable=False, comment="检索块持久化时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uk_knowledge_document_chunk_document_index"),
        comment="知识库文档用于 Embedding 和检索的文本切块表",
    )


def _create_missing_indexes() -> None:
    existing_indexes = {index["name"] for index in inspect(op.get_bind()).get_indexes("knowledge_document")}
    if "ix_knowledge_document_chunk_status" not in existing_indexes:
        op.create_index("ix_knowledge_document_chunk_status", "knowledge_document", ["chunk_status"])

    chunk_indexes = {index["name"] for index in inspect(op.get_bind()).get_indexes("knowledge_document_chunk")}
    if "ix_knowledge_document_chunk_document_id" not in chunk_indexes:
        op.create_index("ix_knowledge_document_chunk_document_id", "knowledge_document_chunk", ["document_id"])


def upgrade() -> None:
    _add_missing_document_columns()
    table_names = set(inspect(op.get_bind()).get_table_names())
    if "knowledge_document_chunk" not in table_names:
        _create_chunk_table()
    _create_missing_indexes()


def downgrade() -> None:
    op.drop_table("knowledge_document_chunk")
    for column_name in ("chunked_at", "chunk_error_message", "chunk_config_json", "chunk_count", "chunk_status"):
        op.drop_column("knowledge_document", column_name)
