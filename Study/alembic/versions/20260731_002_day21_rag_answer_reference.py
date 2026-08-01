"""Day21 add persisted RAG answer references.

Revision ID: 20260731_002
Revises: 20260731_001
Create Date: 2026-07-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260731_002"
down_revision: str | None = "20260731_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    table_name = "ai_rag_answer_reference"
    if table_name not in set(inspect(op.get_bind()).get_table_names()):
        op.create_table(
            table_name,
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
            sa.Column("reference_id", sa.String(length=64), nullable=False, comment="RAG 回答引用记录业务唯一 ID"),
            sa.Column("session_id", sa.String(length=64), nullable=False, comment="所属会话业务 ID"),
            sa.Column("assistant_message_id", sa.String(length=64), nullable=False, comment="产生该引用的 assistant 消息业务 ID"),
            sa.Column("source_id", sa.String(length=16), nullable=False, comment="模型回答中的资料编号，例如 S1"),
            sa.Column("document_id", sa.String(length=64), nullable=False, comment="引用知识库文档业务 ID"),
            sa.Column("version_id", sa.String(length=64), nullable=False, comment="引用时生效的知识库文档版本 ID"),
            sa.Column("chunk_id", sa.String(length=64), nullable=False, comment="引用的知识库检索块业务 ID"),
            sa.Column("chunk_index", sa.Integer(), nullable=False, comment="引用 chunk 在文档版本内的顺序"),
            sa.Column("score", sa.Float(), nullable=False, comment="本次检索时 Milvus 返回的相似度分数快照"),
            sa.Column("locations_json", sa.Text(), nullable=False, comment="引用来源位置快照 JSON，例如段落或页码"),
            sa.Column("created_at", sa.DateTime(), nullable=False, comment="引用记录创建时间"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("reference_id", name="uk_airar_ref_id"),
            sa.UniqueConstraint("assistant_message_id", "source_id", name="uk_airar_msg_source"),
            comment="RAG 回答实际引用来源审计表",
        )

    existing_columns = {
        tuple(index["column_names"])
        for index in inspect(op.get_bind()).get_indexes(table_name)
    }
    expected_indexes = {
        "ix_airar_session": ["session_id"],
        "ix_airar_assistant": ["assistant_message_id"],
        "ix_airar_doc_ver": ["document_id", "version_id"],
    }
    for index_name, columns in expected_indexes.items():
        if tuple(columns) not in existing_columns:
            op.create_index(index_name, table_name, columns)


def downgrade() -> None:
    op.drop_table("ai_rag_answer_reference")
