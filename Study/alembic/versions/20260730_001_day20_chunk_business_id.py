"""Day20 add chunk business ID for MySQL and Milvus mapping.

Revision ID: 20260730_001
Revises: 20260729_001
Create Date: 2026-07-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260730_001"
down_revision: str | None = "20260729_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    existing_columns = {
        column["name"]
        for column in inspector.get_columns("knowledge_document_chunk")
    }
    if "chunk_id" not in existing_columns:
        # 先允许为空，才能兼容 Day19 已经生成的历史 chunk 记录。
        op.add_column(
            "knowledge_document_chunk",
            sa.Column(
                "chunk_id",
                sa.String(length=64),
                nullable=True,
                comment="检索块业务唯一 ID，同时作为 Milvus Entity 主键",
            ),
        )

    # 历史记录用稳定字段计算 64 位 ID；新记录由 Python uuid4().hex 生成 32 位 ID。
    op.execute(
        "UPDATE knowledge_document_chunk "
        "SET chunk_id = SHA2(CONCAT(document_id, ':', chunk_index, ':', id), 256) "
        "WHERE chunk_id IS NULL"
    )
    op.alter_column(
        "knowledge_document_chunk",
        "chunk_id",
        existing_type=sa.String(length=64),
        nullable=False,
        comment="检索块业务唯一 ID，同时作为 Milvus Entity 主键",
    )

    existing_indexes = {
        index["name"]
        for index in inspect(op.get_bind()).get_indexes("knowledge_document_chunk")
    }
    if "ix_knowledge_document_chunk_chunk_id" not in existing_indexes:
        op.create_index(
            "ix_knowledge_document_chunk_chunk_id",
            "knowledge_document_chunk",
            ["chunk_id"],
            unique=True,
        )


def downgrade() -> None:
    op.drop_index(
        "ix_knowledge_document_chunk_chunk_id",
        table_name="knowledge_document_chunk",
    )
    op.drop_column("knowledge_document_chunk", "chunk_id")
