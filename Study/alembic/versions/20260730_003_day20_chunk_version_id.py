"""Day20 link chunks to document index versions.

Revision ID: 20260730_003
Revises: 20260730_002
Create Date: 2026-07-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260730_003"
down_revision: str | None = "20260730_002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    existing_columns = {
        column["name"]
        for column in inspect(op.get_bind()).get_columns("knowledge_document_chunk")
    }
    if "version_id" not in existing_columns:
        # 先允许为空，回填旧 v1 后再收紧为 NOT NULL。
        op.add_column(
            "knowledge_document_chunk",
            sa.Column(
                "version_id",
                sa.String(length=64),
                nullable=True,
                comment="所属文档索引版本业务 ID，用于新旧版本并存和向量检索过滤",
            ),
        )
    op.execute(
        """
        UPDATE knowledge_document_chunk c
        JOIN knowledge_document_version v
          ON v.document_id = c.document_id AND v.version_number = 1
        SET c.version_id = v.version_id
        WHERE c.version_id IS NULL
        """
    )
    op.alter_column(
        "knowledge_document_chunk",
        "version_id",
        existing_type=sa.String(length=64),
        nullable=False,
        comment="所属文档索引版本业务 ID，用于新旧版本并存和向量检索过滤",
    )

    existing_indexes = {
        index["name"]
        for index in inspect(op.get_bind()).get_indexes("knowledge_document_chunk")
    }
    if "ix_knowledge_document_chunk_version_id" not in existing_indexes:
        op.create_index(
            "ix_knowledge_document_chunk_version_id",
            "knowledge_document_chunk",
            ["version_id"],
        )


def downgrade() -> None:
    op.drop_index(
        "ix_knowledge_document_chunk_version_id",
        table_name="knowledge_document_chunk",
    )
    op.drop_column("knowledge_document_chunk", "version_id")
