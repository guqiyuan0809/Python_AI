"""Day18 add parsed knowledge document segment table.

Revision ID: 20260728_002
Revises: 20260728_001
Create Date: 2026-07-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260728_002"
down_revision: str | None = "20260728_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_table() -> None:
    op.create_table(
        "knowledge_document_segment",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
        sa.Column("document_id", sa.String(length=64), nullable=False, comment="所属知识库文档业务 ID"),
        sa.Column("segment_index", sa.Integer(), nullable=False, comment="文本段在原文档中的从 0 开始顺序"),
        sa.Column("content", sa.Text(), nullable=False, comment="解析得到的原始文本内容，尚未经过 Day19 检索切块"),
        sa.Column("location", sa.String(length=255), nullable=False, comment="可追溯的原文位置，例如 Paragraph:12 或 Table:2/Row:4"),
        sa.Column("metadata_json", sa.Text(), nullable=False, comment="解析器输出的来源补充元数据 JSON"),
        sa.Column("created_at", sa.DateTime(), nullable=False, comment="文本段持久化时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "segment_index", name="uk_knowledge_document_segment_document_index"),
        comment="知识库文档解析后的原始文本段表",
    )


def _create_missing_indexes() -> None:
    existing_indexes = {index["name"] for index in inspect(op.get_bind()).get_indexes("knowledge_document_segment")}
    if "ix_knowledge_document_segment_document_id" not in existing_indexes:
        op.create_index(
            "ix_knowledge_document_segment_document_id",
            "knowledge_document_segment",
            ["document_id"],
        )


def upgrade() -> None:
    # 兼容开发阶段 FastAPI startup 可能先通过 create_all 建表的情况。
    table_names = set(inspect(op.get_bind()).get_table_names())
    if "knowledge_document_segment" not in table_names:
        _create_table()
    _create_missing_indexes()


def downgrade() -> None:
    op.drop_table("knowledge_document_segment")
