"""Day20 add document version activation audit.

Revision ID: 20260730_004
Revises: 20260730_003
Create Date: 2026-07-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260730_004"
down_revision: str | None = "20260730_003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_table() -> None:
    op.create_table(
        "knowledge_document_version_activation_audit",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
        sa.Column("activation_id", sa.String(length=64), nullable=False, comment="文档版本切换审计业务唯一 ID"),
        sa.Column("document_id", sa.String(length=64), nullable=False, comment="所属知识库文档业务 ID"),
        sa.Column("activated_version_id", sa.String(length=64), nullable=False, comment="本次切换为 active 的文档版本 ID"),
        sa.Column("previous_version_id", sa.String(length=64), nullable=True, comment="切换前 active 的文档版本 ID，首次发布时为空"),
        sa.Column("activated_by", sa.String(length=64), nullable=False, comment="执行切换的人员标识，接入认证后取自登录上下文"),
        sa.Column("activation_note", sa.Text(), nullable=False, comment="人工确认向量数量和检索质量后的切换说明"),
        sa.Column("activated_at", sa.DateTime(), nullable=False, comment="版本切换完成时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("activation_id"),
        comment="知识库文档索引版本切换审计表",
    )


def _create_missing_indexes() -> None:
    existing_indexes = {
        index["name"]
        for index in inspect(op.get_bind()).get_indexes("knowledge_document_version_activation_audit")
    }
    existing_index_columns = {
        tuple(index["column_names"])
        for index in inspect(op.get_bind()).get_indexes("knowledge_document_version_activation_audit")
    }
    expected_indexes = {
        # MySQL 索引名最大 64 字符，审计表名较长，必须使用稳定短名。
        "ix_kdvaa_doc": (["document_id"], False),
        "ix_kdvaa_active_ver": (["activated_version_id"], False),
        "ix_kdvaa_prev_ver": (["previous_version_id"], False),
    }
    for index_name, (columns, unique) in expected_indexes.items():
        # 兼容这次失败迁移遗留的 MySQL 自动截断索引名；字段等价即可，不能重复建索引。
        if index_name not in existing_indexes and tuple(columns) not in existing_index_columns:
            op.create_index(
                index_name,
                "knowledge_document_version_activation_audit",
                columns,
                unique=unique,
            )


def upgrade() -> None:
    table_names = set(inspect(op.get_bind()).get_table_names())
    if "knowledge_document_version_activation_audit" not in table_names:
        _create_table()
    _create_missing_indexes()


def downgrade() -> None:
    op.drop_table("knowledge_document_version_activation_audit")
