"""Day18 add Prompt rollback audit table.

Revision ID: 20260727_003
Revises: 20260727_002
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260727_003"
down_revision: str | None = "20260727_002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_rollback_audit_table() -> None:
    op.create_table(
        "ai_prompt_rollback_audit",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
        sa.Column("rollback_id", sa.String(length=64), nullable=False, comment="Prompt 回滚业务唯一 ID"),
        sa.Column("publish_id", sa.String(length=64), nullable=False, comment="被回滚的原发布审计 ID；一条发布记录仅允许回滚一次"),
        sa.Column("prompt_name", sa.String(length=64), nullable=False, comment="业务 Prompt 名称，例如 work_order_analysis"),
        sa.Column("rolled_back_prompt_version", sa.String(length=32), nullable=False, comment="被下线的当前 active Prompt 版本"),
        sa.Column("restored_prompt_version", sa.String(length=32), nullable=False, comment="被恢复为 active 的历史 Prompt 版本"),
        sa.Column("rollback_reason", sa.Text(), nullable=False, comment="人工回滚原因，例如线上质量或延迟异常"),
        sa.Column("rolled_back_by", sa.String(length=64), nullable=False, comment="执行回滚的人员标识；接入认证后应取自登录用户"),
        sa.Column("rolled_back_at", sa.DateTime(), nullable=False, comment="实际完成版本状态切换的时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("rollback_id"),
        sa.UniqueConstraint("publish_id"),
        comment="Prompt 人工回滚审计记录表",
    )
    op.create_index("ix_ai_prompt_rollback_audit_rollback_id", "ai_prompt_rollback_audit", ["rollback_id"], unique=True)
    op.create_index("ix_ai_prompt_rollback_audit_publish_id", "ai_prompt_rollback_audit", ["publish_id"], unique=True)
    op.create_index("ix_ai_prompt_rollback_audit_prompt_name", "ai_prompt_rollback_audit", ["prompt_name"])


def _create_missing_indexes() -> None:
    expected_indexes = {
        "ix_ai_prompt_rollback_audit_rollback_id": (["rollback_id"], True),
        "ix_ai_prompt_rollback_audit_publish_id": (["publish_id"], True),
        "ix_ai_prompt_rollback_audit_prompt_name": (["prompt_name"], False),
    }
    existing_indexes = {index["name"] for index in inspect(op.get_bind()).get_indexes("ai_prompt_rollback_audit")}
    for index_name, (columns, unique) in expected_indexes.items():
        if index_name not in existing_indexes:
            op.create_index(index_name, "ai_prompt_rollback_audit", columns, unique=unique)


def _apply_comments() -> None:
    """兼容 create_all 先建表的本地环境，并把乱码备注恢复为 UTF-8 中文备注。"""
    op.alter_column(
        "ai_prompt_rollback_audit",
        "id",
        existing_type=sa.Integer(),
        existing_nullable=False,
        existing_autoincrement=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    column_comments = {
        "rollback_id": (sa.String(length=64), False, "Prompt 回滚业务唯一 ID"),
        "publish_id": (sa.String(length=64), False, "被回滚的原发布审计 ID；一条发布记录仅允许回滚一次"),
        "prompt_name": (sa.String(length=64), False, "业务 Prompt 名称，例如 work_order_analysis"),
        "rolled_back_prompt_version": (sa.String(length=32), False, "被下线的当前 active Prompt 版本"),
        "restored_prompt_version": (sa.String(length=32), False, "被恢复为 active 的历史 Prompt 版本"),
        "rollback_reason": (sa.Text(), False, "人工回滚原因，例如线上质量或延迟异常"),
        "rolled_back_by": (sa.String(length=64), False, "执行回滚的人员标识；接入认证后应取自登录用户"),
        "rolled_back_at": (sa.DateTime(), False, "实际完成版本状态切换的时间"),
    }
    for column_name, (column_type, nullable, comment) in column_comments.items():
        op.alter_column(
            "ai_prompt_rollback_audit",
            column_name,
            existing_type=column_type,
            existing_nullable=nullable,
            comment=comment,
        )
    op.execute("ALTER TABLE ai_prompt_rollback_audit COMMENT = 'Prompt 人工回滚审计记录表'")


def upgrade() -> None:
    # 本地 FastAPI startup 可能已用 create_all 建表；迁移仍负责补齐索引、备注和版本登记。
    table_names = set(inspect(op.get_bind()).get_table_names())
    if "ai_prompt_rollback_audit" not in table_names:
        _create_rollback_audit_table()
    _create_missing_indexes()
    _apply_comments()


def downgrade() -> None:
    op.drop_index("ix_ai_prompt_rollback_audit_prompt_name", table_name="ai_prompt_rollback_audit")
    op.drop_index("ix_ai_prompt_rollback_audit_publish_id", table_name="ai_prompt_rollback_audit")
    op.drop_index("ix_ai_prompt_rollback_audit_rollback_id", table_name="ai_prompt_rollback_audit")
    op.drop_table("ai_prompt_rollback_audit")
