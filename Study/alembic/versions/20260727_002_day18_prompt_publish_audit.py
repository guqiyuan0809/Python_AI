"""Day18 add Prompt publish audit table.

Revision ID: 20260727_002
Revises: 20260727_001
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260727_002"
down_revision: str | None = "20260727_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_publish_audit_table() -> None:
    op.create_table(
        "ai_prompt_publish_audit",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
        sa.Column("publish_id", sa.String(length=64), nullable=False, comment="Prompt 发布业务唯一 ID"),
        sa.Column("gate_id", sa.String(length=64), nullable=False, comment="本次发布依据的评测门禁 ID"),
        sa.Column("prompt_id", sa.String(length=64), nullable=False, comment="被发布的候选 Prompt ID"),
        sa.Column("prompt_name", sa.String(length=64), nullable=False, comment="业务 Prompt 名称，例如 work_order_analysis"),
        sa.Column("candidate_prompt_version", sa.String(length=32), nullable=False, comment="本次发布的候选 Prompt 版本"),
        sa.Column("previous_prompt_version", sa.String(length=32), nullable=True, comment="发布前线上 active Prompt 版本；首次发布时为空"),
        sa.Column("gate_decision", sa.String(length=32), nullable=False, comment="发布时 Gate 结论：pass 或 manual_review"),
        sa.Column("approval_note", sa.Text(), nullable=False, comment="人工批准说明，记录性能或业务权衡依据"),
        sa.Column("approved_by", sa.String(length=64), nullable=False, comment="批准人标识；接入认证后应取自登录用户"),
        sa.Column("published_at", sa.DateTime(), nullable=False, comment="实际完成状态切换的时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("publish_id"),
        comment="Prompt 人工发布审计记录表",
    )
    op.create_index("ix_ai_prompt_publish_audit_publish_id", "ai_prompt_publish_audit", ["publish_id"], unique=True)
    op.create_index("ix_ai_prompt_publish_audit_gate_id", "ai_prompt_publish_audit", ["gate_id"])
    op.create_index("ix_ai_prompt_publish_audit_prompt_id", "ai_prompt_publish_audit", ["prompt_id"])
    op.create_index("ix_ai_prompt_publish_audit_prompt_name", "ai_prompt_publish_audit", ["prompt_name"])


def _apply_comments() -> None:
    """兼容 create_all 先建表的本地学习环境，并统一修正 MySQL 字段备注。"""
    # MySQL 修改字段备注时需要保留 AUTO_INCREMENT，否则自增主键属性会丢失。
    op.alter_column(
        "ai_prompt_publish_audit",
        "id",
        existing_type=sa.Integer(),
        existing_nullable=False,
        existing_autoincrement=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    column_comments = {
        "publish_id": (sa.String(length=64), False, "Prompt 发布业务唯一 ID"),
        "gate_id": (sa.String(length=64), False, "本次发布依据的评测门禁 ID"),
        "prompt_id": (sa.String(length=64), False, "被发布的候选 Prompt ID"),
        "prompt_name": (sa.String(length=64), False, "业务 Prompt 名称，例如 work_order_analysis"),
        "candidate_prompt_version": (sa.String(length=32), False, "本次发布的候选 Prompt 版本"),
        "previous_prompt_version": (sa.String(length=32), True, "发布前线上 active Prompt 版本；首次发布时为空"),
        "gate_decision": (sa.String(length=32), False, "发布时 Gate 结论：pass 或 manual_review"),
        "approval_note": (sa.Text(), False, "人工批准说明，记录性能或业务权衡依据"),
        "approved_by": (sa.String(length=64), False, "批准人标识；接入认证后应取自登录用户"),
        "published_at": (sa.DateTime(), False, "实际完成状态切换的时间"),
    }
    for column_name, (column_type, nullable, comment) in column_comments.items():
        op.alter_column(
            "ai_prompt_publish_audit",
            column_name,
            existing_type=column_type,
            existing_nullable=nullable,
            comment=comment,
        )
    op.execute("ALTER TABLE ai_prompt_publish_audit COMMENT = 'Prompt 人工发布审计记录表'")


def _create_missing_indexes() -> None:
    expected_indexes = {
        "ix_ai_prompt_publish_audit_publish_id": (["publish_id"], True),
        "ix_ai_prompt_publish_audit_gate_id": (["gate_id"], False),
        "ix_ai_prompt_publish_audit_prompt_id": (["prompt_id"], False),
        "ix_ai_prompt_publish_audit_prompt_name": (["prompt_name"], False),
    }
    existing_indexes = {index["name"] for index in inspect(op.get_bind()).get_indexes("ai_prompt_publish_audit")}
    for index_name, (columns, unique) in expected_indexes.items():
        if index_name not in existing_indexes:
            op.create_index(index_name, "ai_prompt_publish_audit", columns, unique=unique)


def upgrade() -> None:
    # 学习环境的 FastAPI startup 可能已经通过 create_all 建表；迁移仍负责补齐治理元数据。
    table_names = set(inspect(op.get_bind()).get_table_names())
    if "ai_prompt_publish_audit" not in table_names:
        _create_publish_audit_table()
    _create_missing_indexes()
    _apply_comments()


def downgrade() -> None:
    op.drop_index("ix_ai_prompt_publish_audit_prompt_name", table_name="ai_prompt_publish_audit")
    op.drop_index("ix_ai_prompt_publish_audit_prompt_id", table_name="ai_prompt_publish_audit")
    op.drop_index("ix_ai_prompt_publish_audit_gate_id", table_name="ai_prompt_publish_audit")
    op.drop_index("ix_ai_prompt_publish_audit_publish_id", table_name="ai_prompt_publish_audit")
    op.drop_table("ai_prompt_publish_audit")
