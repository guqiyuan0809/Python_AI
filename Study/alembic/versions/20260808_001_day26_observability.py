"""Day26 extend AI call logs for trace observability.

Revision ID: 20260808_001
Revises: 20260807_001
Create Date: 2026-08-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260808_001"
down_revision: str | None = "20260807_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _column_names() -> set[str]:
    return {column["name"] for column in inspect(op.get_bind()).get_columns("ai_call_log")}


def _index_names() -> set[str]:
    return {index["name"] for index in inspect(op.get_bind()).get_indexes("ai_call_log")}


def upgrade() -> None:
    columns = _column_names()
    if "task_id" not in columns:
        op.add_column(
            "ai_call_log",
            sa.Column("task_id", sa.String(length=64), nullable=True, comment="关联 ai_async_task 的业务任务 ID"),
        )
    if "run_id" not in columns:
        op.add_column(
            "ai_call_log",
            sa.Column("run_id", sa.String(length=64), nullable=True, comment="关联评测或编排运行的业务 ID"),
        )
    if "stage" not in columns:
        op.add_column(
            "ai_call_log",
            sa.Column("stage", sa.String(length=64), nullable=True, comment="调用来源内的可观测阶段名称"),
        )
    if "detail_json" not in columns:
        op.add_column(
            "ai_call_log",
            sa.Column("detail_json", sa.Text(), nullable=True, comment="可观测事件脱敏详情 JSON"),
        )

    indexes = _index_names()
    if "ix_acl_task" not in indexes:
        op.create_index("ix_acl_task", "ai_call_log", ["task_id"])
    if "ix_acl_run" not in indexes:
        op.create_index("ix_acl_run", "ai_call_log", ["run_id"])
    if "ix_acl_stage" not in indexes:
        op.create_index("ix_acl_stage", "ai_call_log", ["stage"])


def downgrade() -> None:
    indexes = _index_names()
    if "ix_acl_stage" in indexes:
        op.drop_index("ix_acl_stage", table_name="ai_call_log")
    if "ix_acl_run" in indexes:
        op.drop_index("ix_acl_run", table_name="ai_call_log")
    if "ix_acl_task" in indexes:
        op.drop_index("ix_acl_task", table_name="ai_call_log")
    columns = _column_names()
    for column_name in ("detail_json", "stage", "run_id", "task_id"):
        if column_name in columns:
            op.drop_column("ai_call_log", column_name)
