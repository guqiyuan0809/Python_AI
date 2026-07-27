"""day15 eval tables

Revision ID: 20260724_001
Revises: 20260723_002
Create Date: 2026-07-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260724_001"
# 必须引用前一个迁移的完整 revision ID，文件名中的日期序号不是 Alembic 的版本号。
down_revision: str | None = "20260723_002_day15_failure_sample"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ai_eval_run",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("prompt_name", sa.String(length=64), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("schema_valid_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column("category_accuracy", sa.Float(), nullable=False, server_default="0"),
        sa.Column("risk_level_accuracy", sa.Float(), nullable=False, server_default="0"),
        sa.Column("human_review_accuracy", sa.Float(), nullable=False, server_default="0"),
        sa.Column("avg_total_tokens", sa.Float(), nullable=True),
        sa.Column("avg_cost_ms", sa.Float(), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ai_eval_run_run_id", "ai_eval_run", ["run_id"], unique=True)
    op.create_index("ix_ai_eval_run_prompt_name", "ai_eval_run", ["prompt_name"])
    op.create_index("ix_ai_eval_run_prompt_version", "ai_eval_run", ["prompt_version"])
    op.create_index("ix_ai_eval_run_dataset_version", "ai_eval_run", ["dataset_version"])

    op.create_table(
        "ai_eval_case_result",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("sample_id", sa.String(length=64), nullable=False),
        sa.Column("schema_valid", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("category_match", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("risk_level_match", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("human_review_match", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_ms", sa.Integer(), nullable=True),
        sa.Column("error_type", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("expected_json", sa.Text(), nullable=True),
        sa.Column("actual_json", sa.Text(), nullable=True),
        sa.Column("row_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ai_eval_case_result_run_id", "ai_eval_case_result", ["run_id"])
    op.create_index("ix_ai_eval_case_result_sample_id", "ai_eval_case_result", ["sample_id"])
    op.create_index("ix_ai_eval_case_result_error_type", "ai_eval_case_result", ["error_type"])


def downgrade() -> None:
    op.drop_index("ix_ai_eval_case_result_error_type", table_name="ai_eval_case_result")
    op.drop_index("ix_ai_eval_case_result_sample_id", table_name="ai_eval_case_result")
    op.drop_index("ix_ai_eval_case_result_run_id", table_name="ai_eval_case_result")
    op.drop_table("ai_eval_case_result")

    op.drop_index("ix_ai_eval_run_dataset_version", table_name="ai_eval_run")
    op.drop_index("ix_ai_eval_run_prompt_version", table_name="ai_eval_run")
    op.drop_index("ix_ai_eval_run_prompt_name", table_name="ai_eval_run")
    op.drop_index("ix_ai_eval_run_run_id", table_name="ai_eval_run")
    op.drop_table("ai_eval_run")
