"""Day16 add Prompt and evaluation master tables.

Revision ID: 20260724_002
Revises: 20260724_001
Create Date: 2026-07-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260724_002"
down_revision: str | None = "20260724_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_ai_prompt_version_table() -> None:
    op.create_table(
        "ai_prompt_version",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("prompt_id", sa.String(length=64), nullable=False),
        sa.Column("prompt_name", sa.String(length=64), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("user_prompt_template", sa.Text(), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("max_tokens", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="draft"),
        sa.Column("created_by", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("prompt_id"),
    )
    op.create_index("ix_ai_prompt_version_prompt_id", "ai_prompt_version", ["prompt_id"])
    op.create_index("ix_ai_prompt_version_prompt_name", "ai_prompt_version", ["prompt_name"])
    op.create_index("ix_ai_prompt_version_prompt_version", "ai_prompt_version", ["prompt_version"])
    op.create_index("ix_ai_prompt_version_status", "ai_prompt_version", ["status"])


def _create_ai_eval_dataset_table() -> None:
    op.create_table(
        "ai_eval_dataset",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_name", sa.String(length=64), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=True),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("created_by", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_id"),
    )
    op.create_index("ix_ai_eval_dataset_dataset_id", "ai_eval_dataset", ["dataset_id"])
    op.create_index("ix_ai_eval_dataset_dataset_name", "ai_eval_dataset", ["dataset_name"])
    op.create_index("ix_ai_eval_dataset_dataset_version", "ai_eval_dataset", ["dataset_version"])
    op.create_index("ix_ai_eval_dataset_status", "ai_eval_dataset", ["status"])


def _create_ai_eval_sample_table() -> None:
    op.create_table(
        "ai_eval_sample",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("sample_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("sample_type", sa.String(length=32), nullable=False, server_default="normal"),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("expected_json", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False, server_default="manual"),
        sa.Column("source_ref_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("created_by", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sample_id"),
    )
    op.create_index("ix_ai_eval_sample_sample_id", "ai_eval_sample", ["sample_id"])
    op.create_index("ix_ai_eval_sample_dataset_id", "ai_eval_sample", ["dataset_id"])
    op.create_index("ix_ai_eval_sample_dataset_version", "ai_eval_sample", ["dataset_version"])
    op.create_index("ix_ai_eval_sample_sample_type", "ai_eval_sample", ["sample_type"])
    op.create_index("ix_ai_eval_sample_source_type", "ai_eval_sample", ["source_type"])
    op.create_index("ix_ai_eval_sample_source_ref_id", "ai_eval_sample", ["source_ref_id"])
    op.create_index("ix_ai_eval_sample_status", "ai_eval_sample", ["status"])


def upgrade() -> None:
    # Day16 曾使用临时脚本建表；兼容已有环境，同时保证新环境可从 Alembic 完整建表。
    table_names = set(inspect(op.get_bind()).get_table_names())
    if "ai_prompt_version" not in table_names:
        _create_ai_prompt_version_table()
    if "ai_eval_dataset" not in table_names:
        _create_ai_eval_dataset_table()
    if "ai_eval_sample" not in table_names:
        _create_ai_eval_sample_table()


def downgrade() -> None:
    table_names = set(inspect(op.get_bind()).get_table_names())
    if "ai_eval_sample" in table_names:
        op.drop_table("ai_eval_sample")
    if "ai_eval_dataset" in table_names:
        op.drop_table("ai_eval_dataset")
    if "ai_prompt_version" in table_names:
        op.drop_table("ai_prompt_version")
