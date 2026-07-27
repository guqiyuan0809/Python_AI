"""Day17 add Prompt evaluation gate decisions.

Revision ID: 20260725_001
Revises: 20260724_001
Create Date: 2026-07-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260725_001"
down_revision: str | None = "20260724_002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ai_eval_gate_decision",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("gate_id", sa.String(length=64), nullable=False),
        sa.Column("baseline_run_id", sa.String(length=64), nullable=False),
        sa.Column("candidate_run_id", sa.String(length=64), nullable=False),
        sa.Column("prompt_name", sa.String(length=64), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("comparison_json", sa.Text(), nullable=False),
        sa.Column("reason_json", sa.Text(), nullable=False),
        sa.Column("rule_snapshot_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ai_eval_gate_decision_gate_id", "ai_eval_gate_decision", ["gate_id"], unique=True)
    op.create_index("ix_ai_eval_gate_decision_baseline_run_id", "ai_eval_gate_decision", ["baseline_run_id"])
    op.create_index("ix_ai_eval_gate_decision_candidate_run_id", "ai_eval_gate_decision", ["candidate_run_id"])
    op.create_index("ix_ai_eval_gate_decision_prompt_name", "ai_eval_gate_decision", ["prompt_name"])
    op.create_index("ix_ai_eval_gate_decision_dataset_version", "ai_eval_gate_decision", ["dataset_version"])
    op.create_index("ix_ai_eval_gate_decision_decision", "ai_eval_gate_decision", ["decision"])


def downgrade() -> None:
    op.drop_index("ix_ai_eval_gate_decision_decision", table_name="ai_eval_gate_decision")
    op.drop_index("ix_ai_eval_gate_decision_dataset_version", table_name="ai_eval_gate_decision")
    op.drop_index("ix_ai_eval_gate_decision_prompt_name", table_name="ai_eval_gate_decision")
    op.drop_index("ix_ai_eval_gate_decision_candidate_run_id", table_name="ai_eval_gate_decision")
    op.drop_index("ix_ai_eval_gate_decision_baseline_run_id", table_name="ai_eval_gate_decision")
    op.drop_index("ix_ai_eval_gate_decision_gate_id", table_name="ai_eval_gate_decision")
    op.drop_table("ai_eval_gate_decision")
