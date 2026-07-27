"""Day17 add Chinese comments to evaluation gate table.

Revision ID: 20260725_002
Revises: 20260725_001
Create Date: 2026-07-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260725_002"
down_revision: str | None = "20260725_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # MySQL 会通过 MODIFY COLUMN 写入字段 COMMENT；existing_type/nullable 防止修改字段定义。
    op.alter_column(
        "ai_eval_gate_decision", "id", existing_type=sa.Integer(), existing_nullable=False, comment="数据库自增主键"
    )
    op.alter_column(
        "ai_eval_gate_decision", "gate_id", existing_type=sa.String(length=64), existing_nullable=False, comment="评测门禁业务唯一 ID"
    )
    op.alter_column(
        "ai_eval_gate_decision", "baseline_run_id", existing_type=sa.String(length=64), existing_nullable=False, comment="已上线基线 Prompt 的评测运行 ID"
    )
    op.alter_column(
        "ai_eval_gate_decision", "candidate_run_id", existing_type=sa.String(length=64), existing_nullable=False, comment="候选 Prompt 的评测运行 ID"
    )
    op.alter_column(
        "ai_eval_gate_decision", "prompt_name", existing_type=sa.String(length=64), existing_nullable=False, comment="业务 Prompt 名称，例如 work_order_analysis"
    )
    op.alter_column(
        "ai_eval_gate_decision", "dataset_version", existing_type=sa.String(length=64), existing_nullable=False, comment="本次比较使用的评测数据集版本"
    )
    op.alter_column(
        "ai_eval_gate_decision", "decision", existing_type=sa.String(length=32), existing_nullable=False, comment="门禁结论：pass、reject 或 manual_review"
    )
    op.alter_column(
        "ai_eval_gate_decision", "comparison_json", existing_type=sa.Text(), existing_nullable=False, comment="基线与候选评测指标差异 JSON"
    )
    op.alter_column(
        "ai_eval_gate_decision", "reason_json", existing_type=sa.Text(), existing_nullable=False, comment="命中门禁规则与判定原因 JSON"
    )
    op.alter_column(
        "ai_eval_gate_decision", "rule_snapshot_json", existing_type=sa.Text(), existing_nullable=False, comment="本次门禁使用的规则快照 JSON"
    )
    op.alter_column(
        "ai_eval_gate_decision", "created_at", existing_type=sa.DateTime(), existing_nullable=False, comment="门禁判定创建时间"
    )
    op.execute("ALTER TABLE ai_eval_gate_decision COMMENT = 'Prompt 评测准入门禁记录表'")


def downgrade() -> None:
    # 注释属于元数据；回退时保留空注释，不删除任何业务记录。
    for column_name, column_type in (
        ("id", sa.Integer()),
        ("gate_id", sa.String(length=64)),
        ("baseline_run_id", sa.String(length=64)),
        ("candidate_run_id", sa.String(length=64)),
        ("prompt_name", sa.String(length=64)),
        ("dataset_version", sa.String(length=64)),
        ("decision", sa.String(length=32)),
        ("comparison_json", sa.Text()),
        ("reason_json", sa.Text()),
        ("rule_snapshot_json", sa.Text()),
        ("created_at", sa.DateTime()),
    ):
        op.alter_column(
            "ai_eval_gate_decision",
            column_name,
            existing_type=column_type,
            existing_nullable=False,
            comment=None,
        )
    op.execute("ALTER TABLE ai_eval_gate_decision COMMENT = ''")
