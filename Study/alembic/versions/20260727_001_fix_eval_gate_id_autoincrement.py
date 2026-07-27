"""Fix auto increment on evaluation gate primary key.

Revision ID: 20260727_001
Revises: 20260725_002
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260727_001"
down_revision: str | None = "20260725_002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # MySQL MODIFY COLUMN 在补 COMMENT 时需要显式保留 AUTO_INCREMENT，否则自增属性会丢失。
    op.alter_column(
        "ai_eval_gate_decision",
        "id",
        existing_type=sa.Integer(),
        existing_nullable=False,
        existing_autoincrement=True,
        autoincrement=True,
        comment="数据库自增主键",
    )


def downgrade() -> None:
    op.alter_column(
        "ai_eval_gate_decision",
        "id",
        existing_type=sa.Integer(),
        existing_nullable=False,
        existing_autoincrement=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
