"""Day20 persist candidate document version rebuild note.

Revision ID: 20260731_001
Revises: 20260730_004
Create Date: 2026-07-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260731_001"
down_revision: str | None = "20260730_004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    existing_columns = {
        column["name"]
        for column in inspect(op.get_bind()).get_columns("knowledge_document_version")
    }
    if "rebuild_note" not in existing_columns:
        op.add_column(
            "knowledge_document_version",
            sa.Column(
                "rebuild_note",
                sa.String(length=500),
                nullable=True,
                comment="创建候选索引版本的变更说明，例如调整切块参数或更新原文件",
            ),
        )


def downgrade() -> None:
    existing_columns = {
        column["name"]
        for column in inspect(op.get_bind()).get_columns("knowledge_document_version")
    }
    if "rebuild_note" in existing_columns:
        op.drop_column("knowledge_document_version", "rebuild_note")
