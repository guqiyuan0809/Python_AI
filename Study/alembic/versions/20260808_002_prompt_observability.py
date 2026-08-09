"""Day26 Prompt identity fields for AI call observability."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260808_002"
down_revision: str | None = "20260808_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _column_names() -> set[str]:
    return {column["name"] for column in inspect(op.get_bind()).get_columns("ai_call_log")}


def _index_names() -> set[str]:
    return {index["name"] for index in inspect(op.get_bind()).get_indexes("ai_call_log")}


def upgrade() -> None:
    columns = _column_names()
    definitions = {
        "prompt_id": sa.Column(
            "prompt_id", sa.String(length=64), nullable=True,
            comment="Prompt Registry 业务 ID；代码托管 Prompt 为空",
        ),
        "prompt_name": sa.Column(
            "prompt_name", sa.String(length=128), nullable=True,
            comment="本次模型调用使用的 Prompt 名称",
        ),
        "prompt_version": sa.Column(
            "prompt_version", sa.String(length=64), nullable=True,
            comment="本次模型调用实际使用的 Prompt 版本",
        ),
        "prompt_template_hash": sa.Column(
            "prompt_template_hash", sa.String(length=64), nullable=True,
            comment="Prompt 稳定模板 SHA-256，不包含业务输入",
        ),
    }
    for name, column in definitions.items():
        if name not in columns:
            op.add_column("ai_call_log", column)

    indexes = _index_names()
    for name, column in (("ix_acl_prompt_name", "prompt_name"), ("ix_acl_prompt_version", "prompt_version")):
        if name not in indexes:
            op.create_index(name, "ai_call_log", [column])


def downgrade() -> None:
    indexes = _index_names()
    for name in ("ix_acl_prompt_version", "ix_acl_prompt_name"):
        if name in indexes:
            op.drop_index(name, table_name="ai_call_log")
    columns = _column_names()
    for name in ("prompt_template_hash", "prompt_version", "prompt_name", "prompt_id"):
        if name in columns:
            op.drop_column("ai_call_log", name)
