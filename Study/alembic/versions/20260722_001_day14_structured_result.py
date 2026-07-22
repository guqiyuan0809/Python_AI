"""Day14 structured result table."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260722_001_day14_structured_result"
down_revision = "20260718_001_day12_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ai_structured_result" in inspector.get_table_names():
        return

    op.create_table(
        "ai_structured_result",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("result_id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=True),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("message_id", sa.String(length=64), nullable=True),
        sa.Column("business_type", sa.String(length=64), nullable=False),
        sa.Column("business_id", sa.String(length=64), nullable=True),
        sa.Column("schema_type", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=32), nullable=False),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("result_id"),
    )
    op.create_index("ix_ai_structured_result_result_id", "ai_structured_result", ["result_id"], unique=True)
    op.create_index("ix_ai_structured_result_task_id", "ai_structured_result", ["task_id"], unique=False)
    op.create_index("ix_ai_structured_result_trace_id", "ai_structured_result", ["trace_id"], unique=False)
    op.create_index("ix_ai_structured_result_session_id", "ai_structured_result", ["session_id"], unique=False)
    op.create_index("ix_ai_structured_result_message_id", "ai_structured_result", ["message_id"], unique=False)
    op.create_index("ix_ai_structured_result_business_type", "ai_structured_result", ["business_type"], unique=False)
    op.create_index("ix_ai_structured_result_business_id", "ai_structured_result", ["business_id"], unique=False)
    op.create_index("ix_ai_structured_result_schema_type", "ai_structured_result", ["schema_type"], unique=False)
    op.create_index("ix_ai_structured_result_status", "ai_structured_result", ["status"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ai_structured_result" not in inspector.get_table_names():
        return

    result_indexes = {index["name"] for index in inspector.get_indexes("ai_structured_result")}
    for index_name in [
        "ix_ai_structured_result_status",
        "ix_ai_structured_result_schema_type",
        "ix_ai_structured_result_business_id",
        "ix_ai_structured_result_business_type",
        "ix_ai_structured_result_message_id",
        "ix_ai_structured_result_session_id",
        "ix_ai_structured_result_trace_id",
        "ix_ai_structured_result_task_id",
        "ix_ai_structured_result_result_id",
    ]:
        if index_name in result_indexes:
            op.drop_index(index_name, table_name="ai_structured_result")
    op.drop_table("ai_structured_result")
