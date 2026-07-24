"""Day15 failure sample table."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260723_002_day15_failure_sample"
down_revision = "20260723_001_day15_error_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ai_failure_sample" in inspector.get_table_names():
        return

    op.create_table(
        "ai_failure_sample",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("sample_id", sa.String(length=64), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("task_id", sa.String(length=64), nullable=True),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("message_id", sa.String(length=64), nullable=True),
        sa.Column("call_type", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("schema_type", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=32), nullable=False),
        sa.Column("error_type", sa.String(length=64), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("validation_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sample_id"),
    )
    op.create_index("ix_ai_failure_sample_sample_id", "ai_failure_sample", ["sample_id"], unique=True)
    op.create_index("ix_ai_failure_sample_trace_id", "ai_failure_sample", ["trace_id"], unique=False)
    op.create_index("ix_ai_failure_sample_task_id", "ai_failure_sample", ["task_id"], unique=False)
    op.create_index("ix_ai_failure_sample_session_id", "ai_failure_sample", ["session_id"], unique=False)
    op.create_index("ix_ai_failure_sample_message_id", "ai_failure_sample", ["message_id"], unique=False)
    op.create_index("ix_ai_failure_sample_call_type", "ai_failure_sample", ["call_type"], unique=False)
    op.create_index("ix_ai_failure_sample_schema_type", "ai_failure_sample", ["schema_type"], unique=False)
    op.create_index("ix_ai_failure_sample_schema_version", "ai_failure_sample", ["schema_version"], unique=False)
    op.create_index("ix_ai_failure_sample_error_type", "ai_failure_sample", ["error_type"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "ai_failure_sample" not in inspector.get_table_names():
        return

    indexes = {index["name"] for index in inspector.get_indexes("ai_failure_sample")}
    for index_name in [
        "ix_ai_failure_sample_error_type",
        "ix_ai_failure_sample_schema_version",
        "ix_ai_failure_sample_schema_type",
        "ix_ai_failure_sample_call_type",
        "ix_ai_failure_sample_message_id",
        "ix_ai_failure_sample_session_id",
        "ix_ai_failure_sample_task_id",
        "ix_ai_failure_sample_trace_id",
        "ix_ai_failure_sample_sample_id",
    ]:
        if index_name in indexes:
            op.drop_index(index_name, table_name="ai_failure_sample")
    op.drop_table("ai_failure_sample")
