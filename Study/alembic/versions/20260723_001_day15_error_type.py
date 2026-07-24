"""Day15 add error_type columns."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260723_001_day15_error_type"
down_revision = "20260722_001_day14_structured_result"
branch_labels = None
depends_on = None


def _add_error_type_column(table_name: str) -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    indexes = {index["name"] for index in inspector.get_indexes(table_name)}
    column_name = "error_type"
    index_name = f"ix_{table_name}_error_type"

    if column_name not in columns:
        op.add_column(table_name, sa.Column(column_name, sa.String(length=64), nullable=True))
    if index_name not in indexes:
        op.create_index(index_name, table_name, [column_name], unique=False)


def upgrade() -> None:
    _add_error_type_column("chat_message")
    _add_error_type_column("ai_call_log")
    _add_error_type_column("ai_async_task")


def _drop_error_type_column(table_name: str) -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    indexes = {index["name"] for index in inspector.get_indexes(table_name)}
    column_name = "error_type"
    index_name = f"ix_{table_name}_error_type"

    if index_name in indexes:
        op.drop_index(index_name, table_name=table_name)
    if column_name in columns:
        op.drop_column(table_name, column_name)


def downgrade() -> None:
    _drop_error_type_column("ai_async_task")
    _drop_error_type_column("ai_call_log")
    _drop_error_type_column("chat_message")
