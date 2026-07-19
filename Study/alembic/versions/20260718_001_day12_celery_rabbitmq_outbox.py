"""Day12 Celery RabbitMQ Outbox migration."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260718_001_day12_outbox"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    task_columns = {column["name"] for column in inspector.get_columns("ai_async_task")}
    task_indexes = {index["name"] for index in inspector.get_indexes("ai_async_task")}

    # 任务表补充 Broker 消息 ID，便于排查 RabbitMQ/Celery 投递链路。
    if "broker_task_id" not in task_columns:
        op.add_column(
            "ai_async_task",
            sa.Column("broker_task_id", sa.String(length=64), nullable=True),
        )
    if "ix_ai_async_task_broker_task_id" not in task_indexes:
        op.create_index(
            "ix_ai_async_task_broker_task_id",
            "ai_async_task",
            ["broker_task_id"],
            unique=False,
        )

    # 自动重试字段放在业务任务表，前端/Java 查询 task_id 时可以直接看到重试状态。
    if "retry_count" not in task_columns:
        op.add_column(
            "ai_async_task",
            sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        )
    if "max_retries" not in task_columns:
        op.add_column(
            "ai_async_task",
            sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
        )

    # Outbox 是本地消息表，负责解决 MySQL 与 RabbitMQ 双写一致性问题。
    if "ai_task_outbox" in inspector.get_table_names():
        return

    op.create_table(
        "ai_task_outbox",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("publish_retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("available_at", sa.DateTime(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("published_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id"),
    )
    op.create_index("ix_ai_task_outbox_event_id", "ai_task_outbox", ["event_id"], unique=True)
    op.create_index("ix_ai_task_outbox_task_id", "ai_task_outbox", ["task_id"], unique=False)
    op.create_index("ix_ai_task_outbox_event_type", "ai_task_outbox", ["event_type"], unique=False)
    op.create_index("ix_ai_task_outbox_status", "ai_task_outbox", ["status"], unique=False)
    op.create_index("ix_ai_task_outbox_available_at", "ai_task_outbox", ["available_at"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    # 回滚时先删索引和 Outbox 表，再撤销任务表新增字段。
    if "ai_task_outbox" in inspector.get_table_names():
        outbox_indexes = {index["name"] for index in inspector.get_indexes("ai_task_outbox")}
        for index_name in [
            "ix_ai_task_outbox_available_at",
            "ix_ai_task_outbox_status",
            "ix_ai_task_outbox_event_type",
            "ix_ai_task_outbox_task_id",
            "ix_ai_task_outbox_event_id",
        ]:
            if index_name in outbox_indexes:
                op.drop_index(index_name, table_name="ai_task_outbox")
        op.drop_table("ai_task_outbox")

    task_columns = {column["name"] for column in inspector.get_columns("ai_async_task")}
    task_indexes = {index["name"] for index in inspector.get_indexes("ai_async_task")}
    if "max_retries" in task_columns:
        op.drop_column("ai_async_task", "max_retries")
    if "retry_count" in task_columns:
        op.drop_column("ai_async_task", "retry_count")
    if "ix_ai_async_task_broker_task_id" in task_indexes:
        op.drop_index("ix_ai_async_task_broker_task_id", table_name="ai_async_task")
    if "broker_task_id" in task_columns:
        op.drop_column("ai_async_task", "broker_task_id")
