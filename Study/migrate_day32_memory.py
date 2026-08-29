"""Day32：创建会话轮次、Agent 工作记忆与长期语义记忆表，并补齐摘要/消息字段。

学习项目仍使用 SQLAlchemy 的显式迁移脚本；部署环境应将同等 DDL 纳入 Alembic/Flyway，
并在发布前执行，而不是依赖 Web 进程 ``create_all`` 隐式改表。
"""

from sqlalchemy import inspect, text

from day04_app.database import Base, engine
from day04_app.models import (
    AgentWorkingMemorySnapshot,
    ChatSessionTurn,
    SessionMemory,
)


def _ensure_column(table_name: str, column_name: str, ddl: str) -> None:
    inspector = inspect(engine)
    existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
    if column_name not in existing_columns:
        with engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {ddl}"))


def _ensure_index(table_name: str, index_name: str, columns: str) -> None:
    """显式补齐既有表的索引；``table.create(checkfirst=True)`` 不会改旧表结构。"""

    inspector = inspect(engine)
    existing_indexes = {index["name"] for index in inspector.get_indexes(table_name)}
    if index_name not in existing_indexes:
        with engine.begin() as connection:
            connection.execute(
                text(f"CREATE INDEX {index_name} ON {table_name} ({columns})")
            )


def main() -> None:
    # 新表可幂等创建；不要调用 Base.metadata.create_all，避免把不属于本次迁移的表也带入。
    for table in (
        ChatSessionTurn.__table__,
        AgentWorkingMemorySnapshot.__table__,
        SessionMemory.__table__,
    ):
        table.create(bind=engine, checkfirst=True)

    # 已有表通过加列向前兼容。索引由模型声明；本地 auto_create_tables 会补齐，部署应由
    # 正式迁移管理工具创建对应索引。
    _ensure_column("chat_message", "turn_no", "turn_no INT NULL")
    _ensure_column("chat_session_summary", "summary_until_turn_no", "summary_until_turn_no INT NULL")
    _ensure_column("chat_session_summary", "source_turn_count", "source_turn_count INT NOT NULL DEFAULT 0")
    _ensure_column("chat_session_summary", "source_token_count", "source_token_count INT NOT NULL DEFAULT 0")
    _ensure_column("chat_session_turn", "task_id", "task_id VARCHAR(64) NULL")
    _ensure_column("session_memory", "embedding_error_message", "embedding_error_message TEXT NULL")
    _ensure_index("chat_session_turn", "ix_chat_turn_task", "task_id")
    print("Day32 memory schema migration completed")


if __name__ == "__main__":
    main()
