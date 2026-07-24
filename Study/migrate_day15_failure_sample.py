"""
Day15 临时迁移脚本：创建 ai_failure_sample 表。

说明：
当前虚拟环境里的 Alembic 命令入口不可用，所以先用这个脚本完成本地学习环境迁移。
"""

from sqlalchemy import inspect, text

from day04_app.database import engine


CREATE_TABLE_SQL = """
CREATE TABLE ai_failure_sample (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sample_id VARCHAR(64) NOT NULL UNIQUE,
    trace_id VARCHAR(64) NULL,
    task_id VARCHAR(64) NULL,
    session_id VARCHAR(64) NULL,
    message_id VARCHAR(64) NULL,
    call_type VARCHAR(64) NOT NULL,
    model VARCHAR(64) NULL,
    schema_type VARCHAR(64) NOT NULL,
    schema_version VARCHAR(32) NOT NULL,
    error_type VARCHAR(64) NOT NULL,
    error_message TEXT NOT NULL,
    raw_text TEXT NULL,
    validation_error TEXT NULL,
    created_at DATETIME NOT NULL
)
"""


INDEXES = {
    "ix_ai_failure_sample_sample_id": "sample_id",
    "ix_ai_failure_sample_trace_id": "trace_id",
    "ix_ai_failure_sample_task_id": "task_id",
    "ix_ai_failure_sample_session_id": "session_id",
    "ix_ai_failure_sample_message_id": "message_id",
    "ix_ai_failure_sample_call_type": "call_type",
    "ix_ai_failure_sample_schema_type": "schema_type",
    "ix_ai_failure_sample_schema_version": "schema_version",
    "ix_ai_failure_sample_error_type": "error_type",
}


def main() -> None:
    inspector = inspect(engine)
    with engine.begin() as connection:
        if "ai_failure_sample" not in inspector.get_table_names():
            connection.execute(text(CREATE_TABLE_SQL))
            print("已创建 ai_failure_sample")
        else:
            print("ai_failure_sample 已存在，跳过")

    inspector = inspect(engine)
    existing_indexes = {index["name"] for index in inspector.get_indexes("ai_failure_sample")}
    with engine.begin() as connection:
        for index_name, column_name in INDEXES.items():
            if index_name in existing_indexes:
                print(f"{index_name} 已存在，跳过")
                continue
            connection.execute(
                text(f"CREATE INDEX {index_name} ON ai_failure_sample ({column_name})")
            )
            print(f"已创建 {index_name}")
    print("Day15 ai_failure_sample 数据库迁移完成")


if __name__ == "__main__":
    main()
