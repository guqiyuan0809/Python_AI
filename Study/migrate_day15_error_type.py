"""
Day15 临时迁移脚本：给已有表补充 error_type 字段。

说明：
当前虚拟环境里的 Alembic 命令入口不可用，所以先用这个脚本完成本地学习环境迁移。
后续修复 Alembic 环境后，仍以 alembic/versions 下的迁移文件作为正式版本记录。
"""

from sqlalchemy import inspect, text

from day04_app.database import engine


TARGET_TABLES = ["chat_message", "ai_call_log", "ai_async_task"]


def add_error_type_column(table_name: str) -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    indexes = {index["name"] for index in inspector.get_indexes(table_name)}
    index_name = f"ix_{table_name}_error_type"

    with engine.begin() as connection:
        if "error_type" not in columns:
            connection.execute(
                text(f"ALTER TABLE {table_name} ADD COLUMN error_type VARCHAR(64) NULL")
            )
            print(f"已添加 {table_name}.error_type")
        else:
            print(f"{table_name}.error_type 已存在，跳过")

        if index_name not in indexes:
            connection.execute(
                text(f"CREATE INDEX {index_name} ON {table_name} (error_type)")
            )
            print(f"已创建 {index_name}")
        else:
            print(f"{index_name} 已存在，跳过")


def main() -> None:
    for table_name in TARGET_TABLES:
        add_error_type_column(table_name)
    print("Day15 error_type 数据库迁移完成")


if __name__ == "__main__":
    main()
