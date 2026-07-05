"""
Day08 数据库迁移脚本

给 chat_session 表增加 summary 字段。
后续学习 Alembic 后，会用正式迁移工具替代这种脚本。
"""

from sqlalchemy import inspect, text

from day04_app.database import engine


def main() -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("chat_session")}

    if "summary" in columns:
        print("summary 字段已存在，无需迁移")
        return

    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE chat_session ADD COLUMN summary TEXT NULL"))
    print("summary 字段添加完成")


if __name__ == "__main__":
    main()
