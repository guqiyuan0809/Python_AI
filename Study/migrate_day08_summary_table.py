"""
Day08 数据库迁移脚本

创建 chat_session_summary 表，用于保存会话摘要版本。
后续学习 Alembic 后，会用正式迁移工具替代这种脚本。
"""

from day04_app.database import Base, engine
from day04_app.models import ChatSessionSummary


def main() -> None:
    # 学习阶段先用 SQLAlchemy 直接创建新增表。
    # 后续学习 Alembic 后，会改成正式的数据库迁移脚本。
    ChatSessionSummary.__table__.create(bind=engine, checkfirst=True)
    print("chat_session_summary 表初始化完成")


if __name__ == "__main__":
    main()
