"""
初始化数据库表

运行：
python init_db.py
"""

from day04_app.database import Base, engine
from day04_app import models  # noqa: F401


def main() -> None:
    Base.metadata.create_all(bind=engine)
    print("数据库表初始化完成")


if __name__ == "__main__":
    main()
