"""
更像 Java 启动类的启动入口

以后你既可以用 uvicorn 命令启动，
也可以直接 python run_day04.py 启动。
"""

import uvicorn

from settings import settings


if __name__ == "__main__":
    uvicorn.run(
        "day04_app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
