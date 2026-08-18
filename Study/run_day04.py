"""
更像 Java 启动类的启动入口

开发环境可以直接 python run_day04.py 启动；生产环境应由进程管理器或容器
执行 uvicorn，并通过 APP_ENV=production 自动关闭热重载。
"""

import uvicorn

from settings import settings


if __name__ == "__main__":
    uvicorn.run(
        "day04_app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=not settings.is_production,
    )
