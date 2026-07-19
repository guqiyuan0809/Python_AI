"""
Day04 分层版 FastAPI 应用入口

启动命令：
uvicorn day04_app.main:app --reload --host 127.0.0.1 --port 8000
"""

from fastapi import FastAPI

import day04_app.models
from day04_app.database import Base, engine
from day04_app.exception_handlers import register_exception_handlers
from day04_app.middlewares import register_middlewares
from day04_app.routers.chat_router import router as chat_router
from day04_app.routers.system_router import router as system_router
from settings import settings


app = FastAPI(
    title="Day04 AI Service",
    description="基于 APIRouter 拆分后的 FastAPI AI 接口服务",
    version="1.0.0",
)


register_middlewares(app)
register_exception_handlers(app)


@app.on_event("startup")
def startup_check() -> None:
    if not settings.dashscope_api_key:
        raise ValueError("没有读取到通义千问 API Key，请检查 Study/.env")
    # 学习阶段自动创建缺失的数据表；企业项目通常使用 Alembic/Flyway 管理表结构变更。
    Base.metadata.create_all(bind=engine)


app.include_router(system_router)
app.include_router(chat_router)
