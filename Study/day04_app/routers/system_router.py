"""
系统接口路由层

放健康检查这类通用接口。
"""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from day04_app.common.response import ApiResponse, success
from day04_app.database import engine
from settings import settings


router = APIRouter(tags=["system"])


def _live_payload(request: Request) -> ApiResponse[dict]:
    return success(
        {
            "status": "UP",
            "environment": settings.app_env,
            "model": settings.dashscope_model,
        },
        message="service is running",
        trace_id=request.state.trace_id,
    )


@router.get("/health/live")
def live(request: Request) -> ApiResponse[dict]:
    """进程存活探针：不访问外部依赖，供容器判断进程是否还活着。"""
    return _live_payload(request)


@router.get("/health/ready", response_model=None)
def ready(request: Request) -> ApiResponse[dict] | JSONResponse:
    """就绪探针：只读检查核心数据库连接是否可用。"""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "code": 503,
                "message": "service is not ready",
                "data": {"status": "DOWN", "dependencies": {"database": "DOWN"}},
                "trace_id": request.state.trace_id,
            },
        )
    return success(
        {
            "status": "UP",
            "environment": settings.app_env,
            "dependencies": {"database": "UP"},
        },
        message="service is ready",
        trace_id=request.state.trace_id,
    )


@router.get("/health")
def health(request: Request) -> ApiResponse[dict]:
    """兼容旧客户端；新部署探针应分别使用 live/ready。"""
    return _live_payload(request)
