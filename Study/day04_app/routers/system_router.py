"""
系统接口路由层

放健康检查这类通用接口。
"""

from fastapi import APIRouter, Request

from day04_app.common.response import ApiResponse, success
from settings import settings


router = APIRouter(tags=["system"])


@router.get("/health")
def health(request: Request) -> ApiResponse[dict]:
    return success(
        {
            "model": settings.dashscope_model,
        },
        message="service is running",
        trace_id=request.state.trace_id,
    )
