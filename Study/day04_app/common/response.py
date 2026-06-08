"""
统一响应体

类似 Java 项目里的 Result<T> / R<T>。
"""

from typing import Generic, TypeVar

from pydantic import BaseModel


T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int
    message: str
    data: T | None = None
    trace_id: str | None = None


def success(
    data: T | None = None, message: str = "success", trace_id: str | None = None
) -> ApiResponse[T]:
    return ApiResponse(code=200, message=message, data=data, trace_id=trace_id)


def fail(
    code: int = 500, message: str = "server error", trace_id: str | None = None
) -> ApiResponse[None]:
    return ApiResponse(code=code, message=message, data=None, trace_id=trace_id)
