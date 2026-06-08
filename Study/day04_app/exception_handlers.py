"""
全局异常处理

类似 Java Spring Boot 中的 @RestControllerAdvice。
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from day04_app.common.exceptions import BusinessException
from day04_app.common.response import fail


def get_trace_id(request: Request) -> str | None:
    return getattr(request.state, "trace_id", None)


def get_validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "请求参数校验失败"

    first_error = errors[0]
    field_path = ".".join(str(item) for item in first_error.get("loc", []))
    error_type = first_error.get("type")
    if error_type == "missing":
        error_message = "字段必填"
    elif error_type == "string_too_short":
        error_message = "字段长度过短"
    else:
        error_message = first_error.get("msg", "参数不合法")
    return f"{field_path}: {error_message}"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BusinessException)
    async def business_exception_handler(
        request: Request, exc: BusinessException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content=fail(
                code=exc.code,
                message=exc.message,
                trace_id=get_trace_id(request),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=fail(
                code=40001,
                message=get_validation_message(exc),
                trace_id=get_trace_id(request),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=fail(
                code=50000,
                message="服务器内部错误",
                trace_id=get_trace_id(request),
            ).model_dump(),
        )
