"""
请求中间件

用于生成 trace_id，并记录基础请求日志。
"""

import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request


logger = logging.getLogger("day04_app")


def register_middlewares(app: FastAPI) -> None:
    @app.middleware("http")
    async def trace_log_middleware(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or uuid4().hex
        request.state.trace_id = trace_id

        start_time = time.perf_counter()
        response = await call_next(request)
        cost_ms = round((time.perf_counter() - start_time) * 1000, 2)

        response.headers["X-Trace-Id"] = trace_id
        logger.info(
            "trace_id=%s method=%s path=%s status_code=%s cost_ms=%s",
            trace_id,
            request.method,
            request.url.path,
            response.status_code,
            cost_ms,
        )
        return response
