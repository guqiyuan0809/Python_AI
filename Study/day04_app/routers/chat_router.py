"""
聊天接口路由层

类似 Java 里的 ChatController。
"""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from day04_app.common.response import ApiResponse, success
from day04_app.schemas.chat_schema import ChatRequest, ChatResponse
from day04_app.services.chat_service import safe_chat, stream_chat_events


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ApiResponse[ChatResponse])
def chat(request_body: ChatRequest, request: Request) -> ApiResponse[ChatResponse]:
    result = safe_chat(request_body.message)
    return success(result, trace_id=request.state.trace_id)


@router.post("/stream")
def stream_chat(request_body: ChatRequest, request: Request) -> StreamingResponse:
    trace_id = request.state.trace_id
    return StreamingResponse(
        stream_chat_events(request_body.message, trace_id),
        media_type="text/event-stream",
        headers={"X-Trace-Id": trace_id},
    )


@router.get("/path-variable/{user_id}")
def path_variable(user_id: str, request: Request) -> ApiResponse[dict]:
    return success(
        {"user_id": user_id},
        message="路由参数",
        trace_id=request.state.trace_id,
    )
