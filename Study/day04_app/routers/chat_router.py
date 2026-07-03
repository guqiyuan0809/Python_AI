"""
聊天接口路由层

类似 Java 里的 ChatController。
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from day04_app.common.exceptions import ModelCallException
from day04_app.common.response import ApiResponse, success
from day04_app.database import get_db
from day04_app.schemas.chat_schema import (
    ChatMessageItem,
    ChatRequest,
    ChatResponse,
    CreateSessionResponse,
    SessionChatRequest,
    SessionMessagesResponse,
)
from day04_app.services.chat_service import (
    safe_chat,
    safe_chat_with_messages,
    stream_chat_events,
)
from day04_app.services.session_service import (
    add_message,
    build_messages,
    create_session,
    get_session_messages,
    update_message,
)
from settings import settings


router = APIRouter(prefix="/api/chat", tags=["chat"])


def to_message_item(message) -> ChatMessageItem:
    return ChatMessageItem(
        message_id=message.message_id,
        session_id=message.session_id,
        trace_id=message.trace_id,
        stream_id=message.stream_id,
        role=message.role,
        content=message.content,
        model=message.model,
        prompt_tokens=message.prompt_tokens,
        completion_tokens=message.completion_tokens,
        total_tokens=message.total_tokens,
        status=message.status,
        error_message=message.error_message,
        created_at=message.created_at.isoformat(timespec="seconds"),
    )


@router.post("", response_model=ApiResponse[ChatResponse])
def chat(request_body: ChatRequest, request: Request) -> ApiResponse[ChatResponse]:
    result = safe_chat(request_body.message)
    return success(result, trace_id=request.state.trace_id)


@router.post("/sessions", response_model=ApiResponse[CreateSessionResponse])
def create_chat_session(
    request: Request, db: Session = Depends(get_db)
) -> ApiResponse[CreateSessionResponse]:
    session_id = create_session(db)
    return success(
        CreateSessionResponse(session_id=session_id),
        message="会话创建成功",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=ApiResponse[SessionMessagesResponse],
)
def list_session_messages(
    session_id: str, request: Request, db: Session = Depends(get_db)
) -> ApiResponse[SessionMessagesResponse]:
    messages = get_session_messages(db, session_id)
    return success(
        SessionMessagesResponse(
            session_id=session_id,
            messages=[to_message_item(message) for message in messages],
        ),
        trace_id=request.state.trace_id,
    )


@router.post("/sessions/chat", response_model=ApiResponse[ChatResponse])
def session_chat(
    request_body: SessionChatRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[ChatResponse]:
    trace_id = request.state.trace_id
    messages = build_messages(
        db=db,
        session_id=request_body.session_id,
        current_question=request_body.message,
        history_limit=request_body.history_limit,
    )
    add_message(db, request_body.session_id, "user", request_body.message, trace_id=trace_id)
    assistant_message = add_message(
        db,
        request_body.session_id,
        "assistant",
        "AI 回答生成中",
        trace_id=trace_id,
        model=settings.dashscope_model,
        status="pending",
    )

    try:
        result = safe_chat_with_messages(messages)
    except ModelCallException as exc:
        update_message(
            db,
            assistant_message.message_id,
            content=exc.message,
            status="error",
            error_message=exc.message,
        )
        raise

    update_message(
        db,
        assistant_message.message_id,
        content=result.answer,
        status="success",
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
    )
    return success(result, trace_id=trace_id)


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
