"""
聊天接口路由层

类似 Java 里的 ChatController。
"""

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from day04_app.common.exceptions import ModelCallException
from day04_app.common.response import ApiResponse, success
from day04_app.database import get_db
from day04_app.schemas.chat_schema import (
    ChatMessageItem,
    ChatRequest,
    ChatResponse,
    ChatSessionItem,
    CreateSessionResponse,
    RefreshSessionSummaryResponse,
    SessionChatRequest,
    SessionListResponse,
    SessionMessagesResponse,
    SessionMessagesPageResponse,
    SessionStatusResponse,
    SessionStreamChatRequest,
    SessionTitleResponse,
    UpdateSessionTitleRequest,
)
from day04_app.services.chat_service import (
    safe_chat,
    safe_chat_with_messages,
    stream_chat_events,
    stream_session_chat_events,
)
from day04_app.services.session_service import (
    add_message,
    archive_session,
    build_messages,
    create_session,
    generate_session_title,
    get_session,
    get_session_messages,
    get_session_messages_page,
    list_sessions,
    refresh_session_summary,
    restore_session,
    should_refresh_summary_for_session,
    update_message,
    update_session_title,
)
from settings import settings


router = APIRouter(prefix="/api/chat", tags=["AI 聊天"])


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


def to_session_item(session) -> ChatSessionItem:
    return ChatSessionItem(
        session_id=session.session_id,
        user_id=session.user_id,
        title=session.title,
        summary=session.summary,
        status=session.status,
        created_at=session.created_at.isoformat(timespec="seconds"),
        updated_at=session.updated_at.isoformat(timespec="seconds"),
    )


@router.post("", response_model=ApiResponse[ChatResponse], summary="普通单轮聊天")
def chat(request_body: ChatRequest, request: Request) -> ApiResponse[ChatResponse]:
    result = safe_chat(request_body.message)
    return success(result, trace_id=request.state.trace_id)


@router.post(
    "/sessions",
    response_model=ApiResponse[CreateSessionResponse],
    summary="创建聊天会话",
)
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
    "/sessions",
    response_model=ApiResponse[SessionListResponse],
    summary="分页查询会话列表",
)
def list_chat_sessions(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(10, ge=1, le=50, description="每页数量"),
    user_id: str | None = Query(None, description="用户 ID，可选筛选条件"),
    db: Session = Depends(get_db),
) -> ApiResponse[SessionListResponse]:
    # 查询会话列表时只返回会话级信息，不返回消息明细，避免列表页数据过大。
    sessions, total = list_sessions(
        db,
        page=page,
        page_size=page_size,
        user_id=user_id,
    )
    return success(
        SessionListResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[to_session_item(session) for session in sessions],
        ),
        trace_id=request.state.trace_id,
    )


@router.patch(
    "/sessions/{session_id}/title",
    response_model=ApiResponse[SessionTitleResponse],
    summary="手动修改会话标题",
)
# 用户输入会话标题
def update_chat_session_title(
    session_id: str,
    request_body: UpdateSessionTitleRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[SessionTitleResponse]:
    # 手动标题以用户输入为准，适合前端提供“重命名会话”功能。
    chat_session = update_session_title(db, session_id, request_body.title)
    return success(
        SessionTitleResponse(
            session_id=chat_session.session_id,
            title=chat_session.title or "",
        ),
        message="会话标题更新成功",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/sessions/{session_id}/title/generate",
    response_model=ApiResponse[SessionTitleResponse],
    summary="自动生成会话标题",
)
# 根据会话历史自动生成标题
def generate_chat_session_title(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[SessionTitleResponse]:
    # 自动生成标题会读取会话历史，优先调用模型生成，失败时使用规则标题兜底。
    chat_session = generate_session_title(db, session_id)
    return success(
        SessionTitleResponse(
            session_id=chat_session.session_id,
            title=chat_session.title or "",
        ),
        message="会话标题生成成功",
        trace_id=request.state.trace_id,
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=ApiResponse[SessionStatusResponse],
    summary="归档会话",
)
def archive_chat_session(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[SessionStatusResponse]:
    # 对外表现为删除会话，底层只做逻辑归档，不物理删除聊天记录。
    chat_session = archive_session(db, session_id)
    return success(
        SessionStatusResponse(
            session_id=chat_session.session_id,
            status=chat_session.status,
        ),
        message="会话已归档",
        trace_id=request.state.trace_id,
    )


@router.patch(
    "/sessions/{session_id}/restore",
    response_model=ApiResponse[SessionStatusResponse],
    summary="恢复归档会话",
)
def restore_chat_session(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[SessionStatusResponse]:
    # 恢复归档会话后，它会重新出现在默认会话列表中。
    chat_session = restore_session(db, session_id)
    return success(
        SessionStatusResponse(
            session_id=chat_session.session_id,
            status=chat_session.status,
        ),
        message="会话已恢复",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=ApiResponse[SessionMessagesResponse],
    summary="查询会话全部消息",
)
def list_session_messages(
    session_id: str, request: Request, db: Session = Depends(get_db)
) -> ApiResponse[SessionMessagesResponse]:
    chat_session = get_session(db, session_id)
    messages = get_session_messages(db, session_id)
    return success(
        SessionMessagesResponse(
            session_id=session_id,
            summary=chat_session.summary,
            messages=[to_message_item(message) for message in messages],
        ),
        trace_id=request.state.trace_id,
    )


@router.get(
    "/sessions/{session_id}/messages/page",
    response_model=ApiResponse[SessionMessagesPageResponse],
    summary="分页查询会话消息",
)
def list_session_messages_page(
    session_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页消息数量"),
    db: Session = Depends(get_db),
) -> ApiResponse[SessionMessagesPageResponse]:
    chat_session = get_session(db, session_id)
    messages, total = get_session_messages_page(
        db,
        session_id=session_id,
        page=page,
        page_size=page_size,
    )
    return success(
        SessionMessagesPageResponse(
            session_id=session_id,
            summary=chat_session.summary,
            total=total,
            page=page,
            page_size=page_size,
            messages=[to_message_item(message) for message in messages],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/sessions/{session_id}/summary/refresh",
    response_model=ApiResponse[RefreshSessionSummaryResponse],
    summary="手动刷新会话摘要",
)
def refresh_summary(
    session_id: str, request: Request, db: Session = Depends(get_db)
) -> ApiResponse[RefreshSessionSummaryResponse]:
    summary_record = refresh_session_summary(db, session_id)
    return success(
        RefreshSessionSummaryResponse(
            session_id=session_id,
            summary_id=summary_record.summary_id,
            summary=summary_record.summary,
            summary_until_message_id=summary_record.summary_until_message_id,
            version=summary_record.version,
        ),
        message="会话摘要刷新成功",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/sessions/chat",
    response_model=ApiResponse[ChatResponse],
    summary="会话多轮聊天",
)
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
    if should_refresh_summary_for_session(db, request_body.session_id):
        refresh_session_summary(db, request_body.session_id)
    return success(result, trace_id=trace_id)


@router.post("/stream", summary="普通流式聊天")
def stream_chat(request_body: ChatRequest, request: Request) -> StreamingResponse:
    trace_id = request.state.trace_id
    return StreamingResponse(
        stream_chat_events(request_body.message, trace_id),
        media_type="text/event-stream",
        headers={"X-Trace-Id": trace_id},
    )


@router.post("/sessions/stream", summary="会话流式聊天")
def stream_session_chat(
    request_body: SessionStreamChatRequest,
    request: Request,
) -> StreamingResponse:
    trace_id = request.state.trace_id

    # SSE 接口返回的是事件流，不适合包一层统一 ApiResponse。
    return StreamingResponse(
        stream_session_chat_events(
            session_id=request_body.session_id,
            message=request_body.message,
            trace_id=trace_id,
            history_limit=request_body.history_limit,
        ),
        media_type="text/event-stream",
        headers={"X-Trace-Id": trace_id},
    )


@router.get("/path-variable/{user_id}", summary="路径参数测试")
def path_variable(user_id: str, request: Request) -> ApiResponse[dict]:
    return success(
        {"user_id": user_id},
        message="路由参数",
        trace_id=request.state.trace_id,
    )
