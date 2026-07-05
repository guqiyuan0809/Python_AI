"""
聊天接口请求体和响应体

类似 Java 项目里的 DTO / VO。
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入的问题")


class SessionChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="会话 ID")
    message: str = Field(..., min_length=1, description="用户输入的问题")
    history_limit: int = Field(6, ge=0, le=20, description="携带最近多少条历史消息")


class ChatResponse(BaseModel):
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CreateSessionResponse(BaseModel):
    session_id: str


class RefreshSessionSummaryResponse(BaseModel):
    session_id: str
    summary: str


class ChatMessageItem(BaseModel):
    message_id: str
    session_id: str
    trace_id: str | None = None
    stream_id: str | None = None
    role: str
    content: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    status: str
    error_message: str | None = None
    created_at: str


class SessionMessagesResponse(BaseModel):
    session_id: str
    summary: str | None = None
    messages: list[ChatMessageItem]
