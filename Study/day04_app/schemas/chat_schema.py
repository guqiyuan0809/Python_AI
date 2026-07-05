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


class ChatSessionItem(BaseModel):
    session_id: str
    user_id: str | None = None
    title: str | None = None
    summary: str | None = None
    status: str
    created_at: str
    updated_at: str


class SessionListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[ChatSessionItem]


class RefreshSessionSummaryResponse(BaseModel):
    session_id: str
    # 本次新生成的摘要版本 ID。
    summary_id: str | None = None
    # 最新摘要内容，会作为后续长对话的长期上下文。
    summary: str
    # 表示这份摘要已经覆盖到哪一条消息。
    summary_until_message_id: str | None = None
    # 当前会话下的摘要版本号。
    version: int | None = None


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


class SessionMessagesPageResponse(BaseModel):
    session_id: str
    summary: str | None = None
    total: int
    page: int
    page_size: int
    messages: list[ChatMessageItem]
