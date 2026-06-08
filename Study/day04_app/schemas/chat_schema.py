"""
聊天接口请求体和响应体

类似 Java 项目里的 DTO / VO。
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入的问题")


class ChatResponse(BaseModel):
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
