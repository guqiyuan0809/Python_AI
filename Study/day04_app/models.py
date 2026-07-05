"""
数据库表模型

类似 Java 项目中的 Entity。
"""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from day04_app.database import Base


class ChatSession(Base):
    __tablename__ = "chat_session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )


class ChatMessage(Base):
    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stream_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="success")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ChatSessionSummary(Base):
    __tablename__ = "chat_session_summary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 摘要记录的业务唯一 ID，接口返回时优先使用它，而不是数据库自增 id。
    summary_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # 这条摘要属于哪个会话，用于关联 chat_session.session_id。
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    # 压缩后的会话摘要内容，会作为后续模型调用的长期上下文。
    summary: Mapped[str] = mapped_column(Text)
    # 表示这份摘要已经覆盖到了哪一条消息，后续只需要再携带这条消息之后的新消息。
    summary_until_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 同一个会话下的摘要版本号，越大表示越新的摘要。
    version: Mapped[int] = mapped_column(Integer, default=1)
    # 生成摘要时使用的模型，便于后续排查摘要质量和成本。
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 摘要生成状态，当前主要使用 success，后续可扩展 pending/error。
    status: Mapped[str] = mapped_column(String(32), default="success")
    # 摘要失败时记录错误信息，方便排查，不直接返回给普通用户。
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
