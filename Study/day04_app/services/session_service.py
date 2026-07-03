"""
会话与消息服务层

当前版本使用 MySQL 保存会话和消息。
"""

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import ChatMessage, ChatSession


SYSTEM_PROMPT = "你是一个专业、简洁的 Python AI 应用开发老师。"


def create_session(db: Session, user_id: str | None = None, title: str | None = None) -> str:
    session_id = uuid4().hex
    chat_session = ChatSession(
        session_id=session_id,
        user_id=user_id,
        title=title,
    )
    db.add(chat_session)
    db.commit()
    return session_id


def get_session(db: Session, session_id: str) -> ChatSession:
    statement = select(ChatSession).where(ChatSession.session_id == session_id)
    chat_session = db.scalars(statement).first()
    if chat_session is None:
        raise BusinessException(code=40004, message="会话不存在")
    return chat_session


def add_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    trace_id: str | None = None,
    stream_id: str | None = None,
    model: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    status: str = "success",
    error_message: str | None = None,
) -> ChatMessage:
    get_session(db, session_id)

    message = ChatMessage(
        message_id=uuid4().hex,
        session_id=session_id,
        trace_id=trace_id,
        stream_id=stream_id,
        role=role,
        content=content,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        status=status,
        error_message=error_message,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_message(db: Session, message_id: str) -> ChatMessage:
    statement = select(ChatMessage).where(ChatMessage.message_id == message_id)
    message = db.scalars(statement).first()
    if message is None:
        raise BusinessException(code=40005, message="消息不存在")
    return message


def update_message(
    db: Session,
    message_id: str,
    content: str | None = None,
    status: str | None = None,
    error_message: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> ChatMessage:
    message = get_message(db, message_id)

    if content is not None:
        message.content = content
    if status is not None:
        message.status = status
    if error_message is not None:
        message.error_message = error_message
    if prompt_tokens is not None:
        message.prompt_tokens = prompt_tokens
    if completion_tokens is not None:
        message.completion_tokens = completion_tokens
    if total_tokens is not None:
        message.total_tokens = total_tokens

    db.commit()
    db.refresh(message)
    return message


def get_session_messages(db: Session, session_id: str) -> list[ChatMessage]:
    get_session(db, session_id)
    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
    )
    return list(db.scalars(statement).all())


def get_recent_messages(db: Session, session_id: str, limit: int = 6) -> list[ChatMessage]:
    get_session(db, session_id)
    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
    )
    messages = list(db.scalars(statement).all())
    return list(reversed(messages))


def build_messages(
    db: Session,
    session_id: str,
    current_question: str,
    history_limit: int = 6,
) -> list[dict]:
    history = get_recent_messages(db, session_id, limit=history_limit)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in history:
        messages.append(
            {
                "role": item.role,
                "content": item.content,
            }
        )

    messages.append({"role": "user", "content": current_question})
    return messages
