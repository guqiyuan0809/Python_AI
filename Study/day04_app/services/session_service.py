"""
会话与消息服务层

当前版本使用 MySQL 保存会话和消息。
"""

from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import ChatMessage, ChatSession, ChatSessionSummary
from settings import settings


SYSTEM_PROMPT = "你是一个专业、简洁的 Python AI 应用开发老师。"
SUMMARY_PROMPT_PREFIX = "以下是本会话早期重要信息摘要，请在回答时作为背景参考："
SUMMARY_REFRESH_MESSAGE_THRESHOLD = 8
SUMMARY_REFRESH_TOKEN_THRESHOLD = 1200
CHARS_PER_TOKEN_ESTIMATE = 4


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


def update_session_summary(db: Session, session_id: str, summary: str) -> ChatSession:
    chat_session = get_session(db, session_id)
    chat_session.summary = summary
    db.commit()
    db.refresh(chat_session)
    return chat_session


def list_sessions(
    db: Session,
    page: int = 1,
    page_size: int = 10,
    user_id: str | None = None,
) -> tuple[list[ChatSession], int]:
    # 先构造基础查询条件，后续可以很方便地扩展 user_id、状态、关键字等筛选条件。
    filters = [ChatSession.status == "active"]
    if user_id:
        filters.append(ChatSession.user_id == user_id)

    # count 查询用于告诉前端总共有多少条会话，方便前端渲染分页器。
    total_statement = select(func.count()).select_from(ChatSession).where(*filters)
    total = db.scalar(total_statement) or 0

    # offset 表示跳过多少条，limit 表示本页取多少条。
    statement = (
        select(ChatSession)
        .where(*filters)
        .order_by(ChatSession.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total


def get_latest_session_summary(
    db: Session, session_id: str
) -> ChatSessionSummary | None:
    # 查询当前会话最新的一条成功摘要，后续会作为长期记忆使用。
    statement = (
        select(ChatSessionSummary)
        .where(
            ChatSessionSummary.session_id == session_id,
            ChatSessionSummary.status == "success",
        )
        .order_by(ChatSessionSummary.version.desc())
        .limit(1)
    )
    return db.scalars(statement).first()


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


def get_session_messages_page(
    db: Session,
    session_id: str,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[ChatMessage], int]:
    get_session(db, session_id)

    # 先统计当前会话消息总数，避免前端不知道还有没有下一页。
    total_statement = (
        select(func.count())
        .select_from(ChatMessage)
        .where(ChatMessage.session_id == session_id)
    )
    total = db.scalar(total_statement) or 0

    # 按 id 升序查询，保证消息展示顺序是从早到晚。
    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return list(db.scalars(statement).all()), total


def get_success_dialog_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    # 过滤 pending/error，只保留真正可以进入模型上下文的成功对话。
    return [
        message
        for message in messages
        if message.status == "success" and message.role in {"user", "assistant"}
    ]


def build_simple_summary(messages: list[ChatMessage], max_items: int = 12) -> str:
    # 先筛出成功的用户和 AI 对话，失败消息不参与摘要。
    important_messages = get_success_dialog_messages(messages)
    if not important_messages:
        return ""

    summary_lines = []
    # 只取最近 max_items 条，避免规则摘要本身过长。
    for message in important_messages[-max_items:]:
        role_name = "用户" if message.role == "user" else "AI"
        content = message.content.replace("\n", " ").strip()
        # 单条内容过长时截断，防止摘要字符串无限膨胀。
        if len(content) > 120:
            content = content[:120] + "..."
        summary_lines.append(f"{role_name}: {content}")

    return "；".join(summary_lines)


def get_messages_after_summary(
    db: Session,
    session_id: str,
    latest_summary: ChatSessionSummary | None,
) -> list[ChatMessage]:
    # 没有历史摘要时，说明还没有压缩边界，直接返回本会话全部成功对话。
    if latest_summary is None or latest_summary.summary_until_message_id is None:
        return get_success_dialog_messages(get_session_messages(db, session_id))

    # 找到“最新摘要已经覆盖到的那条消息”，它就是后续增量查询的边界。
    boundary_statement = select(ChatMessage).where(
        ChatMessage.message_id == latest_summary.summary_until_message_id
    )
    boundary_message = db.scalars(boundary_statement).first()
    if boundary_message is None:
        return get_success_dialog_messages(get_session_messages(db, session_id))

    # 只查询摘要覆盖边界之后的新消息，避免把已经被摘要压缩过的历史重复发送给模型。
    statement = (
        select(ChatMessage)
        .where(
            ChatMessage.session_id == session_id,
            ChatMessage.id > boundary_message.id,
            ChatMessage.status == "success",
            ChatMessage.role.in_(["user", "assistant"]),
        )
        .order_by(ChatMessage.id.asc())
    )
    return list(db.scalars(statement).all())


def get_next_summary_version(db: Session, session_id: str) -> int:
    # 摘要版本号按会话递增：没有历史摘要就是第 1 版。
    latest_summary = get_latest_session_summary(db, session_id)
    if latest_summary is None:
        return 1
    return latest_summary.version + 1


def create_session_summary(
    db: Session,
    session_id: str,
    summary: str,
    summary_until_message_id: str | None,
    status: str = "success",
    error_message: str | None = None,
) -> ChatSessionSummary:
    # 每次刷新摘要都新增一条版本记录，不覆盖旧记录，方便后续审计和回滚。
    summary_record = ChatSessionSummary(
        summary_id=uuid4().hex,
        session_id=session_id,
        summary=summary,
        summary_until_message_id=summary_until_message_id,
        version=get_next_summary_version(db, session_id),
        model=settings.dashscope_model,
        status=status,
        error_message=error_message,
    )
    db.add(summary_record)
    db.commit()
    db.refresh(summary_record)
    return summary_record


def refresh_session_summary(db: Session, session_id: str) -> ChatSessionSummary:
    # 先查询当前会话最新摘要，用它判断本次只需要压缩哪些新增消息。
    latest_summary = get_latest_session_summary(db, session_id)
    new_messages = get_messages_after_summary(db, session_id, latest_summary)
    all_messages = get_session_messages(db, session_id)

    # 优先摘要“最新摘要之后的新消息”；如果没有新消息，就退回到全部消息。
    messages_for_summary = new_messages if new_messages else all_messages

    # 先生成一份规则版摘要，作为模型摘要失败时的兜底结果。
    fallback_summary = build_simple_summary(messages_for_summary)

    # 记录这份摘要覆盖到哪条消息，后续构造上下文时只取它之后的新消息。
    summary_until_message_id = (
        messages_for_summary[-1].message_id if messages_for_summary else None
    )

    # 如果没有可摘要内容，也写入一条空摘要版本，保持接口返回结构稳定。
    if not fallback_summary:
        update_session_summary(db, session_id, "")
        return create_session_summary(
            db,
            session_id=session_id,
            summary="",
            summary_until_message_id=summary_until_message_id,
        )

    try:
        # 在函数内部导入，避免 chat_service 和 session_service 互相导入造成循环依赖。
        from day04_app.services.chat_service import summarize_messages_with_model

        # 如果已有旧摘要，本次让模型把“旧摘要 + 新增内容”合并成新版摘要。
        if latest_summary and latest_summary.summary:
            summary_input = (
                f"已有会话摘要：{latest_summary.summary}\n"
                f"新增对话内容：{fallback_summary}"
            )
        else:
            summary_input = fallback_summary
        summary = summarize_messages_with_model(summary_input)
    except Exception:
        # 模型摘要失败时不影响主流程，使用规则摘要做降级。
        if latest_summary and latest_summary.summary:
            summary = f"{latest_summary.summary}；{fallback_summary}"
        else:
            summary = fallback_summary

    # 主表保存最新摘要，方便列表页或后续查询快速读取。
    update_session_summary(db, session_id, summary)

    # 摘要版本表保存每一次摘要记录，方便追踪摘要历史。
    return create_session_summary(
        db,
        session_id=session_id,
        summary=summary,
        summary_until_message_id=summary_until_message_id,
    )


def should_refresh_summary(db: Session, session_id: str) -> bool:
    # 只统计最新摘要之后的新消息数量，不用整个会话累计消息数触发刷新。
    latest_summary = get_latest_session_summary(db, session_id)
    new_messages = get_messages_after_summary(db, session_id, latest_summary)
    return len(new_messages) >= SUMMARY_REFRESH_MESSAGE_THRESHOLD


def estimate_text_tokens(text: str) -> int:
    # 学习阶段先用字符数粗略估算 token，后续企业级版本应替换为正式 tokenizer。
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    return sum(estimate_text_tokens(message.content) for message in messages)


def should_refresh_summary_by_token_budget(db: Session, session_id: str) -> bool:
    # 只估算新增消息的 token，避免已经被摘要覆盖的历史重复触发刷新。
    latest_summary = get_latest_session_summary(db, session_id)
    new_messages = get_messages_after_summary(db, session_id, latest_summary)
    return estimate_messages_tokens(new_messages) >= SUMMARY_REFRESH_TOKEN_THRESHOLD


def should_refresh_summary_for_session(db: Session, session_id: str) -> bool:
    return should_refresh_summary(db, session_id) or should_refresh_summary_by_token_budget(
        db, session_id
    )


def get_recent_messages(db: Session, session_id: str, limit: int = 6) -> list[ChatMessage]:
    get_session(db, session_id)
    statement = (
        select(ChatMessage)
        .where(
            ChatMessage.session_id == session_id,
            ChatMessage.status == "success",
            ChatMessage.role.in_(["user", "assistant"]),
        )
        .order_by(ChatMessage.id.desc())
        .limit(limit)
    )
    messages = list(db.scalars(statement).all())
    return list(reversed(messages))


def get_context_messages_after_summary(
    db: Session,
    session_id: str,
    latest_summary: ChatSessionSummary | None,
    limit: int = 6,
) -> list[ChatMessage]:
    # 没有摘要时，直接取最近 N 条成功消息作为短期上下文。
    if latest_summary is None:
        return get_recent_messages(db, session_id, limit=limit)

    # 有摘要时，只取摘要之后的新消息，避免和长期摘要重复。
    messages = get_messages_after_summary(db, session_id, latest_summary)
    return messages[-limit:] if limit > 0 else []


def build_messages(
    db: Session,
    session_id: str,
    current_question: str,
    history_limit: int = 6,
) -> list[dict]:
    chat_session = get_session(db, session_id)
    latest_summary = get_latest_session_summary(db, session_id)

    # 获取本次请求需要携带的短期历史消息。
    history = get_context_messages_after_summary(
        db,
        session_id=session_id,
        latest_summary=latest_summary,
        limit=history_limit,
    )

    # 第一条 system 消息用于定义模型角色和回答风格。
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 优先使用摘要版本表中的最新摘要；没有版本记录时，兼容读取会话主表 summary。
    summary = latest_summary.summary if latest_summary else chat_session.summary
    if summary:
        # 把长期摘要作为 system 背景信息交给模型。
        messages.append(
            {
                "role": "system",
                "content": f"{SUMMARY_PROMPT_PREFIX}\n{summary}",
            }
        )

    # 把短期历史消息转换成模型要求的 role/content 字典格式。
    for item in history:
        messages.append(
            {
                "role": item.role,
                "content": item.content,
            }
        )

    # 最后追加当前用户问题，模型会基于前面的角色、摘要和历史来回答它。
    messages.append({"role": "user", "content": current_question})
    return messages
