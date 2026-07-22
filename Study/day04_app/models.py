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


class AiCallLog(Base):
    __tablename__ = "ai_call_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 调用日志业务 ID，使用雪花 ID，适合后续跨服务追踪和对外查询。
    call_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # 一次请求链路 ID，Java 和 Python 之间通过 X-Trace-Id 透传。
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 会话 ID，便于从某个会话反查本次模型调用。
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 本次调用关联的 assistant 消息 ID，便于从日志定位到最终展示给用户的回答。
    message_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 调用来源，例如 session_chat、session_stream_chat、summary、title。
    call_type: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 调用耗时，后续可以用于慢调用分析和模型性能监控。
    cost_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="success")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AiAsyncTask(Base):
    __tablename__ = "ai_async_task"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 异步任务业务 ID，前端或 Java 后端后续用它查询任务状态。
    task_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # 请求链路 ID，用于把提交任务和后台执行日志串起来。
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 任务所属会话，当前先支持会话聊天异步执行。
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    # 任务最终关联的 assistant 消息 ID，任务完成后可定位到聊天记录。
    message_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # Celery 投递到 Broker 后产生的内部消息 ID，用于调度侧排查，不替代业务 task_id。
    broker_task_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 任务类型，例如 session_chat，后续可以扩展 summary、rag_import、report_generate。
    task_type: Mapped[str] = mapped_column(String(64), index=True)
    # 用户本次提交的问题，方便任务排查和失败重试。
    input_text: Mapped[str] = mapped_column(Text)
    # AI 最终输出内容，任务成功后写入。
    result_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # pending/running/success/error，用于前端轮询展示任务进度。
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # 自动重试次数和最大重试次数，避免模型异常时无限消耗资源。
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )


class AiTaskOutbox(Base):
    """任务投递事件表，用于解决 MySQL 与消息队列的双写一致性问题。"""

    __tablename__ = "ai_task_outbox"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 事件 ID 独立于 task_id，一次任务可以因自动重试产生多条投递事件。
    event_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    # JSON 字符串，保存 Worker 执行任务所需的最小参数。
    payload: Mapped[str] = mapped_column(Text)
    # pending/published，发布失败后保持 pending，由 Beat 定时补偿。
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    publish_retry_count: Mapped[int] = mapped_column(Integer, default=0)
    # 自动重试时延迟到指定时间后再投递，避免故障期间高频空转。
    available_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )


class AiStructuredResult(Base):
    """AI 结构化结果表，用于持久化模型生成的标准 JSON 业务结果。"""

    __tablename__ = "ai_structured_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 结构化结果业务 ID，前端/Java 可以通过它定位一份标准 JSON 结果。
    result_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # 异步任务 ID；同步测试接口可以为空，后续接入 Worker 后会关联 ai_async_task.task_id。
    task_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    message_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 业务归属，例如 work_order、contract、jingtangling_audit。
    business_type: Mapped[str] = mapped_column(String(64), index=True)
    business_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 结构化结果类型和版本，方便后续 prompt/DTO 升级时做兼容。
    schema_type: Mapped[str] = mapped_column(String(64), index=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="v1")
    # 保存标准 JSON 字符串；查询接口再反序列化成前端可直接消费的对象。
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="success", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )
