"""
AI 异步任务服务层

任务表负责业务状态，Outbox 表负责可靠投递到 Celery Broker。
"""

import json
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import AiAsyncTask, AiTaskOutbox, ChatMessage
from day04_app.utils.snowflake_id import next_snowflake_id


TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"

OUTBOX_STATUS_PENDING = "pending"
OUTBOX_STATUS_PUBLISHED = "published"
OUTBOX_EVENT_SESSION_CHAT = "session_chat.execute"


def _build_task_payload(task: AiAsyncTask, history_limit: int) -> str:
    """Outbox 中只存 Worker 执行所需的最小参数，不直接存 ORM 对象。"""
    return json.dumps(
        {
            "task_id": task.task_id,
            "session_id": task.session_id,
            "message": task.input_text,
            "trace_id": task.trace_id,
            "history_limit": history_limit,
        },
        ensure_ascii=False,
    )


def _create_outbox_event(
    task: AiAsyncTask,
    history_limit: int,
    delay_seconds: int = 0,
) -> AiTaskOutbox:
    return AiTaskOutbox(
        event_id=next_snowflake_id(),
        task_id=task.task_id,
        event_type=OUTBOX_EVENT_SESSION_CHAT,
        payload=_build_task_payload(task, history_limit),
        status=OUTBOX_STATUS_PENDING,
        available_at=datetime.now() + timedelta(seconds=delay_seconds),
    )


def create_async_session_chat_task(
    db: Session,
    session_id: str,
    input_text: str,
    trace_id: str | None,
    model: str | None,
    history_limit: int,
    max_retries: int,
) -> tuple[AiAsyncTask, AiTaskOutbox]:
    """一次事务写入用户消息、任务记录和 Outbox 件事。"""
    user_message = ChatMessage(
        message_id=uuid4().hex,
        session_id=session_id,
        trace_id=trace_id,
        role="user",
        content=input_text,
        status="success",
    )
    task = AiAsyncTask(
        task_id=next_snowflake_id(),
        trace_id=trace_id,
        session_id=session_id,
        task_type="session_chat",
        input_text=input_text,
        model=model,
        max_retries=max_retries,
        status=TASK_STATUS_PENDING,
    )
    outbox_event = _create_outbox_event(task, history_limit)

    # 三类业务数据一起提交，避免出现“有任务但没有用户问题”的不完整记录。
    db.add_all([user_message, task, outbox_event])
    db.commit()
    db.refresh(task)
    db.refresh(outbox_event)
    return task, outbox_event


def get_async_task(db: Session, task_id: str) -> AiAsyncTask:
    statement = select(AiAsyncTask).where(AiAsyncTask.task_id == task_id)
    task = db.scalars(statement).first()
    if task is None:
        raise BusinessException(code=40007, message="异步任务不存在")
    return task


def get_outbox_event(db: Session, event_id: str) -> AiTaskOutbox:
    statement = select(AiTaskOutbox).where(AiTaskOutbox.event_id == event_id)
    event = db.scalars(statement).first()
    if event is None:
        raise BusinessException(code=40010, message="任务投递事件不存在")
    return event


def list_pending_outbox_events(db: Session, limit: int = 100) -> list[AiTaskOutbox]:
    statement = (
        select(AiTaskOutbox)
        .where(
            AiTaskOutbox.status == OUTBOX_STATUS_PENDING,
            AiTaskOutbox.available_at <= datetime.now(),
        )
        .order_by(AiTaskOutbox.id.asc())
        .limit(limit)
    )
    return list(db.scalars(statement).all())


def mark_outbox_event_published(
    db: Session,
    event_id: str,
    broker_task_id: str,
) -> AiTaskOutbox:
    event = get_outbox_event(db, event_id)
    if event.status == OUTBOX_STATUS_PUBLISHED:
        return event

    event.status = OUTBOX_STATUS_PUBLISHED
    event.error_message = None
    event.published_at = datetime.now()
    task = get_async_task(db, event.task_id)
    task.broker_task_id = broker_task_id
    db.commit()
    db.refresh(event)
    return event


def mark_outbox_event_publish_failed(
    db: Session,
    event_id: str,
    error_message: str,
) -> AiTaskOutbox:
    event = get_outbox_event(db, event_id)
    # 发布失败不改成终态，Beat 下次仍会扫描并补发。
    event.status = OUTBOX_STATUS_PENDING
    event.publish_retry_count += 1
    event.error_message = error_message
    db.commit()
    db.refresh(event)
    return event


def claim_pending_task_for_execution(db: Session, task_id: str) -> AiAsyncTask | None:
    """原子领取 pending 任务，保证重复消息最多只有一个 Worker 真正执行。"""
    statement = (
        update(AiAsyncTask)
        .where(
            AiAsyncTask.task_id == task_id,
            AiAsyncTask.status == TASK_STATUS_PENDING,
        )
        .values(
            status=TASK_STATUS_RUNNING,
            error_message=None,
            updated_at=datetime.now(),
        )
    )
    result = db.execute(statement)
    db.commit()
    if result.rowcount != 1:
        return None
    return get_async_task(db, task_id)


def bind_task_message(
    db: Session,
    task_id: str,
    message_id: str,
) -> AiAsyncTask | None:
    task = get_async_task(db, task_id)
    if task.status != TASK_STATUS_RUNNING:
        return None
    task.message_id = message_id
    db.commit()
    db.refresh(task)
    return task


def mark_task_success(
    db: Session,
    task_id: str,
    result_text: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cost_ms: int | None,
) -> AiAsyncTask | None:
    task = get_async_task(db, task_id)
    # 如果超时扫描已把任务改成 error，晚到的 Worker 结果不能覆盖终态。
    if task.status != TASK_STATUS_RUNNING:
        return None
    task.status = TASK_STATUS_SUCCESS
    task.result_text = result_text
    task.prompt_tokens = prompt_tokens
    task.completion_tokens = completion_tokens
    task.total_tokens = total_tokens
    task.cost_ms = cost_ms
    db.commit()
    db.refresh(task)
    return task


def mark_task_error(
    db: Session,
    task_id: str,
    error_message: str,
    cost_ms: int | None,
) -> AiAsyncTask | None:
    task = get_async_task(db, task_id)
    if task.status != TASK_STATUS_RUNNING:
        return None
    task.status = TASK_STATUS_ERROR
    task.error_message = error_message
    task.cost_ms = cost_ms
    db.commit()
    db.refresh(task)
    return task


def prepare_task_retry(
    db: Session,
    task_id: str,
    history_limit: int,
    delay_seconds: int = 0,
) -> tuple[AiAsyncTask, AiTaskOutbox]:
    """把失败任务重置为 pending，并在同一事务中新建一次可靠投递事件。"""
    task = get_async_task(db, task_id)
    if task.status != TASK_STATUS_ERROR:
        raise BusinessException(code=40008, message="只有失败任务可以重试")
    if task.retry_count >= task.max_retries:
        raise BusinessException(code=40009, message="任务已达到最大重试次数，请人工处理")

    task.status = TASK_STATUS_PENDING
    task.message_id = None
    task.broker_task_id = None
    task.result_text = None
    task.prompt_tokens = None
    task.completion_tokens = None
    task.total_tokens = None
    task.cost_ms = None
    task.error_message = None
    task.retry_count += 1
    outbox_event = _create_outbox_event(task, history_limit, delay_seconds)

    db.add(outbox_event)
    db.commit()
    db.refresh(task)
    db.refresh(outbox_event)
    return task, outbox_event


def find_timeout_tasks(
    db: Session,
    timeout_minutes: int = 10,
) -> list[AiAsyncTask]:
    timeout_before = datetime.now() - timedelta(minutes=timeout_minutes)
    statement = (
        select(AiAsyncTask)
        .where(
            AiAsyncTask.status.in_([TASK_STATUS_PENDING, TASK_STATUS_RUNNING]),
            AiAsyncTask.updated_at < timeout_before,
        )
        .order_by(AiAsyncTask.id.asc())
    )
    return list(db.scalars(statement).all())


def mark_timeout_tasks_error(
    db: Session,
    timeout_minutes: int = 10,
) -> list[AiAsyncTask]:
    timeout_tasks = find_timeout_tasks(db, timeout_minutes=timeout_minutes)
    for task in timeout_tasks:
        error_message = f"异步任务超过 {timeout_minutes} 分钟未完成，已标记为超时失败"
        task.status = TASK_STATUS_ERROR
        task.error_message = error_message
        # 已创建 assistant 占位消息时，同步更新聊天历史，避免前端长期显示“生成中”。
        if task.message_id:
            db.execute(
                update(ChatMessage)
                .where(
                    ChatMessage.message_id == task.message_id,
                    ChatMessage.status.in_(["pending", "streaming"]),
                )
                .values(
                    content=error_message,
                    status="error",
                    error_message=error_message,
                )
            )
    db.commit()
    for task in timeout_tasks:
        db.refresh(task)
    return timeout_tasks
