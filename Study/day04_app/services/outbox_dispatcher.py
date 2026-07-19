"""Outbox 发布器：负责把已经提交到 MySQL 的任务事件投递到 Celery Broker。"""

import json
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from day04_app.celery_app import celery_app
from day04_app.services.async_task_service import (
    OUTBOX_STATUS_PUBLISHED,
    get_outbox_event,
    list_pending_outbox_events,
    mark_outbox_event_publish_failed,
    mark_outbox_event_published,
)


logger = logging.getLogger("day04_app.outbox")
SESSION_CHAT_TASK_NAME = "day04_app.tasks.ai_tasks.execute_session_chat_task"


def dispatch_outbox_event(db: Session, event_id: str) -> bool:
    """发布一条 Outbox 事件。Broker 接收成功后才把事件改为 published。"""
    event = get_outbox_event(db, event_id)
    if event.status == OUTBOX_STATUS_PUBLISHED:
        return True
    if event.available_at > datetime.now():
        # 自动重试尚未到达退避时间，等待 Beat 后续扫描再投递。
        return True

    try:
        payload = json.loads(event.payload)
        result = celery_app.send_task(SESSION_CHAT_TASK_NAME, kwargs=payload)
        mark_outbox_event_published(db, event_id, result.id)
        return True
    except Exception as exc:
        error_message = f"任务投递到 Celery Broker 失败：{type(exc).__name__}"
        mark_outbox_event_publish_failed(db, event_id, error_message)
        logger.exception("event_id=%s %s", event_id, error_message)
        return False


def dispatch_pending_outbox_events(db: Session, limit: int = 100) -> tuple[int, int]:
    """供 Celery Beat 定期调用，补发上次未成功投递的事件。"""
    events = list_pending_outbox_events(db, limit=limit)
    published_count = 0
    failed_count = 0
    for event in events:
        if dispatch_outbox_event(db, event.event_id):
            published_count += 1
        else:
            failed_count += 1
    return published_count, failed_count
