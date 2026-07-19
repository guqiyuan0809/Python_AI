"""Celery Beat 定时执行的补偿任务。"""

from day04_app.celery_app import celery_app
from day04_app.database import SessionLocal
from day04_app.services.async_task_service import mark_timeout_tasks_error
from day04_app.services.outbox_dispatcher import dispatch_pending_outbox_events
from settings import settings


@celery_app.task(name="day04_app.tasks.maintenance_tasks.dispatch_pending_task_outbox")
def dispatch_pending_task_outbox() -> dict:
    db = SessionLocal()
    try:
        published_count, failed_count = dispatch_pending_outbox_events(db)
        return {"published_count": published_count, "failed_count": failed_count}
    finally:
        db.close()


@celery_app.task(name="day04_app.tasks.maintenance_tasks.scan_timeout_ai_tasks")
def scan_timeout_ai_tasks() -> dict:
    db = SessionLocal()
    try:
        tasks = mark_timeout_tasks_error(
            db,
            timeout_minutes=settings.async_task_timeout_minutes,
        )
        return {"timeout_count": len(tasks)}
    finally:
        db.close()
