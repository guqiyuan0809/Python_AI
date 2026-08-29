"""Celery Beat 定时执行的补偿任务。"""

from day04_app.celery_app import celery_app
from day04_app.database import SessionLocal
from day04_app.services.async_task_service import mark_timeout_tasks_error
from day04_app.services.outbox_dispatcher import dispatch_pending_outbox_events
from day04_app.models import SessionMemory
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


@celery_app.task(name="day04_app.tasks.maintenance_tasks.retry_pending_session_memory_index")
def retry_pending_session_memory_index() -> dict:
    """补偿长期记忆向量化：只扫描 MySQL 已治理的 active memory。"""

    db = SessionLocal()
    try:
        records = list(
            db.query(SessionMemory)
            .filter(
                SessionMemory.status == "active",
                SessionMemory.embedding_status.in_(["pending", "error"]),
            )
            .order_by(SessionMemory.id.asc())
            .limit(20)
            .all()
        )
        dispatched = 0
        for record in records:
            celery_app.send_task(
                "day04_app.tasks.ai_tasks.index_session_memory",
                kwargs={"memory_id": record.memory_id},
            )
            record.embedding_status = "indexing"
            dispatched += 1
        if dispatched:
            db.commit()
        return {"dispatched": dispatched}
    finally:
        db.close()
