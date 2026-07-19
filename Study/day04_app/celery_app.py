"""Celery 应用配置，类似 Java 中 MQ Consumer / Scheduler 的基础配置。"""

from celery import Celery

from settings import settings


celery_app = Celery(
    "day04_ai_service",
    broker=settings.celery_broker,
    include=[
        "day04_app.tasks.ai_tasks",
        "day04_app.tasks.maintenance_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # 业务结果已经写入 MySQL 的 ai_async_task/chat_message，不再使用 Celery 自带结果后端。
    task_ignore_result=True,
    timezone="Asia/Shanghai",
    enable_utc=False,
    task_track_started=True,
    # Worker 真正完成任务后才确认消息；进程异常退出时 Broker 可以重新投递。
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # AI 调用耗时长，一个 Worker 预取过多任务会造成排队不均衡。
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    beat_schedule={
        "dispatch-pending-ai-task-outbox": {
            "task": "day04_app.tasks.maintenance_tasks.dispatch_pending_task_outbox",
            "schedule": 30.0,
        },
        "scan-timeout-ai-tasks": {
            "task": "day04_app.tasks.maintenance_tasks.scan_timeout_ai_tasks",
            "schedule": 60.0,
        },
    },
)
