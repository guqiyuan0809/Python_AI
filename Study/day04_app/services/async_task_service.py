"""
AI 异步任务服务层

任务表负责业务状态，Outbox 表负责可靠投递到 Celery Broker。
"""

import json
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException, ERROR_TYPE_TASK_TIMEOUT
from day04_app.models import AiAsyncTask, AiTaskOutbox, ChatMessage
from day04_app.utils.snowflake_id import next_snowflake_id


TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"

OUTBOX_STATUS_PENDING = "pending"
OUTBOX_STATUS_PUBLISHED = "published"
OUTBOX_EVENT_SESSION_CHAT = "session_chat.execute"
OUTBOX_EVENT_SESSION_RAG = "session_rag.execute"
OUTBOX_EVENT_WORK_ORDER_ANALYSIS = "work_order_analysis.execute"
OUTBOX_EVENT_WORK_ORDER_EVAL = "work_order_eval.execute"
OUTBOX_EVENT_KNOWLEDGE_CONTEXTUAL_INDEX = "knowledge_contextual_index.execute"


def _build_session_chat_payload(task: AiAsyncTask, history_limit: int) -> str:
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


def _build_session_rag_payload(
    task: AiAsyncTask,
    document_id: str,
    retrieval_top_k: int,
    max_context_characters: int,
    use_reranker: bool,
    rerank_top_n: int,
    score_threshold: float | None,
) -> str:
    """持久化 RAG 检索参数快照，重试时不能依赖 HTTP 请求仍然存在。"""
    return json.dumps(
        {
            "task_id": task.task_id,
            "session_id": task.session_id,
            "message": task.input_text,
            "trace_id": task.trace_id,
            "document_id": document_id,
            "retrieval_top_k": retrieval_top_k,
            "max_context_characters": max_context_characters,
            "use_reranker": use_reranker,
            "rerank_top_n": rerank_top_n,
            "score_threshold": score_threshold,
        },
        ensure_ascii=False,
    )


def _build_work_order_analysis_payload(
    task: AiAsyncTask,
    business_id: str | None,
) -> str:
    return json.dumps(
        {
            "task_id": task.task_id,
            "session_id": task.session_id,
            "content": task.input_text,
            "trace_id": task.trace_id,
            "business_id": business_id,
        },
        ensure_ascii=False,
    )


def _build_work_order_eval_payload(
    task: AiAsyncTask,
    prompt_name: str,
    prompt_version: str,
    dataset_version: str,
) -> str:
    return json.dumps(
        {
            "task_id": task.task_id,
            "trace_id": task.trace_id,
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "dataset_version": dataset_version,
        },
        ensure_ascii=False,
    )


def _build_contextual_index_payload(
    task: AiAsyncTask,
    version_id: str,
    context_model: str | None,
    context_max_tokens: int,
) -> str:
    """保存知识索引后台任务的参数快照，Worker 重试不依赖原 HTTP 请求。"""
    return json.dumps(
        {
            "task_id": task.task_id,
            "version_id": version_id,
            "context_model": context_model,
            "context_max_tokens": context_max_tokens,
            "trace_id": task.trace_id,
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
        payload=_build_session_chat_payload(task, history_limit),
        status=OUTBOX_STATUS_PENDING,
        available_at=datetime.now() + timedelta(seconds=delay_seconds),
    )


def _create_session_rag_outbox_event(
    task: AiAsyncTask,
    document_id: str,
    retrieval_top_k: int,
    max_context_characters: int,
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    score_threshold: float | None = None,
    delay_seconds: int = 0,
) -> AiTaskOutbox:
    return AiTaskOutbox(
        event_id=next_snowflake_id(),
        task_id=task.task_id,
        event_type=OUTBOX_EVENT_SESSION_RAG,
        payload=_build_session_rag_payload(
            task,
            document_id=document_id,
            retrieval_top_k=retrieval_top_k,
            max_context_characters=max_context_characters,
            use_reranker=use_reranker,
            rerank_top_n=rerank_top_n,
            score_threshold=score_threshold,
        ),
        status=OUTBOX_STATUS_PENDING,
        available_at=datetime.now() + timedelta(seconds=delay_seconds),
    )


def _create_work_order_analysis_outbox_event(
    task: AiAsyncTask,
    business_id: str | None,
    delay_seconds: int = 0,
) -> AiTaskOutbox:
    return AiTaskOutbox(
        event_id=next_snowflake_id(),
        task_id=task.task_id,
        event_type=OUTBOX_EVENT_WORK_ORDER_ANALYSIS,
        payload=_build_work_order_analysis_payload(task, business_id),
        status=OUTBOX_STATUS_PENDING,
        available_at=datetime.now() + timedelta(seconds=delay_seconds),
    )


def _create_work_order_eval_outbox_event(
    task: AiAsyncTask,
    prompt_name: str,
    prompt_version: str,
    dataset_version: str,
    delay_seconds: int = 0,
) -> AiTaskOutbox:
    return AiTaskOutbox(
        event_id=next_snowflake_id(),
        task_id=task.task_id,
        event_type=OUTBOX_EVENT_WORK_ORDER_EVAL,
        payload=_build_work_order_eval_payload(
            task,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            dataset_version=dataset_version,
        ),
        status=OUTBOX_STATUS_PENDING,
        available_at=datetime.now() + timedelta(seconds=delay_seconds),
    )


def _create_contextual_index_outbox_event(
    task: AiAsyncTask,
    version_id: str,
    context_model: str | None,
    context_max_tokens: int,
    delay_seconds: int = 0,
) -> AiTaskOutbox:
    return AiTaskOutbox(
        event_id=next_snowflake_id(),
        task_id=task.task_id,
        event_type=OUTBOX_EVENT_KNOWLEDGE_CONTEXTUAL_INDEX,
        payload=_build_contextual_index_payload(
            task,
            version_id=version_id,
            context_model=context_model,
            context_max_tokens=context_max_tokens,
        ),
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


def create_async_session_rag_task(
    db: Session,
    session_id: str,
    input_text: str,
    trace_id: str | None,
    model: str | None,
    document_id: str,
    retrieval_top_k: int,
    max_context_characters: int,
    max_retries: int,
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    score_threshold: float | None = None,
) -> tuple[AiAsyncTask, AiTaskOutbox]:
    """一次事务提交用户消息、RAG 任务和 Outbox 投递事件。"""
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
        task_type="session_rag",
        input_text=input_text,
        model=model,
        max_retries=max_retries,
        status=TASK_STATUS_PENDING,
    )
    outbox_event = _create_session_rag_outbox_event(
        task,
        document_id=document_id,
        retrieval_top_k=retrieval_top_k,
        max_context_characters=max_context_characters,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
        score_threshold=score_threshold,
    )

    # 用户看到的问题、可轮询任务和待投递消息必须同时存在或同时回滚。
    db.add_all([user_message, task, outbox_event])
    db.commit()
    db.refresh(task)
    db.refresh(outbox_event)
    return task, outbox_event


def create_async_work_order_eval_task(
    db: Session,
    trace_id: str | None,
    prompt_name: str,
    prompt_version: str,
    dataset_version: str,
    max_retries: int,
) -> tuple[AiAsyncTask, AiTaskOutbox]:
    """创建后台评测任务；评测不属于某个用户会话，因此 session_id 使用 system_eval。"""
    input_text = json.dumps(
        {
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "dataset_version": dataset_version,
        },
        ensure_ascii=False,
    )
    task = AiAsyncTask(
        task_id=next_snowflake_id(),
        trace_id=trace_id,
        session_id="system_eval",
        task_type="work_order_eval",
        input_text=input_text,
        model=None,
        max_retries=max_retries,
        status=TASK_STATUS_PENDING,
    )
    outbox_event = _create_work_order_eval_outbox_event(
        task,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        dataset_version=dataset_version,
    )

    # 评测任务和 Outbox 在同一事务提交，避免出现有任务但没有 MQ 投递事件。
    db.add_all([task, outbox_event])
    db.commit()
    db.refresh(task)
    db.refresh(outbox_event)
    return task, outbox_event


def create_async_contextual_index_task(
    db: Session,
    *,
    version_id: str,
    trace_id: str | None,
    context_model: str | None,
    context_max_tokens: int,
    max_retries: int,
) -> tuple[AiAsyncTask, AiTaskOutbox]:
    """创建知识库上下文化索引任务；它不是用户会话，因此使用 system_knowledge 会话标识。"""
    input_text = json.dumps(
        {
            "version_id": version_id,
            "context_model": context_model,
            "context_max_tokens": context_max_tokens,
        },
        ensure_ascii=False,
    )
    task = AiAsyncTask(
        task_id=next_snowflake_id(),
        trace_id=trace_id,
        session_id="system_knowledge",
        task_type="knowledge_contextual_index",
        input_text=input_text,
        model=context_model,
        max_retries=max_retries,
        status=TASK_STATUS_PENDING,
    )
    outbox_event = _create_contextual_index_outbox_event(
        task,
        version_id=version_id,
        context_model=context_model,
        context_max_tokens=context_max_tokens,
    )
    db.add_all([task, outbox_event])
    db.commit()
    db.refresh(task)
    db.refresh(outbox_event)
    return task, outbox_event


def create_async_work_order_analysis_task(
    db: Session,
    session_id: str,
    content: str,
    trace_id: str | None,
    model: str | None,
    business_id: str | None,
    max_retries: int,
) -> tuple[AiAsyncTask, AiTaskOutbox]:
    """一次事务写入用户消息、结构化分析任务和 Outbox 事件。"""
    user_message = ChatMessage(
        message_id=uuid4().hex,
        session_id=session_id,
        trace_id=trace_id,
        role="user",
        content=content,
        status="success",
    )
    task = AiAsyncTask(
        task_id=next_snowflake_id(),
        trace_id=trace_id,
        session_id=session_id,
        task_type="work_order_analysis",
        input_text=content,
        model=model,
        max_retries=max_retries,
        status=TASK_STATUS_PENDING,
    )
    outbox_event = _create_work_order_analysis_outbox_event(task, business_id)

    # 结构化分析也走本地消息表，保证任务记录和 MQ 投递事件在同一个数据库事务中提交。
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
    commit: bool = True,
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
    task.error_type = None
    task.error_message = None
    if commit:
        db.commit()
        db.refresh(task)
    return task


def mark_task_error(
    db: Session,
    task_id: str,
    error_message: str,
    cost_ms: int | None,
    error_type: str | None = None,
) -> AiAsyncTask | None:
    task = get_async_task(db, task_id)
    if task.status != TASK_STATUS_RUNNING:
        return None
    task.status = TASK_STATUS_ERROR
    task.error_type = error_type
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
    task.error_type = None
    task.error_message = None
    task.retry_count += 1
    outbox_event = _create_outbox_event(task, history_limit, delay_seconds)

    db.add(outbox_event)
    db.commit()
    db.refresh(task)
    db.refresh(outbox_event)
    return task, outbox_event


def prepare_session_rag_task_retry(
    db: Session,
    task_id: str,
    document_id: str,
    retrieval_top_k: int,
    max_context_characters: int,
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    score_threshold: float | None = None,
    delay_seconds: int = 0,
) -> tuple[AiAsyncTask, AiTaskOutbox]:
    """重试 RAG 时复用初次提交的检索参数，并生成新的延迟 Outbox 事件。"""
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
    task.error_type = None
    task.error_message = None
    task.retry_count += 1
    outbox_event = _create_session_rag_outbox_event(
        task,
        document_id=document_id,
        retrieval_top_k=retrieval_top_k,
        max_context_characters=max_context_characters,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
        score_threshold=score_threshold,
        delay_seconds=delay_seconds,
    )

    db.add(outbox_event)
    db.commit()
    db.refresh(task)
    db.refresh(outbox_event)
    return task, outbox_event


def get_session_rag_retry_parameters(
    db: Session,
    task_id: str,
) -> tuple[str, int, int, bool, int, float | None]:
    """从已发布的 Outbox 快照恢复 RAG 重试参数，避免人工重试误投递为普通聊天任务。"""
    latest_event = db.scalars(
        select(AiTaskOutbox)
        .where(
            AiTaskOutbox.task_id == task_id,
            AiTaskOutbox.event_type == OUTBOX_EVENT_SESSION_RAG,
        )
        .order_by(AiTaskOutbox.id.desc())
        .limit(1)
    ).first()
    if latest_event is None:
        raise BusinessException(code=50058, message="RAG 任务缺少 Outbox 参数快照，不能重试")
    try:
        payload = json.loads(latest_event.payload)
        document_id = str(payload["document_id"])
        retrieval_top_k = int(payload["retrieval_top_k"])
        max_context_characters = int(payload["max_context_characters"])
        use_reranker = bool(payload.get("use_reranker", False))
        rerank_top_n = int(payload.get("rerank_top_n", 20))
        raw_score_threshold = payload.get("score_threshold")
        score_threshold = (
            float(raw_score_threshold) if raw_score_threshold is not None else None
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BusinessException(code=50058, message="RAG 任务 Outbox 参数快照损坏，不能重试") from exc
    return (
        document_id,
        retrieval_top_k,
        max_context_characters,
        use_reranker,
        rerank_top_n,
        score_threshold,
    )


def get_contextual_index_retry_parameters(
    db: Session,
    task_id: str,
) -> tuple[str, str | None, int]:
    """从最近一次上下文化 Outbox 快照恢复索引任务参数。"""
    latest_event = db.scalars(
        select(AiTaskOutbox)
        .where(
            AiTaskOutbox.task_id == task_id,
            AiTaskOutbox.event_type == OUTBOX_EVENT_KNOWLEDGE_CONTEXTUAL_INDEX,
        )
        .order_by(AiTaskOutbox.id.desc())
        .limit(1)
    ).first()
    if latest_event is None:
        raise BusinessException(code=50063, message="上下文化索引任务缺少 Outbox 参数快照，不能重试")
    try:
        payload = json.loads(latest_event.payload)
        version_id = str(payload["version_id"])
        context_model = payload.get("context_model")
        context_max_tokens = int(payload.get("context_max_tokens", 180))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BusinessException(code=50063, message="上下文化索引 Outbox 参数快照损坏，不能重试") from exc
    return version_id, context_model, context_max_tokens


def prepare_work_order_analysis_task_retry(
    db: Session,
    task_id: str,
    business_id: str | None,
    delay_seconds: int = 0,
) -> tuple[AiAsyncTask, AiTaskOutbox]:
    """重试结构化分析任务，并新建对应的 Outbox 事件。"""
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
    task.error_type = None
    task.error_message = None
    task.retry_count += 1
    outbox_event = _create_work_order_analysis_outbox_event(task, business_id, delay_seconds)

    db.add(outbox_event)
    db.commit()
    db.refresh(task)
    db.refresh(outbox_event)
    return task, outbox_event


def prepare_work_order_eval_task_retry(
    db: Session,
    task_id: str,
    prompt_name: str,
    prompt_version: str,
    dataset_version: str,
    delay_seconds: int = 0,
) -> tuple[AiAsyncTask, AiTaskOutbox]:
    """重试评测任务，并新建对应的 Outbox 事件。"""
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
    task.error_type = None
    task.error_message = None
    task.retry_count += 1
    outbox_event = _create_work_order_eval_outbox_event(
        task,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        dataset_version=dataset_version,
        delay_seconds=delay_seconds,
    )

    db.add(outbox_event)
    db.commit()
    db.refresh(task)
    db.refresh(outbox_event)
    return task, outbox_event


def prepare_contextual_index_task_retry(
    db: Session,
    task_id: str,
    version_id: str,
    context_model: str | None,
    context_max_tokens: int,
    delay_seconds: int = 0,
) -> tuple[AiAsyncTask, AiTaskOutbox]:
    """重试知识库上下文化索引任务，并保留其专用 Outbox 事件类型。"""
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
    task.error_type = None
    task.error_message = None
    task.retry_count += 1
    outbox_event = _create_contextual_index_outbox_event(
        task,
        version_id=version_id,
        context_model=context_model,
        context_max_tokens=context_max_tokens,
        delay_seconds=delay_seconds,
    )
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
        task.error_type = ERROR_TYPE_TASK_TIMEOUT
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
                    error_type=ERROR_TYPE_TASK_TIMEOUT,
                    error_message=error_message,
                )
            )
    db.commit()
    for task in timeout_tasks:
        db.refresh(task)
    return timeout_tasks
