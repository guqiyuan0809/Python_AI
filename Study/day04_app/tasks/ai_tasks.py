"""Celery Worker 中实际执行 AI 异步任务的代码。"""

import logging
import time

from day04_app.celery_app import celery_app
from day04_app.common.exceptions import (
    ERROR_TYPE_WORKER_EXECUTION_ERROR,
    BusinessException,
    ModelCallException,
)
from day04_app.database import SessionLocal
from day04_app.services.async_task_service import (
    bind_task_message,
    claim_pending_task_for_execution,
    mark_task_error,
    mark_task_success,
    prepare_session_rag_task_retry,
    prepare_work_order_eval_task_retry,
    prepare_work_order_analysis_task_retry,
    prepare_task_retry,
)
from day04_app.services.call_log_service import create_call_log
from day04_app.services.chat_service import analyze_work_order_structured, safe_chat_with_messages
from day04_app.services.outbox_dispatcher import dispatch_outbox_event
from day04_app.services.rag_context_service import generate_rag_answer
from day04_app.services.session_rag_service import (
    add_rag_answer_reference_records,
    prepare_session_rag_context,
)
from day04_app.services.session_service import (
    add_message,
    build_messages,
    get_message,
    refresh_session_summary,
    should_refresh_summary_for_session,
    update_message,
)
from day04_app.services.structured_result_service import create_structured_result
from day04_app.services.eval_result_service import save_eval_report
from day04_app.services.work_order_eval_runner import run_work_order_eval
from settings import settings


logger = logging.getLogger("day04_app.worker")


@celery_app.task(name="day04_app.tasks.ai_tasks.execute_session_rag_task")
def execute_session_rag_task(
    task_id: str,
    session_id: str,
    message: str,
    trace_id: str | None,
    document_id: str,
    retrieval_top_k: int,
    max_context_characters: int,
) -> dict:
    """异步会话 RAG：检索、回答、引用快照和任务终态在 Worker 内完成。"""
    db = SessionLocal()
    assistant_message_id: str | None = None
    start_time: float | None = None
    try:
        task = claim_pending_task_for_execution(db, task_id)
        if task is None:
            return {"task_id": task_id, "status": "ignored"}

        assistant_message = add_message(
            db,
            session_id,
            "assistant",
            "知识库回答生成中",
            trace_id=trace_id,
            model=settings.dashscope_model,
            status="pending",
        )
        assistant_message_id = assistant_message.message_id
        if bind_task_message(db, task_id, assistant_message_id) is None:
            return {"task_id": task_id, "status": "ignored"}

        start_time = time.perf_counter()
        retrieval_context = prepare_session_rag_context(
            db,
            document_id=document_id,
            message=message,
            retrieval_top_k=retrieval_top_k,
            max_context_characters=max_context_characters,
        )
        generation = generate_rag_answer(
            question=message,
            context_result=retrieval_context.context_result,
        )
        cost_ms = round((time.perf_counter() - start_time) * 1000)

        completed_task = mark_task_success(
            db,
            task_id=task_id,
            result_text=generation.answer,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=generation.total_tokens,
            cost_ms=cost_ms,
            commit=False,
        )
        if completed_task is None:
            return {"task_id": task_id, "status": "ignored"}

        # 回答、实际引用和调用日志统一提交，避免成功消息没有可追溯来源。
        persistent_assistant = get_message(db, assistant_message_id)
        persistent_assistant.content = generation.answer
        persistent_assistant.status = "success"
        persistent_assistant.model = generation.model
        persistent_assistant.prompt_tokens = generation.prompt_tokens
        persistent_assistant.completion_tokens = generation.completion_tokens
        persistent_assistant.total_tokens = generation.total_tokens
        add_rag_answer_reference_records(
            db,
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            references=generation.references,
        )
        if generation.model:
            create_call_log(
                db,
                call_type="async_session_rag_knowledge_answer",
                trace_id=trace_id,
                session_id=session_id,
                message_id=assistant_message_id,
                model=generation.model,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                total_tokens=generation.total_tokens,
                cost_ms=cost_ms,
                status="success",
                commit=False,
            )
        db.commit()

        if should_refresh_summary_for_session(db, session_id):
            refresh_session_summary(db, session_id)
        return {
            "task_id": task_id,
            "status": "success",
            "assistant_message_id": assistant_message_id,
            "active_version_id": retrieval_context.active_version_id,
        }
    except (ModelCallException, BusinessException) as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000) if start_time else None
        error_message = exc.message
        if assistant_message_id:
            update_message(
                db,
                assistant_message_id,
                content=error_message,
                status="error",
                error_message=error_message,
            )
        create_call_log(
            db,
            call_type="async_session_rag_knowledge_answer",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=getattr(exc, "error_type", ERROR_TYPE_WORKER_EXECUTION_ERROR),
            error_message=error_message,
        )
        failed_task = mark_task_error(
            db,
            task_id,
            error_message,
            cost_ms,
            error_type=getattr(exc, "error_type", ERROR_TYPE_WORKER_EXECUTION_ERROR),
        )
        if failed_task and failed_task.retry_count < failed_task.max_retries:
            delay_seconds = min(300, 5 * (2 ** failed_task.retry_count))
            retry_task, retry_event = prepare_session_rag_task_retry(
                db,
                task_id=task_id,
                document_id=document_id,
                retrieval_top_k=retrieval_top_k,
                max_context_characters=max_context_characters,
                delay_seconds=delay_seconds,
            )
            dispatch_outbox_event(db, retry_event.event_id)
            return {
                "task_id": retry_task.task_id,
                "status": "retry_scheduled",
                "retry_count": retry_task.retry_count,
            }
        return {"task_id": task_id, "status": "error"}
    except Exception as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000) if start_time else None
        error_message = f"异步会话 RAG 任务执行异常：{type(exc).__name__}"
        logger.exception("task_id=%s %s", task_id, error_message)
        if assistant_message_id:
            update_message(
                db,
                assistant_message_id,
                content=error_message,
                status="error",
                error_message=error_message,
            )
        create_call_log(
            db,
            call_type="async_session_rag_knowledge_answer",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=ERROR_TYPE_WORKER_EXECUTION_ERROR,
            error_message=error_message,
        )
        mark_task_error(
            db,
            task_id,
            error_message,
            cost_ms,
            error_type=ERROR_TYPE_WORKER_EXECUTION_ERROR,
        )
        return {"task_id": task_id, "status": "error"}
    finally:
        db.close()


@celery_app.task(name="day04_app.tasks.ai_tasks.execute_session_chat_task")
def execute_session_chat_task(
    task_id: str,
    session_id: str,
    message: str,
    trace_id: str | None,
    history_limit: int,
) -> dict:
    """处理一条会话聊天任务；同一任务的重复 Broker 消息会被条件领取逻辑忽略。"""
    db = SessionLocal()
    assistant_message_id: str | None = None
    start_time: float | None = None
    try:
        task = claim_pending_task_for_execution(db, task_id)
        if task is None:
            return {"task_id": task_id, "status": "ignored"}

        # 用户问题在接口事务中已经落库，这里先从短期历史移除再追加，保证模型只收到一次。
        messages = build_messages(
            db=db,
            session_id=session_id,
            current_question=message,
            history_limit=history_limit,
            exclude_latest_matching_user_message=True,
        )
        assistant_message = add_message(
            db,
            session_id,
            "assistant",
            "AI 异步回答生成中",
            trace_id=trace_id,
            model=settings.dashscope_model,
            status="pending",
        )
        assistant_message_id = assistant_message.message_id
        if bind_task_message(db, task_id, assistant_message_id) is None:
            return {"task_id": task_id, "status": "ignored"}

        start_time = time.perf_counter()
        result = safe_chat_with_messages(messages)
        cost_ms = round((time.perf_counter() - start_time) * 1000)

        # 先确认任务仍处于 running，避免超时扫描后的晚到结果覆盖 error 状态。
        completed_task = mark_task_success(
            db,
            task_id=task_id,
            result_text=result.answer,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_ms=cost_ms,
        )
        if completed_task is None:
            return {"task_id": task_id, "status": "ignored"}

        update_message(
            db,
            assistant_message_id,
            content=result.answer,
            status="success",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        )
        create_call_log(
            db,
            call_type="async_session_chat",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=settings.dashscope_model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_ms=cost_ms,
            status="success",
        )
        if should_refresh_summary_for_session(db, session_id):
            refresh_session_summary(db, session_id)
        return {"task_id": task_id, "status": "success"}
    except ModelCallException as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000) if start_time else None
        error_message = exc.message
        if assistant_message_id:
            update_message(
                db,
                assistant_message_id,
                content=error_message,
                status="error",
                error_message=error_message,
            )
        create_call_log(
            db,
            call_type="async_session_chat",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=exc.error_type,
            error_message=error_message,
        )
        failed_task = mark_task_error(db, task_id, error_message, cost_ms, error_type=exc.error_type)
        if failed_task and failed_task.retry_count < failed_task.max_retries:
            # 指数退避：第 1 次 5 秒、第 2 次 10 秒，最大等待 5 分钟。
            delay_seconds = min(300, 5 * (2 ** failed_task.retry_count))
            retry_task, retry_event = prepare_task_retry(
                db,
                task_id=task_id,
                history_limit=history_limit,
                delay_seconds=delay_seconds,
            )
            dispatch_outbox_event(db, retry_event.event_id)
            return {
                "task_id": retry_task.task_id,
                "status": "retry_scheduled",
                "retry_count": retry_task.retry_count,
            }
        return {"task_id": task_id, "status": "error"}
    except Exception as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000) if start_time else None
        error_message = f"异步任务执行异常：{type(exc).__name__}"
        logger.exception("task_id=%s %s", task_id, error_message)
        if assistant_message_id:
            update_message(
                db,
                assistant_message_id,
                content=error_message,
                status="error",
                error_message=error_message,
            )
        create_call_log(
            db,
            call_type="async_session_chat",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=ERROR_TYPE_WORKER_EXECUTION_ERROR,
            error_message=error_message,
        )
        mark_task_error(db, task_id, error_message, cost_ms, error_type=ERROR_TYPE_WORKER_EXECUTION_ERROR)
        return {"task_id": task_id, "status": "error"}
    finally:
        db.close()


@celery_app.task(name="day04_app.tasks.ai_tasks.execute_work_order_analysis_task")
def execute_work_order_analysis_task(
    task_id: str,
    session_id: str,
    content: str,
    trace_id: str | None,
    business_id: str | None,
) -> dict:
    """处理一条工单结构化分析任务，成功后写入 ai_structured_result。"""
    db = SessionLocal()
    assistant_message_id: str | None = None
    start_time: float | None = None
    try:
        task = claim_pending_task_for_execution(db, task_id)
        if task is None:
            return {"task_id": task_id, "status": "ignored"}

        assistant_message = add_message(
            db,
            session_id,
            "assistant",
            "AI 工单结构化分析中",
            trace_id=trace_id,
            model=settings.dashscope_model,
            status="pending",
        )
        assistant_message_id = assistant_message.message_id
        if bind_task_message(db, task_id, assistant_message_id) is None:
            return {"task_id": task_id, "status": "ignored"}

        start_time = time.perf_counter()
        result = analyze_work_order_structured(content)
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        result_json = result.analysis.model_dump()

        structured_result = create_structured_result(
            db,
            business_type="work_order",
            business_id=business_id,
            schema_type="work_order_analysis",
            schema_version="v1",
            result_data=result_json,
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=assistant_message_id,
            status="success",
        )
        result_text = result.analysis.model_dump_json()
        completed_task = mark_task_success(
            db,
            task_id=task_id,
            result_text=result_text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_ms=cost_ms,
        )
        if completed_task is None:
            return {"task_id": task_id, "status": "ignored"}

        # 聊天消息保存结构化摘要，完整标准 JSON 以 ai_structured_result 为准。
        update_message(
            db,
            assistant_message_id,
            content=f"工单结构化分析完成，result_id={structured_result.result_id}",
            status="success",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        )
        create_call_log(
            db,
            call_type="async_work_order_analysis",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=settings.dashscope_model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_ms=cost_ms,
            status="success",
        )
        return {"task_id": task_id, "status": "success", "result_id": structured_result.result_id}
    except ModelCallException as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000) if start_time else None
        error_message = exc.message
        if assistant_message_id:
            update_message(
                db,
                assistant_message_id,
                content=error_message,
                status="error",
                error_message=error_message,
            )
        create_call_log(
            db,
            call_type="async_work_order_analysis",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=exc.error_type,
            error_message=error_message,
        )
        failed_task = mark_task_error(db, task_id, error_message, cost_ms, error_type=exc.error_type)
        if failed_task and failed_task.retry_count < failed_task.max_retries:
            delay_seconds = min(300, 5 * (2 ** failed_task.retry_count))
            retry_task, retry_event = prepare_work_order_analysis_task_retry(
                db,
                task_id=task_id,
                business_id=business_id,
                delay_seconds=delay_seconds,
            )
            dispatch_outbox_event(db, retry_event.event_id)
            return {
                "task_id": retry_task.task_id,
                "status": "retry_scheduled",
                "retry_count": retry_task.retry_count,
            }
        return {"task_id": task_id, "status": "error"}
    except Exception as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000) if start_time else None
        error_message = f"结构化分析异步任务执行异常：{type(exc).__name__}"
        logger.exception("task_id=%s %s", task_id, error_message)
        if assistant_message_id:
            update_message(
                db,
                assistant_message_id,
                content=error_message,
                status="error",
                error_message=error_message,
            )
        create_call_log(
            db,
            call_type="async_work_order_analysis",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=ERROR_TYPE_WORKER_EXECUTION_ERROR,
            error_message=error_message,
        )
        mark_task_error(db, task_id, error_message, cost_ms, error_type=ERROR_TYPE_WORKER_EXECUTION_ERROR)
        return {"task_id": task_id, "status": "error"}
    finally:
        db.close()


@celery_app.task(name="day04_app.tasks.ai_tasks.execute_work_order_eval_task")
def execute_work_order_eval_task(
    task_id: str,
    trace_id: str | None,
    prompt_name: str,
    prompt_version: str,
    dataset_version: str,
) -> dict:
    """处理一条工单评测任务，成功后写入 ai_eval_run 和 ai_eval_case_result。"""
    db = SessionLocal()
    start_time: float | None = None
    try:
        task = claim_pending_task_for_execution(db, task_id)
        if task is None:
            return {"task_id": task_id, "status": "ignored"}

        start_time = time.perf_counter()
        report = run_work_order_eval(
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            dataset_version=dataset_version,
        )
        save_eval_report(db, report)
        cost_ms = round((time.perf_counter() - start_time) * 1000)

        result_text = (
            f"评测完成，run_id={report['run_id']}，"
            f"schema_valid_rate={report['metrics']['schema_valid_rate']}，"
            f"risk_level_accuracy={report['metrics']['risk_level_accuracy']}"
        )
        completed_task = mark_task_success(
            db,
            task_id=task_id,
            result_text=result_text,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cost_ms=cost_ms,
        )
        if completed_task is None:
            return {"task_id": task_id, "status": "ignored"}

        create_call_log(
            db,
            call_type="async_work_order_eval",
            trace_id=trace_id,
            model=None,
            total_tokens=int(report["metrics"]["avg_total_tokens"] * report["metrics"]["sample_count"]),
            cost_ms=cost_ms,
            status="success",
        )
        return {"task_id": task_id, "status": "success", "run_id": report["run_id"]}
    except Exception as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000) if start_time else None
        error_message = f"工单评测异步任务执行异常：{type(exc).__name__}"
        logger.exception("task_id=%s %s", task_id, error_message)
        create_call_log(
            db,
            call_type="async_work_order_eval",
            trace_id=trace_id,
            model=None,
            cost_ms=cost_ms,
            status="error",
            error_type=ERROR_TYPE_WORKER_EXECUTION_ERROR,
            error_message=error_message,
        )
        failed_task = mark_task_error(
            db,
            task_id,
            error_message,
            cost_ms,
            error_type=ERROR_TYPE_WORKER_EXECUTION_ERROR,
        )
        if failed_task and failed_task.retry_count < failed_task.max_retries:
            delay_seconds = min(300, 5 * (2 ** failed_task.retry_count))
            retry_task, retry_event = prepare_work_order_eval_task_retry(
                db,
                task_id=task_id,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                dataset_version=dataset_version,
                delay_seconds=delay_seconds,
            )
            dispatch_outbox_event(db, retry_event.event_id)
            return {
                "task_id": retry_task.task_id,
                "status": "retry_scheduled",
                "retry_count": retry_task.retry_count,
            }
        return {"task_id": task_id, "status": "error"}
    finally:
        db.close()
