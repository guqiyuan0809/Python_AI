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
    prepare_contextual_index_task_retry,
    prepare_agent_loop_task_retry,
    prepare_agent_loop_eval_task_retry,
    prepare_work_order_eval_task_retry,
    prepare_work_order_analysis_task_retry,
    prepare_task_retry,
)
from day04_app.services.call_log_service import create_call_log
from day04_app.services.chat_service import (
    analyze_work_order_structured_with_runtime_prompt,
    get_active_work_order_analysis_repair_prompt,
    get_work_order_analysis_prompt_identity,
    get_work_order_analysis_repair_prompt_identity,
    safe_chat_with_messages,
)
from day04_app.services.eval_master_service import get_active_prompt_version_for_runtime
from day04_app.services.outbox_dispatcher import dispatch_outbox_event
from day04_app.services.rag_context_service import (
    get_rag_answer_prompt_identity,
)
from day04_app.services.llamaindex_rag_query_service import (
    answer_prepared_document_with_llamaindex,
    prepare_governed_llamaindex_rag,
)
from day04_app.services.session_rag_service import (
    add_rag_answer_reference_records,
)
from day04_app.services.session_service import (
    build_messages,
    create_or_reuse_task_assistant_message,
    get_message,
    refresh_session_summary,
    should_refresh_summary_for_session,
    set_session_turn_status_for_task,
    update_message,
)
from day04_app.services.structured_result_service import create_structured_result
from day04_app.services.eval_result_service import save_eval_report
from day04_app.services.agent_loop_eval_result_service import save_agent_eval_report
from day04_app.services.agent_loop_eval_runner import run_agent_loop_eval
from day04_app.services.agent_loop_service import run_agent_loop
from day04_app.services.agent_memory_service import get_active_memory_for_embedding
from day04_app.services.knowledge_embedding_service import generate_text_embeddings
from day04_app.services.milvus_vector_store_service import upsert_session_memory_vector
from day04_app.security.principal import SecurityPrincipal
from day04_app.services.work_order_eval_runner import run_work_order_eval
from day04_app.services.knowledge_contextualization_service import build_contextual_vector_index
from settings import settings


logger = logging.getLogger("day04_app.worker")


@celery_app.task(name="day04_app.tasks.ai_tasks.index_session_memory")
def index_session_memory(memory_id: str) -> dict:
    """异步向量化一条已经过 MySQL 治理的长期记忆。

    不接受聊天正文或工具 observation 参数，避免调用方绕过 ``SessionMemory`` 的留存、范围
    与状态控制。Milvus 失败时保留 MySQL pending/error 状态，后续可补偿重试。
    """

    db = SessionLocal()
    try:
        record = get_active_memory_for_embedding(db, memory_id)
        model, vectors = generate_text_embeddings([record.content])
        upsert_session_memory_vector(
            record={
                "memory_id": record.memory_id,
                "session_id": record.session_id or "",
                "user_id": record.user_id or "",
                "tenant_id": record.tenant_id or "",
                "memory_type": record.memory_type,
                "status": record.status,
                "embedding": vectors[0],
            }
        )
        record.embedding_status = "indexed"
        record.embedding_error_message = None
        db.commit()
        return {"memory_id": memory_id, "status": "indexed", "embedding_model": model}
    except Exception as exc:
        try:
            record = get_active_memory_for_embedding(db, memory_id)
            record.embedding_status = "error"
            record.embedding_error_message = f"{type(exc).__name__}"
            db.commit()
        except Exception:
            db.rollback()
        logger.exception("memory_id=%s session memory embedding failed", memory_id)
        raise
    finally:
        db.close()


@celery_app.task(name="day04_app.tasks.ai_tasks.execute_agent_loop_task")
def execute_agent_loop_task(
    task_id: str,
    trace_id: str | None,
    message: str,
    max_steps: int,
    principal: dict | None = None,
) -> dict:
    """消费 Outbox 事件后执行一条在线 Agent Loop 请求。"""
    db = SessionLocal()
    start_time: float | None = None
    try:
        task = claim_pending_task_for_execution(db, task_id)
        if task is None:
            return {"task_id": task_id, "status": "ignored"}

        start_time = time.perf_counter()
        response = run_agent_loop(
            db,
            message=message,
            max_steps=max_steps,
            trace_id=trace_id,
            task_id=task_id,
            principal=SecurityPrincipal.from_snapshot(principal),
        )
        completed_task = mark_task_success(
            db,
            task_id=task_id,
            result_text=response.answer,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            cost_ms=response.cost_ms,
        )
        if completed_task is None:
            return {"task_id": task_id, "status": "ignored"}
        create_call_log(
            db,
            call_type="async_agent_loop",
            trace_id=trace_id,
            task_id=task_id,
            stage="agent_task_summary",
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            cost_ms=response.cost_ms,
            status="success",
            detail={
                "agent_status": response.status,
                "step_count": len(response.steps),
            },
        )
        return {"task_id": task_id, "status": "success"}
    except Exception as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000) if start_time else None
        error_message = f"Agent Loop 异步任务执行异常：{type(exc).__name__}"
        logger.exception("task_id=%s %s", task_id, error_message)
        create_call_log(
            db,
            call_type="async_agent_loop",
            trace_id=trace_id,
            task_id=task_id,
            stage="agent_task_summary",
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
            retry_task, retry_event = prepare_agent_loop_task_retry(
                db,
                task_id=task_id,
                max_steps=max_steps,
                principal=SecurityPrincipal.from_snapshot(principal),
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


@celery_app.task(name="day04_app.tasks.ai_tasks.execute_session_rag_task")
def execute_session_rag_task(
    task_id: str,
    session_id: str,
    message: str,
    trace_id: str | None,
    document_id: str,
    retrieval_top_k: int,
    max_context_characters: int,
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    score_threshold: float | None = None,
) -> dict:
    """异步会话 RAG：检索、回答、引用快照和任务终态在 Worker 内完成。"""
    db = SessionLocal()
    assistant_message_id: str | None = None
    start_time: float | None = None
    try:
        task = claim_pending_task_for_execution(db, task_id)
        if task is None:
            return {"task_id": task_id, "status": "ignored"}

        assistant_message = create_or_reuse_task_assistant_message(
            db,
            task_id=task_id,
            session_id=session_id,
            trace_id=trace_id,
            placeholder_content="知识库回答生成中",
            model=settings.dashscope_model,
        )
        assistant_message_id = assistant_message.message_id
        if bind_task_message(db, task_id, assistant_message_id) is None:
            return {"task_id": task_id, "status": "ignored"}

        start_time = time.perf_counter()
        # legacy 对照：旧 Worker 调用 prepare_session_rag_context + generate_rag_answer。
        # 现在与同步入口共享同一条 LlamaIndex QueryEngine 编排，避免线上与异步的
        # 阈值、父块去重、Prompt 填充规则发生漂移。
        preparation = prepare_governed_llamaindex_rag(
            db,
            document_id=document_id,
            retrieval_top_k=retrieval_top_k,
            max_context_characters=max_context_characters,
            use_reranker=use_reranker,
            rerank_top_n=rerank_top_n,
            score_threshold=score_threshold,
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=assistant_message_id,
        )
        framework_result = answer_prepared_document_with_llamaindex(
            question=message,
            preparation=preparation,
        )
        cost_ms = round((time.perf_counter() - start_time) * 1000)

        completed_task = mark_task_success(
            db,
            task_id=task_id,
            result_text=framework_result.answer,
            prompt_tokens=framework_result.prompt_tokens,
            completion_tokens=framework_result.completion_tokens,
            total_tokens=framework_result.total_tokens,
            cost_ms=cost_ms,
            commit=False,
        )
        if completed_task is None:
            return {"task_id": task_id, "status": "ignored"}

        # 回答、实际引用和调用日志统一提交，避免成功消息没有可追溯来源。
        persistent_assistant = get_message(db, assistant_message_id)
        persistent_assistant.content = framework_result.answer
        persistent_assistant.status = "success"
        persistent_assistant.model = framework_result.model
        persistent_assistant.prompt_tokens = framework_result.prompt_tokens
        persistent_assistant.completion_tokens = framework_result.completion_tokens
        persistent_assistant.total_tokens = framework_result.total_tokens
        set_session_turn_status_for_task(
            db,
            task_id=task_id,
            status="success",
            assistant_message_id=assistant_message_id,
            commit=False,
        )
        add_rag_answer_reference_records(
            db,
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            references=framework_result.references,
        )
        create_call_log(
            db,
            call_type="async_session_rag",
            stage="llamaindex_query_engine",
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=framework_result.model,
            prompt_tokens=framework_result.prompt_tokens,
            completion_tokens=framework_result.completion_tokens,
            total_tokens=framework_result.total_tokens,
            cost_ms=cost_ms,
            status="success",
            **(
                framework_result.prompt_identity.as_call_log_fields()
                if framework_result.prompt_identity
                else {}
            ),
            detail={
                "framework": "llamaindex",
                "orchestration": "RetrieverQueryEngine",
                "node_postprocessor": "GovernedRagNodePostprocessor",
                "used_reference_count": len(framework_result.references),
                "model_called": framework_result.model is not None,
                "prompt_source": (
                    framework_result.prompt_identity.prompt_source
                    if framework_result.prompt_identity
                    else "none"
                ),
                "retrieved_node_count": framework_result.retrieved_node_count,
                "included_node_count": framework_result.included_node_count,
                "omitted_node_count": framework_result.omitted_node_count,
                "rejected_by_score_threshold": framework_result.rejected_by_score_threshold,
            },
            commit=False,
        )
        db.commit()

        if should_refresh_summary_for_session(db, session_id):
            refresh_session_summary(db, session_id)
        return {
            "task_id": task_id,
            "status": "success",
            "assistant_message_id": assistant_message_id,
            "active_version_id": framework_result.retrieval.active_version_id,
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
            call_type="async_session_rag",
            stage="llamaindex_query_engine",
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=(preparation.runtime_prompt.model or settings.dashscope_model) if "preparation" in locals() else settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=getattr(exc, "error_type", ERROR_TYPE_WORKER_EXECUTION_ERROR),
            error_message=error_message,
            **(
                get_rag_answer_prompt_identity(preparation.runtime_prompt).as_call_log_fields()
                if "preparation" in locals()
                else {}
            ),
            detail={"framework": "llamaindex", "prompt_source": "database"} if "preparation" in locals() else None,
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
                use_reranker=use_reranker,
                rerank_top_n=rerank_top_n,
                score_threshold=score_threshold,
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
            call_type="async_session_rag",
            stage="llamaindex_query_engine",
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=(preparation.runtime_prompt.model or settings.dashscope_model) if "preparation" in locals() else settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=ERROR_TYPE_WORKER_EXECUTION_ERROR,
            error_message=error_message,
            **(
                get_rag_answer_prompt_identity(preparation.runtime_prompt).as_call_log_fields()
                if "preparation" in locals()
                else {}
            ),
            detail={"framework": "llamaindex", "prompt_source": "database"} if "preparation" in locals() else None,
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
        assistant_message = create_or_reuse_task_assistant_message(
            db,
            task_id=task_id,
            session_id=session_id,
            trace_id=trace_id,
            placeholder_content="AI 异步回答生成中",
            model=settings.dashscope_model,
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

        assistant_message = create_or_reuse_task_assistant_message(
            db,
            task_id=task_id,
            session_id=session_id,
            trace_id=trace_id,
            placeholder_content="AI 工单结构化分析中",
            model=settings.dashscope_model,
        )
        assistant_message_id = assistant_message.message_id
        if bind_task_message(db, task_id, assistant_message_id) is None:
            return {"task_id": task_id, "status": "ignored"}

        start_time = time.perf_counter()
        prompt = get_active_prompt_version_for_runtime(db, "work_order_analysis")
        repair_prompt = get_active_work_order_analysis_repair_prompt(db)
        prompt_identity = get_work_order_analysis_prompt_identity(prompt)
        execution = analyze_work_order_structured_with_runtime_prompt(
            content,
            prompt,
            repair_prompt,
        )
        result = execution.response
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
            stage="prompt_model_generation",
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=prompt.model or settings.dashscope_model,
            prompt_tokens=execution.initial_usage.prompt_tokens,
            completion_tokens=execution.initial_usage.completion_tokens,
            total_tokens=execution.initial_usage.total_tokens,
            cost_ms=execution.initial_usage.cost_ms,
            status="success",
            **prompt_identity.as_call_log_fields(),
            detail={"prompt_source": prompt_identity.prompt_source, "repair_count": result.repair_count},
        )
        if execution.repair_usage:
            create_call_log(
                db,
                call_type="async_work_order_analysis",
                stage="prompt_output_repair",
                trace_id=trace_id,
                task_id=task_id,
                session_id=session_id,
                message_id=assistant_message_id,
                model=repair_prompt.model or settings.dashscope_model,
                prompt_tokens=execution.repair_usage.prompt_tokens,
                completion_tokens=execution.repair_usage.completion_tokens,
                total_tokens=execution.repair_usage.total_tokens,
                cost_ms=execution.repair_usage.cost_ms,
                status="success",
                **get_work_order_analysis_repair_prompt_identity(
                    repair_prompt
                ).as_call_log_fields(),
                detail={"prompt_source": "database", "repair": True},
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
            stage="prompt_model_generation",
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=exc.error_type,
            error_message=error_message,
            **(
                get_work_order_analysis_prompt_identity(prompt).as_call_log_fields()
                if "prompt" in locals()
                else {}
            ),
            detail={"prompt_source": "database"} if "prompt" in locals() else None,
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
            stage="prompt_model_generation",
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=ERROR_TYPE_WORKER_EXECUTION_ERROR,
            error_message=error_message,
            **(
                get_work_order_analysis_prompt_identity(prompt).as_call_log_fields()
                if "prompt" in locals()
                else {}
            ),
            detail={"prompt_source": "database"} if "prompt" in locals() else None,
        )
        mark_task_error(db, task_id, error_message, cost_ms, error_type=ERROR_TYPE_WORKER_EXECUTION_ERROR)
        return {"task_id": task_id, "status": "error"}
    finally:
        db.close()


@celery_app.task(name="day04_app.tasks.ai_tasks.execute_knowledge_contextual_index_task")
def execute_knowledge_contextual_index_task(
    task_id: str,
    version_id: str,
    context_model: str | None,
    context_max_tokens: int,
    trace_id: str | None,
) -> dict:
    """为候选知识库版本生成上下文说明并构建 Milvus 索引。"""
    db = SessionLocal()
    start_time: float | None = None
    try:
        task = claim_pending_task_for_execution(db, task_id)
        if task is None:
            return {"task_id": task_id, "status": "ignored"}

        start_time = time.perf_counter()
        result = build_contextual_vector_index(
            db,
            version_id=version_id,
            context_model=context_model,
            context_max_tokens=context_max_tokens,
            trace_id=trace_id,
        )
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        completed_task = mark_task_success(
            db,
            task_id=task_id,
            result_text=(
                f"上下文化索引完成，version_id={result.version_id}，"
                f"contextualized_chunk_count={result.contextualized_chunk_count}，"
                f"vector_count={result.vector_count}"
            ),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_ms=cost_ms,
        )
        if completed_task is None:
            return {"task_id": task_id, "status": "ignored"}

        create_call_log(
            db,
            call_type="knowledge_contextual_index",
            trace_id=trace_id,
            model=result.embedding_model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_ms=cost_ms,
            status="success",
        )
        return {
            "task_id": task_id,
            "status": "success",
            "version_id": result.version_id,
            "contextualized_chunk_count": result.contextualized_chunk_count,
            "vector_count": result.vector_count,
        }
    except (ModelCallException, BusinessException) as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000) if start_time else None
        error_message = exc.message
        logger.exception("task_id=%s contextual index failed", task_id)
        create_call_log(
            db,
            call_type="knowledge_contextual_index",
            trace_id=trace_id,
            model=context_model,
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
            retry_task, retry_event = prepare_contextual_index_task_retry(
                db,
                task_id=task_id,
                version_id=version_id,
                context_model=context_model,
                context_max_tokens=context_max_tokens,
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
        error_message = f"上下文化索引异步任务执行异常：{type(exc).__name__}"
        logger.exception("task_id=%s contextual index failed", task_id)
        create_call_log(
            db,
            call_type="knowledge_contextual_index",
            trace_id=trace_id,
            model=context_model,
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
            retry_task, retry_event = prepare_contextual_index_task_retry(
                db,
                task_id=task_id,
                version_id=version_id,
                context_model=context_model,
                context_max_tokens=context_max_tokens,
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


@celery_app.task(name="day04_app.tasks.ai_tasks.execute_agent_loop_eval_task")
def execute_agent_loop_eval_task(
    task_id: str,
    trace_id: str | None,
    agent_version: str,
    dataset_version: str,
    sample_limit: int | None = None,
) -> dict:
    """异步执行 Agent Harness，评测范围由 Outbox payload 固定并在重试时复用。"""
    db = SessionLocal()
    start_time: float | None = None
    try:
        task = claim_pending_task_for_execution(db, task_id)
        if task is None:
            return {"task_id": task_id, "status": "ignored"}

        start_time = time.perf_counter()
        report = run_agent_loop_eval(
            agent_version=agent_version,
            dataset_version=dataset_version,
            sample_limit=sample_limit,
            trace_id=trace_id,
            task_id=task_id,
        )
        save_agent_eval_report(db, report)
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        total_tokens = int(
            report["metrics"]["avg_total_tokens"] * report["metrics"]["sample_count"]
        )
        result_text = (
            f"Agent 评测完成，run_id={report['run_id']}，"
            f"full_pass_rate={report['metrics']['full_pass_rate']}，"
            f"safety_case_pass_rate={report['metrics']['safety_case_pass_rate']}"
        )
        completed_task = mark_task_success(
            db,
            task_id=task_id,
            result_text=result_text,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=total_tokens,
            cost_ms=cost_ms,
        )
        if completed_task is None:
            return {"task_id": task_id, "status": "ignored"}
        create_call_log(
            db,
            call_type="async_agent_loop_eval",
            trace_id=trace_id,
            task_id=task_id,
            run_id=report["run_id"],
            stage="agent_eval_summary",
            model=None,
            total_tokens=total_tokens,
            cost_ms=cost_ms,
            status="success",
            detail={
                "sample_count": report["metrics"]["sample_count"],
                "full_pass_rate": report["metrics"]["full_pass_rate"],
                "safety_case_pass_rate": report["metrics"]["safety_case_pass_rate"],
            },
        )
        return {"task_id": task_id, "status": "success", "run_id": report["run_id"]}
    except Exception as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000) if start_time else None
        error_message = f"Agent Loop 评测异步任务执行异常：{type(exc).__name__}"
        logger.exception("task_id=%s %s", task_id, error_message)
        create_call_log(
            db,
            call_type="async_agent_loop_eval",
            trace_id=trace_id,
            task_id=task_id,
            stage="agent_eval_summary",
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
            retry_task, retry_event = prepare_agent_loop_eval_task_retry(
                db,
                task_id=task_id,
                agent_version=agent_version,
                dataset_version=dataset_version,
                sample_limit=sample_limit,
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
