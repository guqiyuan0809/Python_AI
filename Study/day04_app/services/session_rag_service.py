"""会话 RAG 编排：消息、回答引用和模型调用日志形成可追溯闭环。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException, ModelCallException
from day04_app.models import AiRagAnswerReference, ChatMessage
from day04_app.schemas.knowledge_schema import (
    MilvusChunkSearchItem,
    RagAnswerReferenceItem,
    RagContextReference,
)
from day04_app.services.call_log_service import create_call_log
from day04_app.services.knowledge_milvus_search_service import search_active_document_chunks
from day04_app.services.rag_context_service import (
    RagContextBuildResult,
    build_rag_context,
    get_rag_answer_prompt_identity,
)
from day04_app.services.llamaindex_rag_query_service import (
    answer_prepared_document_with_llamaindex,
    prepare_governed_llamaindex_rag,
)
from day04_app.services.session_service import (
    add_message,
    get_message,
    get_session,
    refresh_session_summary,
    should_refresh_summary_for_session,
    update_message,
)


@dataclass(frozen=True)
class SessionRagAnswerResult:
    """会话 RAG 最终结果，HTTP 层只负责转换为响应 DTO。"""

    user_message_id: str
    assistant_message_id: str
    answer: str
    references: list[RagAnswerReferenceItem]
    document_id: str
    active_version_id: str
    retrieved_chunk_count: int
    included_chunk_count: int
    omitted_chunk_count: int
    top_score: float | None
    score_threshold: float | None
    rejected_by_score_threshold: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_ms: int


@dataclass(frozen=True)
class SessionRagRetrievalContext:
    """RAG 检索中间结果，供同步接口和 Celery Worker 复用同一条检索链路。"""

    active_version_id: str
    items: list[MilvusChunkSearchItem]
    context_result: RagContextBuildResult


def _to_reference_item(reference: AiRagAnswerReference) -> RagAnswerReferenceItem:
    """数据库 JSON 快照恢复成对外稳定 DTO，避免前端自行解析 locations_json。"""
    try:
        locations = json.loads(reference.locations_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise BusinessException(code=50057, message="RAG 回答引用来源数据损坏") from exc
    return RagAnswerReferenceItem(
        reference_id=reference.reference_id,
        assistant_message_id=reference.assistant_message_id,
        source_id=reference.source_id,
        document_id=reference.document_id,
        version_id=reference.version_id,
        chunk_id=reference.chunk_id,
        chunk_index=reference.chunk_index,
        score=reference.score,
        locations=locations,
        created_at=reference.created_at.isoformat(timespec="seconds"),
    )


def add_rag_answer_reference_records(
    db: Session,
    *,
    session_id: str,
    assistant_message_id: str,
    references: list[RagContextReference],
) -> list[AiRagAnswerReference]:
    """把模型实际使用的来源写为快照；未被引用的召回 chunk 不应进入审计表。"""
    records = [
        AiRagAnswerReference(
            reference_id=uuid4().hex,
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            source_id=reference.source_id,
            document_id=reference.document_id,
            version_id=reference.version_id,
            chunk_id=reference.chunk_id,
            chunk_index=reference.chunk_index,
            score=reference.score,
            locations_json=json.dumps(reference.locations, ensure_ascii=False),
        )
        for reference in references
    ]
    db.add_all(records)
    return records


def prepare_session_rag_context(
    db: Session,
    *,
    document_id: str,
    message: str,
    retrieval_top_k: int,
    max_context_characters: int,
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    score_threshold: float | None = None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
) -> SessionRagRetrievalContext:
    """旧版无框架 RAG 对照实现，保留用于历史 Run 解释与回归比较。

    它展示了没有 LlamaIndex 时项目必须手写的两步：
    ``search_active_document_chunks → build_rag_context``。Day31 后正式同步与异步会话
    已改为 ``ExistingKnowledgeMilvusRetriever → GovernedRagNodePostprocessor →
    RetrieverQueryEngine``，不再调用本函数；请勿删除，否则历史日志中的
    ``rag_context_build`` 语义将难以理解。
    """
    retrieval_start_time = time.perf_counter()
    embedding_model, _, active_version_id, items = search_active_document_chunks(
        db,
        document_id=document_id,
        question=message,
        top_k=retrieval_top_k,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
    )
    retrieval_cost_ms = round((time.perf_counter() - retrieval_start_time) * 1000)
    create_call_log(
        db,
        call_type="session_rag",
        stage="rag_retrieval",
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
        model=embedding_model,
        cost_ms=retrieval_cost_ms,
        detail={
            "document_id": document_id,
            "version_id": active_version_id,
            "retrieved_chunk_count": len(items),
            "retrieval_top_k": retrieval_top_k,
            "use_reranker": use_reranker,
            "rerank_top_n": rerank_top_n if use_reranker else None,
        },
        commit=False,
    )

    context_start_time = time.perf_counter()
    context_result = build_rag_context(
        items,
        max_context_characters=max_context_characters,
        score_threshold=score_threshold,
    )
    create_call_log(
        db,
        call_type="session_rag",
        stage="rag_context_build",
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
        cost_ms=round((time.perf_counter() - context_start_time) * 1000),
        detail={
            "version_id": active_version_id,
            "included_reference_count": len(context_result.references),
            "omitted_chunk_count": context_result.omitted_chunk_count,
            "score_threshold": context_result.score_threshold,
            "rejected_by_score_threshold": context_result.rejected_by_score_threshold,
        },
        commit=False,
    )
    return SessionRagRetrievalContext(
        active_version_id=active_version_id,
        items=items,
        context_result=context_result,
    )


def _create_rag_request_summary(
    db: Session,
    *,
    trace_id: str | None,
    session_id: str,
    assistant_message_id: str,
    document_id: str,
    active_version_id: str | None,
    model: str | None,
    total_tokens: int | None,
    cost_ms: int,
    status: str,
    rejected_by_score_threshold: bool | None = None,
    used_reference_count: int | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    commit: bool = True,
) -> None:
    """写入同步 RAG 的根摘要，端到端指标不能由阶段事件相加得到。"""
    create_call_log(
        db,
        call_type="session_rag",
        stage="rag_request_summary",
        trace_id=trace_id,
        session_id=session_id,
        message_id=assistant_message_id,
        model=model,
        total_tokens=total_tokens,
        cost_ms=cost_ms,
        status=status,
        error_type=error_type,
        error_message=error_message,
        detail={
            "summary": True,
            "summary_type": "rag_request",
            "document_id": document_id,
            "version_id": active_version_id,
            "model_called": model is not None,
            "rejected_by_score_threshold": rejected_by_score_threshold,
            "used_reference_count": used_reference_count,
        },
        commit=commit,
    )


def answer_session_with_rag(
    db: Session,
    *,
    session_id: str,
    document_id: str,
    message: str,
    trace_id: str | None,
    retrieval_top_k: int,
    max_context_characters: int,
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    score_threshold: float | None = None,
) -> SessionRagAnswerResult:
    """同步完成一次会话 RAG：消息、引用、Trace 与 LlamaIndex 编排统一可追溯。

    旧实现把“检索、字符串上下文拼装、模型调用”分散在本服务和
    ``rag_context_service``。现在本服务仅管理会话事务和审计；LlamaIndex QueryEngine
    管理节点到回答的编排。这样 Celery Worker 也能复用同一个 QueryEngine 入口。
    """
    get_session(db, session_id)
    user_message = add_message(
        db,
        session_id,
        "user",
        message,
        trace_id=trace_id,
    )
    assistant_message = add_message(
        db,
        session_id,
        "assistant",
        "知识库回答生成中",
        trace_id=trace_id,
        status="pending",
    )

    # 根摘要从检索前开始计时，覆盖 Embedding、召回、精排、上下文和答案生成。
    rag_start_time = time.perf_counter()
    try:
        # legacy 对照（Day21，不再调用）：
        # retrieval_context = prepare_session_rag_context(...)
        # generation = generate_rag_answer(...)
        #
        # 正式框架链路：项目先构造受治理的 Retriever/Postprocessor/Prompt，再交给
        # QueryEngine。这里没有把 DB、权限或会话状态交给 LlamaIndex，它只编排 RAG。
        preparation = prepare_governed_llamaindex_rag(
            db,
            document_id=document_id,
            retrieval_top_k=retrieval_top_k,
            max_context_characters=max_context_characters,
            use_reranker=use_reranker,
            rerank_top_n=rerank_top_n,
            score_threshold=score_threshold,
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message.message_id,
        )
        framework_result = answer_prepared_document_with_llamaindex(
            question=message,
            preparation=preparation,
        )
    except (BusinessException, ModelCallException) as exc:
        update_message(
            db,
            assistant_message.message_id,
            content=exc.message,
            status="error",
            error_message=exc.message,
        )
        _create_rag_request_summary(
            db,
            trace_id=trace_id,
            session_id=session_id,
            assistant_message_id=assistant_message.message_id,
            document_id=document_id,
            active_version_id=None,
            model=None,
            total_tokens=None,
            cost_ms=round((time.perf_counter() - rag_start_time) * 1000),
            status="error",
            error_type=getattr(exc, "error_type", type(exc).__name__),
            error_message=exc.message,
        )
        raise

    generation_cost_ms = round((time.perf_counter() - rag_start_time) * 1000)
    rag_total_cost_ms = generation_cost_ms
    try:
        # 回答状态、实际引用和调用日志同一次事务提交，不能出现成功回答却没有来源审计。
        persistent_assistant = get_message(db, assistant_message.message_id)
        persistent_assistant.content = framework_result.answer
        persistent_assistant.status = "success"
        persistent_assistant.model = framework_result.model
        persistent_assistant.prompt_tokens = framework_result.prompt_tokens
        persistent_assistant.completion_tokens = framework_result.completion_tokens
        persistent_assistant.total_tokens = framework_result.total_tokens
        reference_records = add_rag_answer_reference_records(
            db,
            session_id=session_id,
            assistant_message_id=assistant_message.message_id,
            references=framework_result.references,
        )
        # Query Embedding、Milvus search、Reranker 已由受治理 Retriever 各自记阶段日志；
        # 此处补一条框架编排摘要，排查时可清楚看见“不是手工 build_rag_context”。
        create_call_log(
            db,
            call_type="session_rag",
            stage="llamaindex_query_engine",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message.message_id,
            model=framework_result.model,
            prompt_tokens=framework_result.prompt_tokens,
            completion_tokens=framework_result.completion_tokens,
            total_tokens=framework_result.total_tokens,
            cost_ms=generation_cost_ms,
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
        _create_rag_request_summary(
            db,
            trace_id=trace_id,
            session_id=session_id,
            assistant_message_id=assistant_message.message_id,
            document_id=document_id,
            active_version_id=framework_result.retrieval.active_version_id,
            model=framework_result.model,
            total_tokens=framework_result.total_tokens,
            cost_ms=rag_total_cost_ms,
            status="success",
            rejected_by_score_threshold=framework_result.rejected_by_score_threshold,
            used_reference_count=len(framework_result.references),
            commit=False,
        )
        db.commit()
        for reference_record in reference_records:
            db.refresh(reference_record)
    except ModelCallException as exc:
        db.rollback()
        generation_cost_ms = round((time.perf_counter() - rag_start_time) * 1000)
        update_message(
            db,
            assistant_message.message_id,
            content=exc.message,
            status="error",
            error_message=exc.message,
        )
        create_call_log(
            db,
            call_type="session_rag",
            stage="llamaindex_query_engine",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message.message_id,
            model=(preparation.runtime_prompt.model if "preparation" in locals() else None),
            cost_ms=generation_cost_ms,
            status="error",
            error_type=exc.error_type,
            error_message=exc.message,
            **(
                get_rag_answer_prompt_identity(preparation.runtime_prompt).as_call_log_fields()
                if "preparation" in locals()
                else {}
            ),
            detail={"framework": "llamaindex", "prompt_source": "database"} if "preparation" in locals() else None,
        )
        _create_rag_request_summary(
            db,
            trace_id=trace_id,
            session_id=session_id,
            assistant_message_id=assistant_message.message_id,
            document_id=document_id,
            active_version_id=(framework_result.retrieval.active_version_id if "framework_result" in locals() else None),
            model=None,
            total_tokens=None,
            cost_ms=round((time.perf_counter() - rag_start_time) * 1000),
            status="error",
            rejected_by_score_threshold=(framework_result.rejected_by_score_threshold if "framework_result" in locals() else None),
            error_type=exc.error_type,
            error_message=exc.message,
        )
        raise

    if should_refresh_summary_for_session(db, session_id):
        refresh_session_summary(db, session_id)

    return SessionRagAnswerResult(
        user_message_id=user_message.message_id,
        assistant_message_id=assistant_message.message_id,
        answer=framework_result.answer,
        references=[_to_reference_item(record) for record in reference_records],
        document_id=document_id,
        active_version_id=framework_result.retrieval.active_version_id,
        retrieved_chunk_count=framework_result.retrieved_node_count,
        included_chunk_count=framework_result.included_node_count,
        omitted_chunk_count=framework_result.omitted_node_count,
        top_score=framework_result.top_score,
        score_threshold=framework_result.score_threshold,
        rejected_by_score_threshold=framework_result.rejected_by_score_threshold,
        prompt_tokens=framework_result.prompt_tokens,
        completion_tokens=framework_result.completion_tokens,
        total_tokens=framework_result.total_tokens,
        cost_ms=generation_cost_ms,
    )


def list_session_rag_answer_references(
    db: Session,
    *,
    session_id: str,
    assistant_message_id: str,
) -> list[RagAnswerReferenceItem]:
    """按会话和回答消息查询引用，避免其他会话通过 message_id 越权查看来源。"""
    message = get_message(db, assistant_message_id)
    if message.session_id != session_id or message.role != "assistant":
        raise BusinessException(code=40071, message="RAG 回答消息不存在或不属于当前会话")
    records = list(
        db.scalars(
            select(AiRagAnswerReference)
            .where(
                AiRagAnswerReference.session_id == session_id,
                AiRagAnswerReference.assistant_message_id == assistant_message_id,
            )
            .order_by(AiRagAnswerReference.id.asc())
        )
    )
    return [_to_reference_item(record) for record in records]
