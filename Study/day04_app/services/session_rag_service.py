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
    generate_rag_answer,
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
) -> SessionRagRetrievalContext:
    """执行 RAG 检索与资料组装，不写会话消息，便于同步与异步入口共用。"""
    _, _, active_version_id, items = search_active_document_chunks(
        db,
        document_id=document_id,
        question=message,
        top_k=retrieval_top_k,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
    )
    context_result = build_rag_context(
        items,
        max_context_characters=max_context_characters,
        score_threshold=score_threshold,
    )
    return SessionRagRetrievalContext(
        active_version_id=active_version_id,
        items=items,
        context_result=context_result,
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
    """同步完成一次会话 RAG：用户消息、回答、引用和调用日志均可按 trace 追溯。"""
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

    try:
        retrieval_context = prepare_session_rag_context(
            db,
            document_id=document_id,
            message=message,
            retrieval_top_k=retrieval_top_k,
            max_context_characters=max_context_characters,
            use_reranker=use_reranker,
            rerank_top_n=rerank_top_n,
            score_threshold=score_threshold,
        )
    except (BusinessException, ModelCallException) as exc:
        update_message(
            db,
            assistant_message.message_id,
            content=exc.message,
            status="error",
            error_message=exc.message,
        )
        raise

    start_time = time.perf_counter()
    try:
        generation = generate_rag_answer(
            question=message,
            context_result=retrieval_context.context_result,
        )
        cost_ms = round((time.perf_counter() - start_time) * 1000)

        # 回答状态、实际引用和调用日志同一次事务提交，不能出现成功回答却没有来源审计。
        persistent_assistant = get_message(db, assistant_message.message_id)
        persistent_assistant.content = generation.answer
        persistent_assistant.status = "success"
        persistent_assistant.model = generation.model
        persistent_assistant.prompt_tokens = generation.prompt_tokens
        persistent_assistant.completion_tokens = generation.completion_tokens
        persistent_assistant.total_tokens = generation.total_tokens
        reference_records = add_rag_answer_reference_records(
            db,
            session_id=session_id,
            assistant_message_id=assistant_message.message_id,
            references=generation.references,
        )
        if generation.model:
            create_call_log(
                db,
                call_type="session_rag_knowledge_answer",
                trace_id=trace_id,
                session_id=session_id,
                message_id=assistant_message.message_id,
                model=generation.model,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                total_tokens=generation.total_tokens,
                cost_ms=cost_ms,
                status="success",
                commit=False,
            )
        db.commit()
        for reference_record in reference_records:
            db.refresh(reference_record)
    except ModelCallException as exc:
        db.rollback()
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        update_message(
            db,
            assistant_message.message_id,
            content=exc.message,
            status="error",
            error_message=exc.message,
        )
        create_call_log(
            db,
            call_type="session_rag_knowledge_answer",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message.message_id,
            cost_ms=cost_ms,
            status="error",
            error_type=exc.error_type,
            error_message=exc.message,
        )
        raise

    if should_refresh_summary_for_session(db, session_id):
        refresh_session_summary(db, session_id)

    return SessionRagAnswerResult(
        user_message_id=user_message.message_id,
        assistant_message_id=assistant_message.message_id,
        answer=generation.answer,
        references=[_to_reference_item(record) for record in reference_records],
        document_id=document_id,
        active_version_id=retrieval_context.active_version_id,
        retrieved_chunk_count=len(retrieval_context.items),
        included_chunk_count=len(retrieval_context.context_result.references),
        omitted_chunk_count=retrieval_context.context_result.omitted_chunk_count,
        top_score=retrieval_context.context_result.top_score,
        score_threshold=retrieval_context.context_result.score_threshold,
        rejected_by_score_threshold=retrieval_context.context_result.rejected_by_score_threshold,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        total_tokens=generation.total_tokens,
        cost_ms=cost_ms,
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
