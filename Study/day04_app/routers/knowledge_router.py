"""知识库文件上传、索引与 RAG 问答接口。"""

import time

from fastapi import APIRouter, Depends, File, Request, UploadFile
from sqlalchemy.orm import Session

from day04_app.common.response import ApiResponse, success
from day04_app.database import get_db
from day04_app.schemas.knowledge_schema import (
    DocumentChunkRequest,
    DocumentChunkResponse,
    DocumentParseResponse,
    DocumentUploadResponse,
    EmbeddingSimilarityTestRequest,
    EmbeddingSimilarityTestResponse,
    DocumentVersionIndexResponse,
    ActivateDocumentVersionRequest,
    ActivateDocumentVersionResponse,
    InMemoryChunkSearchRequest,
    InMemoryChunkSearchResponse,
    MilvusChunkSearchRequest,
    MilvusChunkSearchResponse,
    RagContextPreviewRequest,
    RagContextPreviewResponse,
    RagAnswerRequest,
    RagAnswerResponse,
    RagAnswerReferenceListResponse,
    AsyncRagTaskSubmitResponse,
    AsyncSessionRagTaskRequest,
    SessionRagAnswerRequest,
    SessionRagAnswerResponse,
    CreateDocumentVersionRequest,
    CreateDocumentVersionResponse,
    VersionChunkResponse,
)
from day04_app.services.knowledge_embedding_service import compare_text_embeddings
from day04_app.services.knowledge_in_memory_search_service import search_document_chunks_in_memory
from day04_app.services.knowledge_milvus_search_service import search_active_document_chunks
from day04_app.services.rag_context_service import build_rag_context, generate_rag_answer
from day04_app.services.session_rag_service import (
    answer_session_with_rag,
    list_session_rag_answer_references,
)
from day04_app.services.async_task_service import create_async_session_rag_task
from day04_app.services.outbox_dispatcher import dispatch_outbox_event
from day04_app.services.session_service import get_session
from day04_app.services.call_log_service import create_call_log
from day04_app.common.exceptions import ModelCallException
from day04_app.services.knowledge_vector_index_service import build_version_vector_index
from day04_app.services.knowledge_document_version_service import activate_document_version
from day04_app.services.knowledge_document_chunk_service import (
    _find_document,
    chunk_document_by_id,
    chunk_document_version_by_id,
)
from day04_app.services.knowledge_document_rebuild_service import create_candidate_document_version
from day04_app.services.knowledge_document_parse_service import parse_document_by_id
from day04_app.services.document_storage_service import delete_stored_document, save_uploaded_document
from day04_app.services.knowledge_document_service import create_uploaded_document
from day04_app.services.text_chunker_service import ChunkingConfig
from settings import settings


router = APIRouter(prefix="/api/knowledge", tags=["知识库"])


@router.post("/documents/upload", response_model=ApiResponse[DocumentUploadResponse], summary="安全上传知识库文件")
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="仅支持 docx、pdf、xlsx，最大 20MB"),
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentUploadResponse]:
    stored_document = await save_uploaded_document(file)
    try:
        document = create_uploaded_document(db, stored_document, request.state.trace_id)
    except Exception:
        # 磁盘和 MySQL 不能共享一个事务；数据库写入失败时主动补偿刚落盘的文件。
        delete_stored_document(stored_document)
        raise
    return success(
        DocumentUploadResponse(
            document_id=stored_document.document_id,
            original_file_name=stored_document.original_file_name,
            file_type=stored_document.file_type,
            file_size=stored_document.file_size,
            status=document.status,
        ),
        message="文件已安全落盘，等待后续解析",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/documents/{document_id}/parse",
    response_model=ApiResponse[DocumentParseResponse],
    summary="按文档 ID 解析已上传文件",
)
def parse_document(
    document_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentParseResponse]:
    document, parsed_document = parse_document_by_id(db, document_id)
    return success(
        DocumentParseResponse(
            document_id=document.document_id,
            status=document.status,
            parser_name=document.parser_name or parsed_document.parser_name,
            parsed_segment_count=document.parsed_segment_count,
            # 仅返回少量预览，完整原文段已经落到数据库，不能随大文件一次性返回。
            preview_segments=parsed_document.segments[:5],
        ),
        message="文档解析完成",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/documents/{document_id}/chunks",
    response_model=ApiResponse[DocumentChunkResponse],
    summary="按文档 ID 生成检索文本切块",
)
def chunk_document(
    document_id: str,
    request_body: DocumentChunkRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentChunkResponse]:
    config = ChunkingConfig(**request_body.model_dump())
    document, chunks = chunk_document_by_id(db, document_id, config)
    return success(
        DocumentChunkResponse(
            document_id=document.document_id,
            chunk_status=document.chunk_status,
            chunk_count=document.chunk_count,
            chunk_config=request_body.model_dump(),
            # 大文档完整 chunk 必须分页查询，此处仅供调用方快速确认结果。
            preview_chunks=chunks[:3],
        ),
        message="文档切块完成",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/documents/{document_id}/versions",
    response_model=ApiResponse[CreateDocumentVersionResponse],
    summary="创建文档候选索引版本",
)
def create_document_version(
    document_id: str,
    request_body: CreateDocumentVersionRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[CreateDocumentVersionResponse]:
    version = create_candidate_document_version(
        db,
        document_id=document_id,
        change_note=request_body.change_note,
    )
    return success(
        CreateDocumentVersionResponse(
            document_id=version.document_id,
            version_id=version.version_id,
            version_number=version.version_number,
            status=version.status,
            change_note=request_body.change_note,
        ),
        message="候选文档版本已创建，尚未参与线上检索",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/documents/{document_id}/versions/{version_id}/chunks",
    response_model=ApiResponse[VersionChunkResponse],
    summary="按候选版本独立生成检索文本切块",
)
def chunk_document_version(
    document_id: str,
    version_id: str,
    request_body: DocumentChunkRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[VersionChunkResponse]:
    document = _find_document(db, document_id)
    config = ChunkingConfig(**request_body.model_dump())
    _, chunks = chunk_document_version_by_id(db, document, version_id, config)
    return success(
        VersionChunkResponse(
            document_id=document.document_id,
            version_id=version_id,
            status="chunked",
            chunk_count=len(chunks),
            chunk_config=request_body.model_dump(),
            preview_chunks=chunks[:3],
        ),
        message="候选版本切块完成，active 版本未受影响",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/embedding-similarity-test",
    response_model=ApiResponse[EmbeddingSimilarityTestResponse],
    summary="比较两段文本的 Embedding 余弦相似度",
)
def embedding_similarity_test(
    request_body: EmbeddingSimilarityTestRequest,
    request: Request,
) -> ApiResponse[EmbeddingSimilarityTestResponse]:
    model, vector_dimension, similarity = compare_text_embeddings(
        request_body.text_a,
        request_body.text_b,
    )
    return success(
        EmbeddingSimilarityTestResponse(
            embedding_model=model,
            vector_dimension=vector_dimension,
            cosine_similarity=round(similarity, 6),
        ),
        message="Embedding 相似度计算完成，向量未持久化",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/documents/{document_id}/in-memory-search",
    response_model=ApiResponse[InMemoryChunkSearchResponse],
    summary="Day19 内存向量 Top-K 检索演示",
)
def in_memory_search_document_chunks(
    document_id: str,
    request_body: InMemoryChunkSearchRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[InMemoryChunkSearchResponse]:
    model, vector_dimension, candidate_count, items = search_document_chunks_in_memory(
        db,
        document_id=document_id,
        question=request_body.question,
        top_k=request_body.top_k,
    )
    return success(
        InMemoryChunkSearchResponse(
            embedding_model=model,
            vector_dimension=vector_dimension,
            candidate_count=candidate_count,
            items=[item.model_copy(update={"score": round(item.score, 6)}) for item in items],
        ),
        message="内存 Top-K 检索完成，候选 chunk 向量未持久化",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/documents/{document_id}/vector-search",
    response_model=ApiResponse[MilvusChunkSearchResponse],
    summary="按文档 active 版本执行真实 Milvus 向量 Top-K 检索",
)
def search_document_chunks_by_milvus(
    document_id: str,
    request_body: MilvusChunkSearchRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[MilvusChunkSearchResponse]:
    model, vector_dimension, active_version_id, items = search_active_document_chunks(
        db,
        document_id=document_id,
        question=request_body.question,
        top_k=request_body.top_k,
    )
    return success(
        MilvusChunkSearchResponse(
            document_id=document_id,
            active_version_id=active_version_id,
            embedding_model=model,
            vector_dimension=vector_dimension,
            vector_collection=settings.milvus_collection_name,
            items=[item.model_copy(update={"score": round(item.score, 6)}) for item in items],
        ),
        message="Milvus Top-K 检索完成，已由 MySQL 回填 chunk 原文和来源",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/documents/{document_id}/rag-context-preview",
    response_model=ApiResponse[RagContextPreviewResponse],
    summary="开发验证：预览 RAG 检索资料包与引用编号",
)
def preview_rag_context(
    document_id: str,
    request_body: RagContextPreviewRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[RagContextPreviewResponse]:
    model, vector_dimension, active_version_id, items = search_active_document_chunks(
        db,
        document_id=document_id,
        question=request_body.question,
        top_k=request_body.retrieval_top_k,
    )
    context_result = build_rag_context(
        items,
        max_context_characters=request_body.max_context_characters,
    )
    return success(
        RagContextPreviewResponse(
            document_id=document_id,
            active_version_id=active_version_id,
            embedding_model=model,
            vector_dimension=vector_dimension,
            retrieved_chunk_count=len(items),
            included_chunk_count=len(context_result.references),
            omitted_chunk_count=context_result.omitted_chunk_count,
            context_char_count=len(context_result.context),
            references=[
                reference.model_copy(update={"score": round(reference.score, 6)})
                for reference in context_result.references
            ],
            context=context_result.context,
        ),
        message="RAG 资料包已组装完成，尚未调用聊天模型",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/documents/{document_id}/rag-answer",
    response_model=ApiResponse[RagAnswerResponse],
    summary="基于 active 知识库版本生成带引用的 RAG 回答",
)
def answer_with_rag(
    document_id: str,
    request_body: RagAnswerRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[RagAnswerResponse]:
    model, _, active_version_id, items = search_active_document_chunks(
        db,
        document_id=document_id,
        question=request_body.question,
        top_k=request_body.retrieval_top_k,
    )
    context_result = build_rag_context(
        items,
        max_context_characters=request_body.max_context_characters,
    )
    start_time = time.perf_counter()
    try:
        generation = generate_rag_answer(
            question=request_body.question,
            context_result=context_result,
        )
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        # 无资料兜底没有发生模型调用，不应伪造一条成功的模型成本日志。
        if generation.model:
            create_call_log(
                db,
                call_type="rag_knowledge_answer",
                trace_id=request.state.trace_id,
                model=generation.model,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                total_tokens=generation.total_tokens,
                cost_ms=cost_ms,
                status="success",
            )
        return success(
            RagAnswerResponse(
                answer=generation.answer,
                references=[
                    reference.model_copy(update={"score": round(reference.score, 6)})
                    for reference in generation.references
                ],
                document_id=document_id,
                active_version_id=active_version_id,
                retrieved_chunk_count=len(items),
                included_chunk_count=len(context_result.references),
                omitted_chunk_count=context_result.omitted_chunk_count,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                total_tokens=generation.total_tokens,
                cost_ms=cost_ms,
            ),
            message="RAG 回答生成完成，引用来源已校验",
            trace_id=request.state.trace_id,
        )
    except ModelCallException as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        create_call_log(
            db,
            call_type="rag_knowledge_answer",
            trace_id=request.state.trace_id,
            model=model,
            cost_ms=cost_ms,
            status="error",
            error_type=exc.error_type,
            error_message=exc.message,
        )
        raise


@router.post(
    "/sessions/{session_id}/rag-answer",
    response_model=ApiResponse[SessionRagAnswerResponse],
    summary="在会话中生成带持久化引用的 RAG 回答",
)
def answer_session_with_rag_endpoint(
    session_id: str,
    request_body: SessionRagAnswerRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[SessionRagAnswerResponse]:
    result = answer_session_with_rag(
        db,
        session_id=session_id,
        document_id=request_body.document_id,
        message=request_body.message,
        trace_id=request.state.trace_id,
        retrieval_top_k=request_body.retrieval_top_k,
        max_context_characters=request_body.max_context_characters,
    )
    return success(
        SessionRagAnswerResponse(
            session_id=session_id,
            user_message_id=result.user_message_id,
            assistant_message_id=result.assistant_message_id,
            answer=result.answer,
            references=[
                reference.model_copy(update={"score": round(reference.score, 6)})
                for reference in result.references
            ],
            document_id=result.document_id,
            active_version_id=result.active_version_id,
            retrieved_chunk_count=result.retrieved_chunk_count,
            included_chunk_count=result.included_chunk_count,
            omitted_chunk_count=result.omitted_chunk_count,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_ms=result.cost_ms,
        ),
        message="会话 RAG 回答生成完成，消息和引用已持久化",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/sessions/{session_id}/rag-answer/async",
    response_model=ApiResponse[AsyncRagTaskSubmitResponse],
    summary="提交异步会话 RAG 任务",
)
def submit_async_session_rag(
    session_id: str,
    request_body: AsyncSessionRagTaskRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[AsyncRagTaskSubmitResponse]:
    # 先验证会话，避免 MySQL 已落任务而 Worker 才发现 session_id 无效。
    get_session(db, session_id)
    task, outbox_event = create_async_session_rag_task(
        db,
        session_id=session_id,
        input_text=request_body.message,
        trace_id=request.state.trace_id,
        model=settings.dashscope_model,
        document_id=request_body.document_id,
        retrieval_top_k=request_body.retrieval_top_k,
        max_context_characters=request_body.max_context_characters,
        max_retries=settings.async_task_max_retries,
    )
    # RabbitMQ 暂不可用时保持 pending，Beat 会继续扫描 Outbox 并补投。
    dispatch_outbox_event(db, outbox_event.event_id)
    return success(
        AsyncRagTaskSubmitResponse(task_id=task.task_id, status=task.status),
        message="异步会话 RAG 任务已提交",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/sessions/{session_id}/messages/{assistant_message_id}/rag-references",
    response_model=ApiResponse[RagAnswerReferenceListResponse],
    summary="查询会话某条 RAG 回答实际引用的知识来源",
)
def get_session_rag_answer_references(
    session_id: str,
    assistant_message_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[RagAnswerReferenceListResponse]:
    references = list_session_rag_answer_references(
        db,
        session_id=session_id,
        assistant_message_id=assistant_message_id,
    )
    return success(
        RagAnswerReferenceListResponse(
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            items=[
                reference.model_copy(update={"score": round(reference.score, 6)})
                for reference in references
            ],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/document-versions/{version_id}/vector-index",
    response_model=ApiResponse[DocumentVersionIndexResponse],
    summary="为指定文档版本构建 Milvus 向量索引",
)
def build_document_version_vector_index(
    version_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentVersionIndexResponse]:
    version = build_version_vector_index(db, version_id)
    return success(
        DocumentVersionIndexResponse(
            version_id=version.version_id,
            document_id=version.document_id,
            status=version.status,
            chunk_count=version.chunk_count,
            vector_count=version.vector_count,
            embedding_model=version.embedding_model or settings.dashscope_embedding_model,
            embedding_dimension=version.embedding_dimension or 0,
            vector_collection=version.vector_collection or settings.milvus_collection_name,
        ),
        message="文档版本向量索引构建完成，等待验证后切换",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/document-versions/{version_id}/activate",
    response_model=ApiResponse[ActivateDocumentVersionResponse],
    summary="校验后切换知识库文档的 active 版本",
)
def activate_document_version_endpoint(
    version_id: str,
    request_body: ActivateDocumentVersionRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[ActivateDocumentVersionResponse]:
    document, version, previous_version_id = activate_document_version(
        db,
        version_id=version_id,
        activated_by=request_body.activated_by,
        activation_note=request_body.activation_note,
    )
    return success(
        ActivateDocumentVersionResponse(
            document_id=document.document_id,
            active_version_id=document.active_version_id or version.version_id,
            previous_version_id=previous_version_id,
            status=version.status,
            activated_at=(version.activated_at or version.updated_at).isoformat(timespec="seconds"),
        ),
        message="文档索引版本已切换为 active",
        trace_id=request.state.trace_id,
    )
