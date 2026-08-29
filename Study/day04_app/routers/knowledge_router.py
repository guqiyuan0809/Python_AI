"""知识库文件上传、索引与 RAG 问答接口。"""

import time

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
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
    LlamaIndexLawRetrievalRequest,
    LlamaIndexLawRetrievalResponse,
    LlamaIndexLawSourceNode,
    LlamaIndexChunkPreviewRequest,
    LlamaIndexChunkPreviewResponse,
    LlamaIndexChunkPreviewNode,
    DocumentVersionChunkSearchResponse,
    CreateRetrievalEvalDatasetRequest,
    CreateRetrievalEvalSampleRequest,
    DocumentSegmentItem,
    DocumentSegmentPageResponse,
    RetrievalEvalDatasetItem,
    RetrievalEvalDatasetPageResponse,
    RetrievalEvalSampleItem,
    RetrievalEvalSamplePageResponse,
    RetrievalEvalRunItem,
    RetrievalEvalRunPageResponse,
    RetrievalEvalCaseResultItem,
    RetrievalEvalCaseResultPageResponse,
    RetrievalEvalRetrievedChunkItem,
    RunRetrievalEvalRequest,
    RagContextPreviewRequest,
    RagContextPreviewResponse,
    RagAnswerRequest,
    RagAnswerResponse,
    MultiDocumentRagRequest,
    MultiDocumentRagContextPreviewResponse,
    MultiDocumentRagAnswerResponse,
    LlamaIndexRagAnswerResponse,
    RagAnswerReferenceListResponse,
    AsyncRagTaskSubmitResponse,
    AsyncSessionRagTaskRequest,
    SessionRagAnswerRequest,
    SessionRagAnswerResponse,
    CreateDocumentVersionRequest,
    CreateDocumentVersionResponse,
    VersionChunkResponse,
    ParentChildChunkRequest,
    ParentChildChunkResponse,
    ContextualIndexBuildRequest,
    ContextualIndexTaskSubmitResponse,
)
from day04_app.services.knowledge_embedding_service import compare_text_embeddings
from day04_app.services.knowledge_in_memory_search_service import search_document_chunks_in_memory
from day04_app.services.knowledge_milvus_search_service import (
    search_active_document_chunks,
    search_document_version_chunks_for_validation,
)
from day04_app.services.llamaindex_law_retrieval_service import (
    retrieve_active_document_as_llamaindex_nodes,
)
from day04_app.services.llamaindex_rag_query_service import (
    answer_active_document_with_llamaindex,
    prepare_governed_llamaindex_rag,
    preview_governed_llamaindex_context,
)
from day04_app.services.llamaindex_multi_document_rag_service import (
    KnowledgeDomain,
    answer_multi_document_with_llamaindex,
    preview_multi_document_llamaindex_context,
)
from day04_app.services.llamaindex_document_service import build_llamaindex_nodes_from_segments
from day04_app.services.session_rag_service import (
    answer_session_with_rag,
    list_session_rag_answer_references,
)
from day04_app.services.async_task_service import (
    create_async_session_rag_task,
    create_async_contextual_index_task,
)
from day04_app.services.outbox_dispatcher import dispatch_outbox_event
from day04_app.services.session_service import get_session
from day04_app.services.knowledge_retrieval_eval_service import (
    create_retrieval_eval_dataset,
    create_retrieval_eval_sample,
    list_document_segments,
    list_retrieval_eval_datasets,
    list_retrieval_eval_case_results,
    list_retrieval_eval_runs,
    list_retrieval_eval_samples,
    run_retrieval_evaluation,
)
from day04_app.services.call_log_service import create_call_log
from day04_app.common.exceptions import ModelCallException
from day04_app.security.dependencies import require_permissions
from day04_app.security.permissions import (
    PERMISSION_AI_EVAL_RUN,
    PERMISSION_AI_INVOKE,
    PERMISSION_KNOWLEDGE_READ,
    PERMISSION_KNOWLEDGE_WRITE,
)
from day04_app.services.knowledge_vector_index_service import build_version_vector_index
from day04_app.services.knowledge_document_version_service import activate_document_version
from day04_app.services.knowledge_document_chunk_service import (
    _find_document,
    chunk_document_by_id,
    chunk_document_version_by_id,
    chunk_document_version_with_parent_child_by_id,
)
from day04_app.services.knowledge_document_rebuild_service import create_candidate_document_version
from day04_app.services.knowledge_document_parse_service import parse_document_by_id
from day04_app.services.document_storage_service import delete_stored_document, save_uploaded_document
from day04_app.services.knowledge_document_service import create_uploaded_document
from day04_app.services.text_chunker_service import ChunkingConfig, ParentChildChunkingConfig
from settings import settings


router = APIRouter(
    prefix="/api/knowledge",
    tags=["知识库"],
    dependencies=[
        Depends(require_permissions(PERMISSION_AI_INVOKE, PERMISSION_KNOWLEDGE_READ))
    ],
)


def _resolve_rag_score_threshold(request_score_threshold: float | None) -> float | None:
    """请求未显式传阈值时使用服务端默认配置，便于统一线上兜底策略。"""
    return (
        request_score_threshold
        if request_score_threshold is not None
        else settings.rag_min_relevance_score
    )


def _to_multi_document_domains(request_body: MultiDocumentRagRequest) -> tuple[
    list[KnowledgeDomain], dict[str, tuple[str, ...]]
]:
    """将 HTTP DTO 转成框架路由契约，不让 Router 自己产生 document_id。

    当前是可观察的教学接口，因此传入的是“业务层已经准入的领域”；落地金汤令时
    这个函数的上游应改为根据 Java 可信身份里的 tenant/park/enterprise 数据范围查询
    ``knowledge_domain_document``，而不是直接使用前端请求中的 document_ids。
    """

    domains = [
        KnowledgeDomain(
            domain_id=domain.domain_id,
            description=domain.description,
            document_ids=tuple(domain.document_ids),
        )
        for domain in request_body.domains
    ]
    keywords = {
        domain.domain_id: tuple(domain.keywords)
        for domain in request_body.domains
    }
    return domains, keywords


def _to_retrieval_eval_dataset_item(dataset) -> RetrievalEvalDatasetItem:
    return RetrievalEvalDatasetItem(
        dataset_id=dataset.dataset_id,
        dataset_name=dataset.dataset_name,
        dataset_version=dataset.dataset_version,
        document_id=dataset.document_id,
        description=dataset.description,
        sample_count=dataset.sample_count,
        status=dataset.status,
        created_by=dataset.created_by,
        created_at=dataset.created_at.isoformat(timespec="seconds"),
        updated_at=dataset.updated_at.isoformat(timespec="seconds"),
    )


def _to_retrieval_eval_sample_item(sample) -> RetrievalEvalSampleItem:
    import json

    return RetrievalEvalSampleItem(
        sample_id=sample.sample_id,
        dataset_id=sample.dataset_id,
        question=sample.question,
        sample_type=sample.sample_type,
        expected_answerable=bool(sample.expected_answerable),
        expected_segment_indexes=json.loads(sample.expected_segment_indexes_json),
        expected_note=sample.expected_note,
        status=sample.status,
        created_by=sample.created_by,
        created_at=sample.created_at.isoformat(timespec="seconds"),
        updated_at=sample.updated_at.isoformat(timespec="seconds"),
    )


def _to_retrieval_eval_run_item(run) -> RetrievalEvalRunItem:
    return RetrievalEvalRunItem(
        use_reranker=bool(getattr(run, "use_reranker", 0)),
        rerank_top_n=getattr(run, "rerank_top_n", None),
        reranker_model=getattr(run, "reranker_model", None),
        run_id=run.run_id,
        dataset_id=run.dataset_id,
        document_id=run.document_id,
        document_version_id=run.document_version_id,
        retrieval_top_k=run.retrieval_top_k,
        score_threshold=run.score_threshold,
        embedding_model=run.embedding_model,
        vector_dimension=run.vector_dimension,
        status=run.status,
        sample_count=run.sample_count,
        success_count=run.success_count,
        error_count=run.error_count,
        answerable_sample_count=run.answerable_sample_count,
        answerable_hit_count=run.answerable_hit_count,
        total_expected_segment_count=getattr(run, "total_expected_segment_count", 0),
        total_hit_segment_count=getattr(run, "total_hit_segment_count", 0),
        total_retrieved_chunk_count=getattr(run, "total_retrieved_chunk_count", 0),
        total_relevant_retrieved_chunk_count=getattr(
            run,
            "total_relevant_retrieved_chunk_count",
            0,
        ),
        hit_at_k=getattr(run, "hit_at_k", None),
        recall_at_k=run.recall_at_k,
        precision_at_k=getattr(run, "precision_at_k", None),
        mrr_at_k=run.mrr_at_k,
        no_answer_sample_count=run.no_answer_sample_count,
        no_answer_false_positive_count=run.no_answer_false_positive_count,
        no_answer_false_positive_rate=run.no_answer_false_positive_rate,
        no_answer_avg_top_score=run.no_answer_avg_top_score,
        elapsed_ms=run.elapsed_ms,
        error_message=run.error_message,
        created_by=run.created_by,
        started_at=run.started_at.isoformat(timespec="seconds"),
        finished_at=(
            run.finished_at.isoformat(timespec="seconds") if run.finished_at else None
        ),
    )


def _to_retrieval_eval_case_result_item(case_result) -> RetrievalEvalCaseResultItem:
    import json

    retrieved_chunks = json.loads(case_result.retrieved_chunks_json)
    return RetrievalEvalCaseResultItem(
        case_result_id=case_result.case_result_id,
        run_id=case_result.run_id,
        sample_id=case_result.sample_id,
        question=case_result.question_snapshot,
        sample_type=case_result.sample_type_snapshot,
        expected_answerable=bool(case_result.expected_answerable_snapshot),
        expected_segment_indexes=json.loads(case_result.expected_segment_indexes_json),
        retrieved_segment_indexes=json.loads(case_result.retrieved_segment_indexes_json),
        retrieved_chunks=[
            RetrievalEvalRetrievedChunkItem.model_validate(chunk)
            for chunk in retrieved_chunks
        ],
        first_hit_rank=case_result.first_hit_rank,
        is_hit=bool(case_result.is_hit) if case_result.is_hit is not None else None,
        hit_segment_count=getattr(case_result, "hit_segment_count", None),
        expected_segment_count=getattr(case_result, "expected_segment_count", None),
        relevant_retrieved_chunk_count=getattr(
            case_result,
            "relevant_retrieved_chunk_count",
            None,
        ),
        precision_at_k=getattr(case_result, "precision_at_k", None),
        top_score=case_result.top_score,
        is_false_positive=(
            bool(case_result.is_false_positive)
            if case_result.is_false_positive is not None
            else None
        ),
        elapsed_ms=case_result.elapsed_ms,
        status=case_result.status,
        error_type=case_result.error_type,
        error_message=case_result.error_message,
        created_at=case_result.created_at.isoformat(timespec="seconds"),
    )


@router.post(
    "/documents/upload",
    response_model=ApiResponse[DocumentUploadResponse],
    summary="安全上传知识库文件",
    dependencies=[Depends(require_permissions(PERMISSION_KNOWLEDGE_WRITE))],
)
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


@router.get(
    "/documents/{document_id}/segments",
    response_model=ApiResponse[DocumentSegmentPageResponse],
    summary="分页查询原始文本段，供检索评测人工标注使用",
)
def get_document_segments(
    document_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(50, ge=1, le=200, description="每页原始段数量"),
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentSegmentPageResponse]:
    segments, total = list_document_segments(
        db,
        document_id=document_id,
        page=page,
        page_size=page_size,
    )
    return success(
        DocumentSegmentPageResponse(
            document_id=document_id,
            total=total,
            page=page,
            page_size=page_size,
            items=[
                DocumentSegmentItem(
                    segment_index=segment.segment_index,
                    content=segment.content,
                    location=segment.location,
                )
                for segment in segments
            ],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/retrieval-eval-datasets",
    response_model=ApiResponse[RetrievalEvalDatasetItem],
    summary="创建 RAG 检索评测数据集",
    dependencies=[Depends(require_permissions(PERMISSION_AI_EVAL_RUN))],
)
def create_rag_retrieval_eval_dataset(
    request_body: CreateRetrievalEvalDatasetRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[RetrievalEvalDatasetItem]:
    dataset = create_retrieval_eval_dataset(
        db,
        dataset_name=request_body.dataset_name,
        dataset_version=request_body.dataset_version,
        document_id=request_body.document_id,
        description=request_body.description,
        created_by=request_body.created_by,
    )
    return success(
        _to_retrieval_eval_dataset_item(dataset),
        message="RAG 检索评测数据集已创建，当前为 draft",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/retrieval-eval-datasets",
    response_model=ApiResponse[RetrievalEvalDatasetPageResponse],
    summary="分页查询 RAG 检索评测数据集",
)
def get_rag_retrieval_eval_datasets(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    document_id: str | None = Query(None, description="可选知识库文档业务 ID"),
    status: str | None = Query(None, description="可选数据集状态，例如 draft/active"),
    db: Session = Depends(get_db),
) -> ApiResponse[RetrievalEvalDatasetPageResponse]:
    datasets, total = list_retrieval_eval_datasets(
        db,
        page=page,
        page_size=page_size,
        document_id=document_id,
        status=status,
    )
    return success(
        RetrievalEvalDatasetPageResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[_to_retrieval_eval_dataset_item(dataset) for dataset in datasets],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/retrieval-eval-datasets/{dataset_id}/samples",
    response_model=ApiResponse[RetrievalEvalSampleItem],
    summary="新增一条 RAG 检索评测样本",
    dependencies=[Depends(require_permissions(PERMISSION_AI_EVAL_RUN))],
)
def create_rag_retrieval_eval_sample(
    dataset_id: str,
    request_body: CreateRetrievalEvalSampleRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[RetrievalEvalSampleItem]:
    sample = create_retrieval_eval_sample(
        db,
        dataset_id=dataset_id,
        question=request_body.question,
        sample_type=request_body.sample_type,
        expected_answerable=request_body.expected_answerable,
        expected_segment_indexes=request_body.expected_segment_indexes,
        expected_note=request_body.expected_note,
        created_by=request_body.created_by,
    )
    return success(
        _to_retrieval_eval_sample_item(sample),
        message="RAG 检索评测样本已标注",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/retrieval-eval-datasets/{dataset_id}/samples",
    response_model=ApiResponse[RetrievalEvalSamplePageResponse],
    summary="分页查询 RAG 检索评测样本",
)
def get_rag_retrieval_eval_samples(
    dataset_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(50, ge=1, le=200, description="每页数量"),
    db: Session = Depends(get_db),
) -> ApiResponse[RetrievalEvalSamplePageResponse]:
    samples, total = list_retrieval_eval_samples(
        db,
        dataset_id=dataset_id,
        page=page,
        page_size=page_size,
    )
    return success(
        RetrievalEvalSamplePageResponse(
            dataset_id=dataset_id,
            total=total,
            page=page,
            page_size=page_size,
            items=[_to_retrieval_eval_sample_item(sample) for sample in samples],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/retrieval-eval-runs",
    response_model=ApiResponse[RetrievalEvalRunItem],
    summary="运行指定文档版本的 RAG 检索评测",
    dependencies=[Depends(require_permissions(PERMISSION_AI_EVAL_RUN))],
)
def run_rag_retrieval_evaluation(
    request_body: RunRetrievalEvalRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[RetrievalEvalRunItem]:
    run = run_retrieval_evaluation(
        db,
        dataset_id=request_body.dataset_id,
        document_version_id=request_body.document_version_id,
        retrieval_top_k=request_body.retrieval_top_k,
        score_threshold=request_body.score_threshold,
        created_by=request_body.created_by,
        use_reranker=request_body.use_reranker,
        rerank_top_n=request_body.rerank_top_n,
        sample_limit=request_body.sample_limit,
    )
    return success(
        _to_retrieval_eval_run_item(run),
        message="RAG 检索评测完成，已保存运行汇总和样本明细",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/retrieval-eval-runs",
    response_model=ApiResponse[RetrievalEvalRunPageResponse],
    summary="分页查询 RAG 检索评测运行记录",
)
def get_rag_retrieval_eval_runs(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    dataset_id: str | None = Query(None, description="可选数据集业务 ID"),
    document_version_id: str | None = Query(None, description="可选被测文档版本业务 ID"),
    db: Session = Depends(get_db),
) -> ApiResponse[RetrievalEvalRunPageResponse]:
    runs, total = list_retrieval_eval_runs(
        db,
        page=page,
        page_size=page_size,
        dataset_id=dataset_id,
        document_version_id=document_version_id,
    )
    return success(
        RetrievalEvalRunPageResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[_to_retrieval_eval_run_item(run) for run in runs],
        ),
        trace_id=request.state.trace_id,
    )


@router.get(
    "/retrieval-eval-runs/{run_id}/cases",
    response_model=ApiResponse[RetrievalEvalCaseResultPageResponse],
    summary="分页查询一次 RAG 检索评测的样本明细",
)
def get_rag_retrieval_eval_case_results(
    run_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(50, ge=1, le=200, description="每页数量"),
    db: Session = Depends(get_db),
) -> ApiResponse[RetrievalEvalCaseResultPageResponse]:
    case_results, total = list_retrieval_eval_case_results(
        db,
        run_id=run_id,
        page=page,
        page_size=page_size,
    )
    return success(
        RetrievalEvalCaseResultPageResponse(
            run_id=run_id,
            total=total,
            page=page,
            page_size=page_size,
            items=[
                _to_retrieval_eval_case_result_item(case_result)
                for case_result in case_results
            ],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/documents/{document_id}/parse",
    response_model=ApiResponse[DocumentParseResponse],
    summary="按文档 ID 解析已上传文件",
    dependencies=[Depends(require_permissions(PERMISSION_KNOWLEDGE_WRITE))],
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
    dependencies=[Depends(require_permissions(PERMISSION_KNOWLEDGE_WRITE))],
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
    dependencies=[Depends(require_permissions(PERMISSION_KNOWLEDGE_WRITE))],
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
    dependencies=[Depends(require_permissions(PERMISSION_KNOWLEDGE_WRITE))],
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
    "/documents/{document_id}/versions/{version_id}/parent-child-chunks",
    response_model=ApiResponse[ParentChildChunkResponse],
    summary="为候选版本生成父子检索切块",
    dependencies=[Depends(require_permissions(PERMISSION_KNOWLEDGE_WRITE))],
)
def chunk_document_version_parent_child(
    document_id: str,
    version_id: str,
    request_body: ParentChildChunkRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[ParentChildChunkResponse]:
    document = _find_document(db, document_id)
    config = ParentChildChunkingConfig(**request_body.model_dump())
    parents, children = chunk_document_version_with_parent_child_by_id(
        db,
        document,
        version_id,
        config,
    )
    return success(
        ParentChildChunkResponse(
            document_id=document.document_id,
            version_id=version_id,
            status="chunked",
            parent_chunk_count=len(parents),
            child_chunk_count=len(children),
            chunk_config={"strategy": "parent_child_contextual", **request_body.model_dump()},
            preview_parent_chunks=parents[:2],
            preview_child_chunks=children[:3],
        ),
        message="候选版本父子切块完成，等待上下文化索引",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/document-versions/{version_id}/contextual-index/async",
    response_model=ApiResponse[ContextualIndexTaskSubmitResponse],
    summary="异步生成父子切块上下文并构建向量索引",
    dependencies=[Depends(require_permissions(PERMISSION_KNOWLEDGE_WRITE))],
)
def submit_contextual_index_task(
    version_id: str,
    request_body: ContextualIndexBuildRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[ContextualIndexTaskSubmitResponse]:
    task, outbox_event = create_async_contextual_index_task(
        db,
        version_id=version_id,
        trace_id=request.state.trace_id,
        context_model=request_body.context_model,
        context_max_tokens=request_body.context_max_tokens,
        max_retries=settings.async_task_max_retries,
    )
    dispatch_outbox_event(db, outbox_event.event_id)
    return success(
        ContextualIndexTaskSubmitResponse(
            task_id=task.task_id,
            version_id=version_id,
            status=task.status,
        ),
        message="上下文化索引任务已提交",
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
        use_reranker=request_body.use_reranker,
        rerank_top_n=request_body.rerank_top_n,
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
    "/documents/{document_id}/llamaindex-retrieval-preview",
    response_model=ApiResponse[LlamaIndexLawRetrievalResponse],
    summary="Day31：使用 LlamaIndex Retriever 预览法规知识节点",
)
def preview_llamaindex_law_retrieval(
    document_id: str,
    request_body: LlamaIndexLawRetrievalRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[LlamaIndexLawRetrievalResponse]:
    """将现有 active 版本的 Milvus 命中适配为 LlamaIndex NodeWithScore。

    此接口故意不调用聊天模型：先让你观察“框架节点”与现有 chunk、版本和来源如何一一对应。
    """

    start_time = time.perf_counter()
    result = retrieve_active_document_as_llamaindex_nodes(
        db,
        document_id=document_id,
        question=request_body.question,
        top_k=request_body.top_k,
        use_reranker=request_body.use_reranker,
        rerank_top_n=request_body.rerank_top_n,
        trace_id=request.state.trace_id,
    )

    create_call_log(
        db,
        call_type="llamaindex_law_rag",
        stage="llamaindex_retrieval_adapter",
        trace_id=request.state.trace_id,
        model=result.embedding_model,
        cost_ms=round((time.perf_counter() - start_time) * 1000),
        detail={
            "framework": "llamaindex",
            "retrieval_backend": "project_milvus",
            "document_id": document_id,
            "version_id": result.active_version_id,
            "node_count": len(result.nodes),
            "use_reranker": request_body.use_reranker,
            "rerank_top_n": request_body.rerank_top_n if request_body.use_reranker else None,
        },
    )
    return success(
        LlamaIndexLawRetrievalResponse(
            document_id=document_id,
            active_version_id=result.active_version_id,
            embedding_model=result.embedding_model,
            vector_dimension=result.vector_dimension,
            node_count=len(result.nodes),
            nodes=[
                LlamaIndexLawSourceNode(
                    node_id=node.node.node_id,
                    document_id=str(node.node.metadata["document_id"]),
                    version_id=str(node.node.metadata["version_id"]),
                    chunk_index=int(node.node.metadata["chunk_index"]),
                    parent_chunk_id=node.node.metadata.get("parent_chunk_id"),
                    score=float(node.score or 0.0),
                    vector_score=node.node.metadata.get("vector_score"),
                    rerank_score=node.node.metadata.get("rerank_score"),
                    content=node.node.get_content(),
                    source_locations=list(node.node.metadata.get("source_locations") or []),
                )
                for node in result.nodes
            ],
        ),
        message="LlamaIndex Retriever 节点转换完成；仍复用项目的 Milvus、版本与来源治理",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/documents/{document_id}/llamaindex-chunk-preview",
    response_model=ApiResponse[LlamaIndexChunkPreviewResponse],
    summary="Day31：使用 LlamaIndex SentenceSplitter 预览法规节点",
)
def preview_llamaindex_document_chunks(
    document_id: str,
    request_body: LlamaIndexChunkPreviewRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[LlamaIndexChunkPreviewResponse]:
    """读取已有原始段，用 LlamaIndex 切块并返回元数据；不改变文档版本状态。"""

    from sqlalchemy import select

    from day04_app.models import KnowledgeDocumentSegment
    from day04_app.schemas.knowledge_schema import ParsedDocumentSegment

    segments = list(
        db.scalars(
            select(KnowledgeDocumentSegment)
            .where(KnowledgeDocumentSegment.document_id == document_id)
            .order_by(KnowledgeDocumentSegment.segment_index.asc())
        )
    )
    if not segments:
        from day04_app.common.exceptions import BusinessException

        raise BusinessException(code=40067, message="文档尚未产生原始文本段，不能执行 LlamaIndex 切块预览")

    result = build_llamaindex_nodes_from_segments(
        document_id=document_id,
        segments=[
            ParsedDocumentSegment(
                segment_index=segment.segment_index,
                text=segment.content,
                location=segment.location,
            )
            for segment in segments
        ],
        chunk_size=request_body.chunk_size,
        chunk_overlap=request_body.chunk_overlap,
    )
    create_call_log(
        db,
        call_type="llamaindex_law_rag",
        stage="llamaindex_document_split",
        trace_id=request.state.trace_id,
        cost_ms=0,
        detail={
            "framework": "llamaindex",
            "document_id": document_id,
            "source_segment_count": result.source_segment_count,
            "node_count": len(result.nodes),
            "chunk_size": result.chunk_size,
            "chunk_overlap": result.chunk_overlap,
            "persisted": False,
        },
    )
    preview_nodes = result.nodes[:50]
    return success(
        LlamaIndexChunkPreviewResponse(
            document_id=document_id,
            source_segment_count=result.source_segment_count,
            node_count=len(result.nodes),
            chunk_size=result.chunk_size,
            chunk_overlap=result.chunk_overlap,
            nodes=[
                LlamaIndexChunkPreviewNode(
                    node_id=node.node_id,
                    content=node.get_content(),
                    document_id=str(node.metadata.get("document_id", document_id)),
                    source_segment_index=(
                        int(node.metadata["segment_index"])
                        if node.metadata.get("segment_index") is not None
                        else None
                    ),
                    source_location=node.metadata.get("source_location"),
                    start_char_idx=node.start_char_idx,
                    end_char_idx=node.end_char_idx,
                )
                for node in preview_nodes
            ],
        ),
        message="LlamaIndex 文档节点切块预览完成，未修改 MySQL 或 Milvus",
        trace_id=request.state.trace_id,
    )
@router.post(
    "/documents/{document_id}/versions/{version_id}/vector-search",
    response_model=ApiResponse[DocumentVersionChunkSearchResponse],
    summary="发布前验证指定文档版本的 Milvus 检索结果",
)
def search_document_version_chunks_by_milvus(
    document_id: str,
    version_id: str,
    request_body: MilvusChunkSearchRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentVersionChunkSearchResponse]:
    # 候选版本只允许验证检索，不能通过本接口修改 document.active_version_id。
    model, vector_dimension, verified_version_id, items = search_document_version_chunks_for_validation(
        db,
        document_id=document_id,
        version_id=version_id,
        question=request_body.question,
        top_k=request_body.top_k,
        use_reranker=request_body.use_reranker,
        rerank_top_n=request_body.rerank_top_n,
    )
    return success(
        DocumentVersionChunkSearchResponse(
            document_id=document_id,
            version_id=verified_version_id,
            embedding_model=model,
            vector_dimension=vector_dimension,
            vector_collection=settings.milvus_collection_name,
            items=[item.model_copy(update={"score": round(item.score, 6)}) for item in items],
        ),
        message="文档版本 Milvus 验证检索完成，未修改线上 active 版本",
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
    """预览正式 LlamaIndex QueryEngine 在调用模型前会采用的资料节点。

    这个接口故意不自己调用 ``search_active_document_chunks + build_rag_context``。
    那样虽然能展示一份资料包，却可能和正式回答的父块去重、分数门禁、上下文预算
    产生偏差。现在预览与正式回答共用 ``Retriever + NodePostprocessor``；区别仅是
    预览在模型调用前停止，因此不会消耗聊天模型 Token，也不会写入回答记录。
    """

    score_threshold = _resolve_rag_score_threshold(request_body.score_threshold)
    preparation = prepare_governed_llamaindex_rag(
        db,
        document_id=document_id,
        retrieval_top_k=request_body.retrieval_top_k,
        max_context_characters=request_body.max_context_characters,
        use_reranker=request_body.use_reranker,
        rerank_top_n=request_body.rerank_top_n,
        score_threshold=score_threshold,
        trace_id=request.state.trace_id,
    )
    preview = preview_governed_llamaindex_context(
        preparation,
        question=request_body.question,
    )
    return success(
        RagContextPreviewResponse(
            document_id=document_id,
            active_version_id=preview.retrieval.active_version_id,
            embedding_model=preview.retrieval.embedding_model,
            vector_dimension=preview.retrieval.vector_dimension,
            retrieved_chunk_count=len(preview.retrieval.nodes),
            included_chunk_count=len(preview.references),
            omitted_chunk_count=preview.omitted_node_count,
            top_score=(
                round(preview.top_score, 6)
                if preview.top_score is not None
                else None
            ),
            score_threshold=preview.score_threshold,
            rejected_by_score_threshold=preview.rejected_by_score_threshold,
            context_char_count=len(preview.context),
            references=[
                reference.model_copy(update={"score": round(reference.score, 6)})
                for reference in preview.references
            ],
            context=preview.context,
        ),
        message="LlamaIndex RAG 资料节点已完成治理筛选，尚未调用聊天模型",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/multi-document-rag-context-preview",
    response_model=ApiResponse[MultiDocumentRagContextPreviewResponse],
    summary="Day31：预览 LlamaIndex 多文档知识域路由与跨文档资料包",
)
def preview_multi_document_rag_context(
    request_body: MultiDocumentRagRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[MultiDocumentRagContextPreviewResponse]:
    """只执行路由、全局检索和资料治理，不调用聊天模型。

    建议先调用本接口确认：问题被路由到了哪个业务领域、该领域允许哪些文档、每篇文档
    使用哪个 active 版本、最终有哪些跨文档证据会进入 Prompt。只有这些事实合理时再调
    正式回答接口，避免把“路由错误”误以为是模型回答质量问题。
    """

    domains, domain_keywords = _to_multi_document_domains(request_body)
    start_time = time.perf_counter()
    preview = preview_multi_document_llamaindex_context(
        db,
        domains=domains,
        domain_keywords=domain_keywords,
        default_domain_id=request_body.default_domain_id,
        question=request_body.question,
        retrieval_top_k=request_body.retrieval_top_k,
        max_context_characters=request_body.max_context_characters,
        use_reranker=request_body.use_reranker,
        rerank_top_n=request_body.rerank_top_n,
        score_threshold=_resolve_rag_score_threshold(request_body.score_threshold),
        trace_id=request.state.trace_id,
    )
    create_call_log(
        db,
        call_type="multi_document_rag",
        stage="llamaindex_knowledge_domain_route",
        trace_id=request.state.trace_id,
        cost_ms=round((time.perf_counter() - start_time) * 1000),
        detail={
            "framework": "llamaindex",
            "orchestration": "RouterRetriever",
            "selected_domain_id": preview.route.selected_domain_id,
            "selected_document_count": len(preview.route.selected_document_ids),
            "route_reason": preview.route.route_reason,
            "active_version_by_document_id": preview.active_version_by_document_id,
            "global_ranking": True,
            # 不记原始问题和资料正文；这些可能属于园区/企业内部数据。
        },
    )
    return success(
        MultiDocumentRagContextPreviewResponse(
            selected_domain_id=preview.route.selected_domain_id,
            selected_document_ids=preview.route.selected_document_ids,
            active_version_by_document_id=preview.active_version_by_document_id,
            route_reason=preview.route.route_reason,
            embedding_model=preview.preview.retrieval.embedding_model,
            vector_dimension=preview.preview.retrieval.vector_dimension,
            retrieved_chunk_count=len(preview.preview.retrieval.nodes),
            included_chunk_count=len(preview.preview.references),
            omitted_chunk_count=preview.preview.omitted_node_count,
            top_score=(round(preview.preview.top_score, 6) if preview.preview.top_score is not None else None),
            score_threshold=preview.preview.score_threshold,
            rejected_by_score_threshold=preview.preview.rejected_by_score_threshold,
            context_char_count=len(preview.preview.context),
            references=[
                reference.model_copy(update={"score": round(reference.score, 6)})
                for reference in preview.preview.references
            ],
            context=preview.preview.context,
        ),
        message="LlamaIndex 已完成知识域路由和领域内跨文档全局召回，尚未调用聊天模型",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/multi-document-rag-answer",
    response_model=ApiResponse[MultiDocumentRagAnswerResponse],
    summary="Day31：通过 LlamaIndex 生成带跨文档引用的知识域 RAG 回答",
)
def answer_multi_document_rag(
    request_body: MultiDocumentRagRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[MultiDocumentRagAnswerResponse]:
    """正式多文档问答入口：领域路由后复用项目的 Prompt、引用校验和审计边界。"""

    domains, domain_keywords = _to_multi_document_domains(request_body)
    start_time = time.perf_counter()
    try:
        result = answer_multi_document_with_llamaindex(
            db,
            domains=domains,
            domain_keywords=domain_keywords,
            default_domain_id=request_body.default_domain_id,
            question=request_body.question,
            retrieval_top_k=request_body.retrieval_top_k,
            max_context_characters=request_body.max_context_characters,
            use_reranker=request_body.use_reranker,
            rerank_top_n=request_body.rerank_top_n,
            score_threshold=_resolve_rag_score_threshold(request_body.score_threshold),
            trace_id=request.state.trace_id,
        )
        answer = result.answer_result
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        create_call_log(
            db,
            call_type="multi_document_rag",
            stage="llamaindex_knowledge_domain_route",
            trace_id=request.state.trace_id,
            cost_ms=0,
            detail={
                "framework": "llamaindex",
                "orchestration": "RouterRetriever",
                "selected_domain_id": result.route.selected_domain_id,
                "selected_document_count": len(result.route.selected_document_ids),
                "route_reason": result.route.route_reason,
                "active_version_by_document_id": result.active_version_by_document_id,
                "global_ranking": True,
            },
            commit=False,
        )
        # 无依据兜底不调用模型，不应伪造模型成本或 Prompt 使用记录。
        if answer.model and answer.prompt_identity:
            create_call_log(
                db,
                call_type="multi_document_rag",
                stage="llamaindex_multi_document_query_engine",
                trace_id=request.state.trace_id,
                model=answer.model,
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                total_tokens=answer.total_tokens,
                cost_ms=cost_ms,
                status="success",
                **answer.prompt_identity.as_call_log_fields(),
                detail={
                    "framework": "llamaindex",
                    "orchestration": "RouterRetriever + RetrieverQueryEngine",
                    "node_postprocessor": "GovernedRagNodePostprocessor",
                    "retrieval_backend": "project_milvus",
                    "selected_domain_id": result.route.selected_domain_id,
                    "selected_document_count": len(result.route.selected_document_ids),
                    "active_version_by_document_id": result.active_version_by_document_id,
                    "global_ranking": True,
                    "prompt_source": answer.prompt_identity.prompt_source,
                },
                commit=False,
            )
        db.commit()
        return success(
            MultiDocumentRagAnswerResponse(
                answer=answer.answer,
                references=[
                    reference.model_copy(update={"score": round(reference.score, 6)})
                    for reference in answer.references
                ],
                selected_domain_id=result.route.selected_domain_id,
                selected_document_ids=result.route.selected_document_ids,
                active_version_by_document_id=result.active_version_by_document_id,
                route_reason=result.route.route_reason,
                retrieved_chunk_count=answer.retrieved_node_count,
                included_chunk_count=answer.included_node_count,
                omitted_chunk_count=answer.omitted_node_count,
                top_score=(round(answer.top_score, 6) if answer.top_score is not None else None),
                score_threshold=answer.score_threshold,
                rejected_by_score_threshold=answer.rejected_by_score_threshold,
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                total_tokens=answer.total_tokens,
                cost_ms=cost_ms,
            ),
            message="LlamaIndex 多文档 RAG 回答完成，路由、版本和引用均可追溯",
            trace_id=request.state.trace_id,
        )
    except ModelCallException as exc:
        db.rollback()
        create_call_log(
            db,
            call_type="multi_document_rag",
            stage="llamaindex_multi_document_query_engine",
            trace_id=request.state.trace_id,
            model=settings.dashscope_model,
            cost_ms=round((time.perf_counter() - start_time) * 1000),
            status="error",
            error_type=exc.error_type,
            error_message=exc.message,
            detail={
                "framework": "llamaindex",
                "orchestration": "RouterRetriever + RetrieverQueryEngine",
                "retrieval_backend": "project_milvus",
            },
        )
        raise


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
    """正式单文档 RAG 入口：保留原 URL，内部已升级为 LlamaIndex 编排。

    这避免调用方为框架迁移改接口，也避免形成“/rag-answer 是旧链路、
    /llamaindex-rag-answer 才是新链路”的双生产事实。旧无框架的
    ``search_active_document_chunks → build_rag_context → generate_rag_answer`` 保留在
    ``rag_context_service`` 供对照和历史回归，不再作为此入口的运行路径。
    """

    score_threshold = _resolve_rag_score_threshold(request_body.score_threshold)
    start_time = time.perf_counter()
    try:
        result = answer_active_document_with_llamaindex(
            db,
            document_id=document_id,
            question=request_body.question,
            retrieval_top_k=request_body.retrieval_top_k,
            max_context_characters=request_body.max_context_characters,
            use_reranker=request_body.use_reranker,
            rerank_top_n=request_body.rerank_top_n,
            score_threshold=score_threshold,
            trace_id=request.state.trace_id,
        )
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        # 无资料兜底没有发生模型调用，不应伪造一条成功的模型成本日志。
        if result.model:
            create_call_log(
                db,
                call_type="rag_knowledge_answer",
                stage="llamaindex_query_engine",
                trace_id=request.state.trace_id,
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                cost_ms=cost_ms,
                status="success",
                **result.prompt_identity.as_call_log_fields(),
                detail={
                    "framework": "llamaindex",
                    "orchestration": "RetrieverQueryEngine",
                    "node_postprocessor": "GovernedRagNodePostprocessor",
                    "retrieval_backend": "project_milvus",
                    "version_id": result.retrieval.active_version_id,
                    "prompt_source": result.prompt_identity.prompt_source,
                },
            )
        return success(
            RagAnswerResponse(
                answer=result.answer,
                references=[
                    reference.model_copy(update={"score": round(reference.score, 6)})
                    for reference in result.references
                ],
                document_id=document_id,
                active_version_id=result.retrieval.active_version_id,
                retrieved_chunk_count=result.retrieved_node_count,
                included_chunk_count=result.included_node_count,
                omitted_chunk_count=result.omitted_node_count,
                top_score=(
                    round(result.top_score, 6)
                    if result.top_score is not None
                    else None
                ),
                score_threshold=result.score_threshold,
                rejected_by_score_threshold=result.rejected_by_score_threshold,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                cost_ms=cost_ms,
            ),
            message="LlamaIndex RAG 回答生成完成，引用来源已校验",
            trace_id=request.state.trace_id,
        )
    except ModelCallException as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        create_call_log(
            db,
            call_type="rag_knowledge_answer",
            stage="llamaindex_query_engine",
            trace_id=request.state.trace_id,
            model=settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=exc.error_type,
            error_message=exc.message,
            detail={"framework": "llamaindex", "orchestration": "RetrieverQueryEngine"},
        )
        raise


@router.post(
    "/documents/{document_id}/llamaindex-rag-answer",
    response_model=ApiResponse[LlamaIndexRagAnswerResponse],
    summary="Day31：通过 LlamaIndex RetrieverQueryEngine 生成带引用的 RAG 回答",
)
def answer_with_llamaindex_rag(
    document_id: str,
    request_body: RagAnswerRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[LlamaIndexRagAnswerResponse]:
    """独立候选链路，保留 /rag-answer 基线接口，方便逐次对照而非静默替换。"""

    score_threshold = _resolve_rag_score_threshold(request_body.score_threshold)
    start_time = time.perf_counter()
    try:
        result = answer_active_document_with_llamaindex(
            db,
            document_id=document_id,
            question=request_body.question,
            retrieval_top_k=request_body.retrieval_top_k,
            max_context_characters=request_body.max_context_characters,
            use_reranker=request_body.use_reranker,
            rerank_top_n=request_body.rerank_top_n,
            score_threshold=score_threshold,
            trace_id=request.state.trace_id,
        )
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        if result.model:
            create_call_log(
                db,
                call_type="llamaindex_law_rag",
                stage="llamaindex_query_engine",
                trace_id=request.state.trace_id,
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                cost_ms=cost_ms,
                status="success",
                **result.prompt_identity.as_call_log_fields(),
                detail={
                    "framework": "llamaindex",
                    "orchestration": "RetrieverQueryEngine",
                    "retrieval_backend": "project_milvus",
                    "document_id": document_id,
                    "version_id": result.retrieval.active_version_id,
                    "retrieved_node_count": result.retrieved_node_count,
                    "included_node_count": result.included_node_count,
                    "omitted_node_count": result.omitted_node_count,
                    "prompt_source": result.prompt_identity.prompt_source,
                },
            )
        return success(
            LlamaIndexRagAnswerResponse(
                answer=result.answer,
                references=[
                    reference.model_copy(update={"score": round(reference.score, 6)})
                    for reference in result.references
                ],
                document_id=document_id,
                active_version_id=result.retrieval.active_version_id,
                retrieved_chunk_count=result.retrieved_node_count,
                included_chunk_count=result.included_node_count,
                omitted_chunk_count=result.omitted_node_count,
                top_score=(round(result.top_score, 6) if result.top_score is not None else None),
                score_threshold=result.score_threshold,
                rejected_by_score_threshold=result.rejected_by_score_threshold,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                cost_ms=cost_ms,
            ),
            message="LlamaIndex QueryEngine RAG 回答生成完成，仍复用项目 Milvus、版本与引用治理",
            trace_id=request.state.trace_id,
        )
    except ModelCallException as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        create_call_log(
            db,
            call_type="llamaindex_law_rag",
            stage="llamaindex_query_engine",
            trace_id=request.state.trace_id,
            cost_ms=cost_ms,
            status="error",
            error_type=exc.error_type,
            error_message=exc.message,
            detail={
                "framework": "llamaindex",
                "orchestration": "RetrieverQueryEngine",
                "retrieval_backend": "project_milvus",
                "document_id": document_id,
            },
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
        use_reranker=request_body.use_reranker,
        rerank_top_n=request_body.rerank_top_n,
        score_threshold=_resolve_rag_score_threshold(request_body.score_threshold),
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
            top_score=round(result.top_score, 6) if result.top_score is not None else None,
            score_threshold=result.score_threshold,
            rejected_by_score_threshold=result.rejected_by_score_threshold,
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
        use_reranker=request_body.use_reranker,
        rerank_top_n=request_body.rerank_top_n,
        score_threshold=_resolve_rag_score_threshold(request_body.score_threshold),
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
    dependencies=[Depends(require_permissions(PERMISSION_KNOWLEDGE_WRITE))],
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
    dependencies=[Depends(require_permissions(PERMISSION_KNOWLEDGE_WRITE))],
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
        # 版本切换审计人来自认证上下文，不能接受客户端伪造身份。
        activated_by=request.state.principal.actor_id,
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
