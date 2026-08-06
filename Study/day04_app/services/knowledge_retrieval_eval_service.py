"""RAG 检索评测数据服务：管理人工标注、运行记录与指标计算。"""

from __future__ import annotations

import json
from datetime import datetime
from time import perf_counter
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.models import (
    KnowledgeDocument,
    KnowledgeDocumentSegment,
    KnowledgeRetrievalEvalCaseResult,
    KnowledgeRetrievalEvalDataset,
    KnowledgeRetrievalEvalRun,
    KnowledgeRetrievalEvalSample,
)
from day04_app.services.knowledge_milvus_search_service import (
    search_document_version_chunks_for_validation,
)
from settings import settings


def create_retrieval_eval_dataset(
    db: Session,
    *,
    dataset_name: str,
    dataset_version: str,
    document_id: str,
    description: str | None,
    created_by: str | None,
) -> KnowledgeRetrievalEvalDataset:
    """创建草稿数据集；样本标注完成后再由后续接口发布为 active。"""
    document = db.scalar(
        select(KnowledgeDocument).where(KnowledgeDocument.document_id == document_id)
    )
    if document is None:
        raise BusinessException(code=40451, message="知识库文档不存在")
    existing = db.scalar(
        select(KnowledgeRetrievalEvalDataset).where(
            KnowledgeRetrievalEvalDataset.dataset_name == dataset_name,
            KnowledgeRetrievalEvalDataset.dataset_version == dataset_version,
        )
    )
    if existing is not None:
        raise BusinessException(code=40969, message="检索评测数据集名称和版本已存在")

    dataset = KnowledgeRetrievalEvalDataset(
        dataset_id=uuid4().hex,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        document_id=document_id,
        description=description,
        created_by=created_by,
        status="draft",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def _get_dataset(db: Session, dataset_id: str) -> KnowledgeRetrievalEvalDataset:
    dataset = db.scalar(
        select(KnowledgeRetrievalEvalDataset).where(
            KnowledgeRetrievalEvalDataset.dataset_id == dataset_id
        )
    )
    if dataset is None:
        raise BusinessException(code=40453, message="检索评测数据集不存在")
    if dataset.status != "draft":
        raise BusinessException(code=40970, message="只有 draft 数据集允许新增或修改样本")
    return dataset


def _validate_expected_segments(
    db: Session,
    *,
    document_id: str,
    expected_segment_indexes: list[int],
) -> list[int]:
    """标注必须指向该文档真实的原始段，防止评测规则本身写错。"""
    normalized_indexes = sorted(set(expected_segment_indexes))
    existing_indexes = set(
        db.scalars(
            select(KnowledgeDocumentSegment.segment_index).where(
                KnowledgeDocumentSegment.document_id == document_id,
                KnowledgeDocumentSegment.segment_index.in_(normalized_indexes),
            )
        )
    )
    missing_indexes = [index for index in normalized_indexes if index not in existing_indexes]
    if missing_indexes:
        raise BusinessException(
            code=40072,
            message=f"期望原文段不存在：{missing_indexes}",
        )
    return normalized_indexes


def create_retrieval_eval_sample(
    db: Session,
    *,
    dataset_id: str,
    question: str,
    sample_type: str,
    expected_answerable: bool,
    expected_segment_indexes: list[int],
    expected_note: str | None,
    created_by: str | None,
) -> KnowledgeRetrievalEvalSample:
    """以 segment_index 固化期望依据，确保重切 chunk 后历史评测仍可复现。"""
    dataset = _get_dataset(db, dataset_id)
    if expected_answerable:
        normalized_indexes = _validate_expected_segments(
            db,
            document_id=dataset.document_id,
            expected_segment_indexes=expected_segment_indexes,
        )
    else:
        normalized_indexes = []

    sample = KnowledgeRetrievalEvalSample(
        sample_id=uuid4().hex,
        dataset_id=dataset.dataset_id,
        question=question,
        sample_type=sample_type,
        expected_answerable=int(expected_answerable),
        expected_segment_indexes_json=json.dumps(normalized_indexes),
        expected_note=expected_note,
        created_by=created_by,
        status="active",
    )
    db.add(sample)
    # 数据集样本数是查询优化快照，每次样本变动时同事务维护。
    dataset.sample_count += 1
    db.commit()
    db.refresh(sample)
    return sample


def list_retrieval_eval_datasets(
    db: Session,
    *,
    page: int,
    page_size: int,
    document_id: str | None,
    status: str | None,
) -> tuple[list[KnowledgeRetrievalEvalDataset], int]:
    filters = []
    if document_id:
        filters.append(KnowledgeRetrievalEvalDataset.document_id == document_id)
    if status:
        filters.append(KnowledgeRetrievalEvalDataset.status == status)
    total = db.scalar(
        select(func.count()).select_from(KnowledgeRetrievalEvalDataset).where(*filters)
    ) or 0
    datasets = list(
        db.scalars(
            select(KnowledgeRetrievalEvalDataset)
            .where(*filters)
            .order_by(KnowledgeRetrievalEvalDataset.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    )
    return datasets, total


def list_retrieval_eval_samples(
    db: Session,
    *,
    dataset_id: str,
    page: int,
    page_size: int,
) -> tuple[list[KnowledgeRetrievalEvalSample], int]:
    dataset_exists = db.scalar(
        select(KnowledgeRetrievalEvalDataset.id).where(
            KnowledgeRetrievalEvalDataset.dataset_id == dataset_id
        )
    )
    if dataset_exists is None:
        raise BusinessException(code=40453, message="检索评测数据集不存在")
    filters = [KnowledgeRetrievalEvalSample.dataset_id == dataset_id]
    total = db.scalar(
        select(func.count()).select_from(KnowledgeRetrievalEvalSample).where(*filters)
    ) or 0
    samples = list(
        db.scalars(
            select(KnowledgeRetrievalEvalSample)
            .where(*filters)
            .order_by(KnowledgeRetrievalEvalSample.id.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    )
    return samples, total


def _get_retrieval_eval_dataset_for_run(
    db: Session,
    dataset_id: str,
) -> KnowledgeRetrievalEvalDataset:
    """运行时允许 draft/active 数据集，archived 数据集只保留历史、不可再发起评测。"""
    dataset = db.scalar(
        select(KnowledgeRetrievalEvalDataset).where(
            KnowledgeRetrievalEvalDataset.dataset_id == dataset_id
        )
    )
    if dataset is None:
        raise BusinessException(code=40453, message="检索评测数据集不存在")
    if dataset.status == "archived":
        raise BusinessException(code=40972, message="已归档的检索评测数据集不能发起新评测")
    return dataset


def _serialize_retrieved_chunks(items) -> tuple[list[int], list[dict[str, object]]]:
    """将召回块的来源段按排名展开，既用于指标也作为不可变的运行结果快照。"""
    retrieved_segment_indexes: list[int] = []
    seen_segment_indexes: set[int] = set()
    retrieved_chunks: list[dict[str, object]] = []
    for rank, item in enumerate(items, start=1):
        segment_indexes = [reference.segment_index for reference in item.source_references]
        for segment_index in segment_indexes:
            if segment_index not in seen_segment_indexes:
                seen_segment_indexes.add(segment_index)
                retrieved_segment_indexes.append(segment_index)
        retrieved_chunks.append(
            {
                "rank": rank,
                "chunk_id": item.chunk_id,
                "chunk_index": item.chunk_index,
                "score": item.score,
                "segment_indexes": segment_indexes,
            }
        )
    return retrieved_segment_indexes, retrieved_chunks


def run_retrieval_evaluation(
    db: Session,
    *,
    dataset_id: str,
    document_version_id: str,
    retrieval_top_k: int,
    score_threshold: float | None,
    created_by: str | None,
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    sample_limit: int | None = None,
) -> KnowledgeRetrievalEvalRun:
    """运行一次版本检索评测，单条失败会留下明细而不会阻断其他样本。"""
    dataset = _get_retrieval_eval_dataset_for_run(db, dataset_id)
    sample_query = (
        select(KnowledgeRetrievalEvalSample)
        .where(
            KnowledgeRetrievalEvalSample.dataset_id == dataset.dataset_id,
            KnowledgeRetrievalEvalSample.status == "active",
        )
        .order_by(KnowledgeRetrievalEvalSample.id.asc())
    )
    if sample_limit is not None:
        sample_query = sample_query.limit(sample_limit)
    samples = list(db.scalars(sample_query))
    if not samples:
        raise BusinessException(code=40971, message="数据集没有 active 样本，不能运行检索评测")

    run = KnowledgeRetrievalEvalRun(
        run_id=uuid4().hex,
        dataset_id=dataset.dataset_id,
        document_id=dataset.document_id,
        document_version_id=document_version_id,
        retrieval_top_k=retrieval_top_k,
        score_threshold=score_threshold,
        use_reranker=int(use_reranker),
        rerank_top_n=rerank_top_n if use_reranker else None,
        reranker_model=settings.dashscope_rerank_model if use_reranker else None,
        sample_count=len(samples),
        created_by=created_by,
        status="running",
    )
    db.add(run)
    # 先固化 running 记录；即使随后进程意外中断，也能在管理端发现未完成运行。
    db.commit()

    run_started = perf_counter()
    answerable_sample_count = 0
    answerable_hit_count = 0
    total_expected_segment_count = 0
    total_hit_segment_count = 0
    total_retrieved_chunk_count = 0
    total_relevant_retrieved_chunk_count = 0
    reciprocal_rank_sum = 0.0
    no_answer_sample_count = 0
    no_answer_top_scores: list[float] = []
    no_answer_false_positive_count = 0

    for sample in samples:
        case_started = perf_counter()
        expected_segment_indexes = json.loads(sample.expected_segment_indexes_json)
        try:
            model, dimension, _, items = search_document_version_chunks_for_validation(
                db,
                document_id=dataset.document_id,
                version_id=document_version_id,
                question=sample.question,
                top_k=retrieval_top_k,
                use_reranker=use_reranker,
                rerank_top_n=rerank_top_n,
            )
            if run.embedding_model is None:
                run.embedding_model = model
                run.vector_dimension = dimension

            retrieved_segment_indexes, retrieved_chunks = _serialize_retrieved_chunks(items)
            top_score = items[0].score if items else None
            expected_segment_index_set = set(expected_segment_indexes)
            first_hit_rank: int | None = None
            hit_segment_indexes: set[int] = set()
            relevant_retrieved_chunk_count: int | None = None
            precision_at_k: float | None = None
            if sample.expected_answerable:
                relevant_retrieved_chunk_count = 0
                for chunk in retrieved_chunks:
                    matched_segment_indexes = expected_segment_index_set.intersection(
                        chunk["segment_indexes"]
                    )
                    if matched_segment_indexes:
                        relevant_retrieved_chunk_count += 1
                        hit_segment_indexes.update(matched_segment_indexes)
                    if first_hit_rank is None and matched_segment_indexes:
                        first_hit_rank = int(chunk["rank"])
                is_hit = first_hit_rank is not None
                expected_segment_count = len(expected_segment_index_set)
                hit_segment_count = len(hit_segment_indexes)
                retrieved_chunk_count = len(retrieved_chunks)
                precision_at_k = (
                    relevant_retrieved_chunk_count / retrieved_chunk_count
                    if retrieved_chunk_count
                    else None
                )
                answerable_sample_count += 1
                total_expected_segment_count += expected_segment_count
                total_hit_segment_count += hit_segment_count
                total_retrieved_chunk_count += retrieved_chunk_count
                total_relevant_retrieved_chunk_count += relevant_retrieved_chunk_count
                if is_hit:
                    answerable_hit_count += 1
                    reciprocal_rank_sum += 1 / first_hit_rank
                is_false_positive = None
            else:
                is_hit = None
                expected_segment_count = None
                hit_segment_count = None
                no_answer_sample_count += 1
                if top_score is not None:
                    no_answer_top_scores.append(top_score)
                # 无答案样本也会召回“最相近”内容；只有配置阈值后才能判断是否误放行。
                is_false_positive = (
                    top_score is not None and top_score >= score_threshold
                    if score_threshold is not None
                    else None
                )
                if is_false_positive:
                    no_answer_false_positive_count += 1

            db.add(
                KnowledgeRetrievalEvalCaseResult(
                    case_result_id=uuid4().hex,
                    run_id=run.run_id,
                    sample_id=sample.sample_id,
                    question_snapshot=sample.question,
                    sample_type_snapshot=sample.sample_type,
                    expected_answerable_snapshot=sample.expected_answerable,
                    expected_segment_indexes_json=json.dumps(expected_segment_indexes),
                    retrieved_segment_indexes_json=json.dumps(retrieved_segment_indexes),
                    retrieved_chunks_json=json.dumps(retrieved_chunks),
                    first_hit_rank=first_hit_rank,
                    is_hit=int(is_hit) if is_hit is not None else None,
                    hit_segment_count=hit_segment_count,
                    expected_segment_count=expected_segment_count,
                    relevant_retrieved_chunk_count=relevant_retrieved_chunk_count,
                    precision_at_k=precision_at_k,
                    top_score=top_score,
                    is_false_positive=(
                        int(is_false_positive) if is_false_positive is not None else None
                    ),
                    elapsed_ms=round((perf_counter() - case_started) * 1000),
                    status="success",
                )
            )
            run.success_count += 1
        except Exception as exc:
            # 评测框架应记录问题而非吞掉问题，且继续执行剩余样本以保留可用数据。
            db.add(
                KnowledgeRetrievalEvalCaseResult(
                    case_result_id=uuid4().hex,
                    run_id=run.run_id,
                    sample_id=sample.sample_id,
                    question_snapshot=sample.question,
                    sample_type_snapshot=sample.sample_type,
                    expected_answerable_snapshot=sample.expected_answerable,
                    expected_segment_indexes_json=json.dumps(expected_segment_indexes),
                    retrieved_segment_indexes_json="[]",
                    retrieved_chunks_json="[]",
                    elapsed_ms=round((perf_counter() - case_started) * 1000),
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            run.error_count += 1

    run.answerable_sample_count = answerable_sample_count
    run.answerable_hit_count = answerable_hit_count
    run.total_expected_segment_count = total_expected_segment_count
    run.total_hit_segment_count = total_hit_segment_count
    run.total_retrieved_chunk_count = total_retrieved_chunk_count
    run.total_relevant_retrieved_chunk_count = total_relevant_retrieved_chunk_count
    run.hit_at_k = answerable_hit_count / answerable_sample_count if answerable_sample_count else None
    run.recall_at_k = (
        total_hit_segment_count / total_expected_segment_count
        if total_expected_segment_count
        else None
    )
    run.precision_at_k = (
        total_relevant_retrieved_chunk_count / total_retrieved_chunk_count
        if total_retrieved_chunk_count
        else None
    )
    run.mrr_at_k = reciprocal_rank_sum / answerable_sample_count if answerable_sample_count else None
    run.no_answer_sample_count = no_answer_sample_count
    run.no_answer_avg_top_score = (
        sum(no_answer_top_scores) / len(no_answer_top_scores) if no_answer_top_scores else None
    )
    if score_threshold is not None:
        run.no_answer_false_positive_count = no_answer_false_positive_count
        run.no_answer_false_positive_rate = (
            no_answer_false_positive_count / no_answer_sample_count if no_answer_sample_count else None
        )
    run.elapsed_ms = round((perf_counter() - run_started) * 1000)
    run.finished_at = datetime.now()
    if run.error_count == 0:
        run.status = "success"
    elif run.success_count > 0:
        run.status = "partial_success"
        run.error_message = f"{run.error_count} 条样本检索失败，详见评测明细"
    else:
        run.status = "error"
        run.error_message = "所有评测样本检索失败，详见评测明细"
    db.commit()
    db.refresh(run)
    return run


def list_retrieval_eval_runs(
    db: Session,
    *,
    page: int,
    page_size: int,
    dataset_id: str | None,
    document_version_id: str | None,
) -> tuple[list[KnowledgeRetrievalEvalRun], int]:
    """按数据集或文档版本查询历史运行，支持管理端版本横向对比。"""
    filters = []
    if dataset_id:
        filters.append(KnowledgeRetrievalEvalRun.dataset_id == dataset_id)
    if document_version_id:
        filters.append(KnowledgeRetrievalEvalRun.document_version_id == document_version_id)
    total = db.scalar(
        select(func.count()).select_from(KnowledgeRetrievalEvalRun).where(*filters)
    ) or 0
    runs = list(
        db.scalars(
            select(KnowledgeRetrievalEvalRun)
            .where(*filters)
            .order_by(KnowledgeRetrievalEvalRun.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    )
    return runs, total


def list_retrieval_eval_case_results(
    db: Session,
    *,
    run_id: str,
    page: int,
    page_size: int,
) -> tuple[list[KnowledgeRetrievalEvalCaseResult], int]:
    """查询一次运行的样本明细，用于定位漏召回和无答案误放行。"""
    run_exists = db.scalar(
        select(KnowledgeRetrievalEvalRun.id).where(
            KnowledgeRetrievalEvalRun.run_id == run_id
        )
    )
    if run_exists is None:
        raise BusinessException(code=40454, message="检索评测运行不存在")
    filters = [KnowledgeRetrievalEvalCaseResult.run_id == run_id]
    total = db.scalar(
        select(func.count()).select_from(KnowledgeRetrievalEvalCaseResult).where(*filters)
    ) or 0
    case_results = list(
        db.scalars(
            select(KnowledgeRetrievalEvalCaseResult)
            .where(*filters)
            .order_by(KnowledgeRetrievalEvalCaseResult.id.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    )
    return case_results, total


def list_document_segments(
    db: Session,
    *,
    document_id: str,
    page: int,
    page_size: int,
) -> tuple[list[KnowledgeDocumentSegment], int]:
    """分页查看稳定原始段，人工标注时不需要直接登录 MySQL 找 segment_index。"""
    document_exists = db.scalar(
        select(KnowledgeDocument.id).where(KnowledgeDocument.document_id == document_id)
    )
    if document_exists is None:
        raise BusinessException(code=40451, message="知识库文档不存在")
    filters = [KnowledgeDocumentSegment.document_id == document_id]
    total = db.scalar(
        select(func.count()).select_from(KnowledgeDocumentSegment).where(*filters)
    ) or 0
    segments = list(
        db.scalars(
            select(KnowledgeDocumentSegment)
            .where(*filters)
            .order_by(KnowledgeDocumentSegment.segment_index.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    )
    return segments, total
