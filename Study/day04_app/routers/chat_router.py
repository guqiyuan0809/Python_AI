"""
聊天接口路由层

类似 Java 里的 ChatController。
"""

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session
import json
import time

from day04_app.common.exceptions import ModelCallException
from day04_app.common.response import ApiResponse, success
from day04_app.database import get_db
from day04_app.schemas.chat_schema import (
    AiCallLogItem,
    AiCallLogPageResponse,
    AiEvalDatasetItem,
    AiEvalDatasetPageResponse,
    AiEvalGateDecisionItem,
    AiEvalGateDecisionPageResponse,
    AiEvalCaseResultItem,
    AiEvalCaseResultPageResponse,
    AiEvalRunItem,
    AiEvalRunPageResponse,
    AiEvalSampleItem,
    AiEvalSamplePageResponse,
    AiFailureSampleItem,
    AiFailureSamplePageResponse,
    AiPromptVersionItem,
    AiPromptVersionPageResponse,
    AiPromptPublishAuditItem,
    AiPromptPublishAuditPageResponse,
    AiPromptRollbackAuditItem,
    AiPromptRollbackAuditPageResponse,
    AsyncWorkOrderAnalysisTaskRequest,
    AsyncWorkOrderEvalTaskRequest,
    AsyncSessionChatTaskRequest,
    AsyncTaskStatusResponse,
    AsyncTaskSubmitResponse,
    AsyncTaskTimeoutScanResponse,
    ChatMessageItem,
    ChatRequest,
    ChatResponse,
    ChatSessionItem,
    ConvertFailureSampleToEvalSampleRequest,
    CreateSessionResponse,
    EvalGateCompareRequest,
    PublishPromptVersionRequest,
    RollbackPromptVersionRequest,
    RefreshSessionSummaryResponse,
    SessionChatRequest,
    SessionListResponse,
    SessionMessagesResponse,
    SessionMessagesPageResponse,
    SessionStatusResponse,
    SessionStreamChatRequest,
    SessionTitleResponse,
    UpdateSessionTitleRequest,
    WorkOrderAnalysisRequest,
    WorkOrderAnalysisParseTestRequest,
    WorkOrderAnalysisResponse,
)
from day04_app.services.async_task_service import (
    create_async_session_chat_task,
    create_async_work_order_eval_task,
    create_async_work_order_analysis_task,
    get_session_rag_retry_parameters,
    get_contextual_index_retry_parameters,
    get_async_task,
    mark_timeout_tasks_error,
    prepare_task_retry,
    prepare_session_rag_task_retry,
    prepare_contextual_index_task_retry,
)
from day04_app.services.call_log_service import create_call_log, list_call_logs
from day04_app.services.failure_sample_service import (
    convert_failure_sample_to_eval_sample,
    create_failure_sample,
    list_failure_samples,
)
from day04_app.services.eval_query_service import list_eval_case_results, list_eval_runs
from day04_app.services.eval_gate_service import create_eval_gate_decision, list_eval_gate_decisions
from day04_app.services.prompt_publish_service import (
    list_prompt_publish_audits,
    list_prompt_rollback_audits,
    publish_prompt_version,
    rollback_prompt_version,
)
from day04_app.services.eval_master_service import (
    list_eval_datasets,
    list_eval_samples_page,
    list_prompt_versions,
)
from day04_app.services.chat_service import (
    analyze_work_order_structured,
    parse_work_order_analysis,
    safe_chat,
    safe_chat_with_messages,
    stream_chat_events,
    stream_session_chat_events,
)
from day04_app.services.session_service import (
    add_message,
    archive_session,
    build_messages,
    create_session,
    generate_session_title,
    get_session,
    get_session_messages,
    get_session_messages_page,
    list_sessions,
    refresh_session_summary,
    restore_session,
    should_refresh_summary_for_session,
    update_message,
    update_session_title,
)
from day04_app.services.outbox_dispatcher import dispatch_outbox_event
from day04_app.services.structured_result_service import create_structured_result
from day04_app.services.structured_result_service import get_structured_result_by_task_id, load_result_json
from day04_app.common.exceptions import (
    ERROR_TYPE_STRUCTURED_FIELD_INVALID,
    ERROR_TYPE_STRUCTURED_JSON_INVALID,
)
from settings import settings


router = APIRouter(prefix="/api/chat", tags=["AI 聊天"])


def to_message_item(message) -> ChatMessageItem:
    return ChatMessageItem(
        message_id=message.message_id,
        session_id=message.session_id,
        trace_id=message.trace_id,
        stream_id=message.stream_id,
        role=message.role,
        content=message.content,
        model=message.model,
        prompt_tokens=message.prompt_tokens,
        completion_tokens=message.completion_tokens,
        total_tokens=message.total_tokens,
        status=message.status,
        error_type=message.error_type,
        error_message=message.error_message,
        created_at=message.created_at.isoformat(timespec="seconds"),
    )


def to_session_item(session) -> ChatSessionItem:
    return ChatSessionItem(
        session_id=session.session_id,
        user_id=session.user_id,
        title=session.title,
        summary=session.summary,
        status=session.status,
        created_at=session.created_at.isoformat(timespec="seconds"),
        updated_at=session.updated_at.isoformat(timespec="seconds"),
    )


def to_call_log_item(call_log) -> AiCallLogItem:
    return AiCallLogItem(
        call_id=call_log.call_id,
        trace_id=call_log.trace_id,
        session_id=call_log.session_id,
        message_id=call_log.message_id,
        call_type=call_log.call_type,
        model=call_log.model,
        prompt_tokens=call_log.prompt_tokens,
        completion_tokens=call_log.completion_tokens,
        total_tokens=call_log.total_tokens,
        cost_ms=call_log.cost_ms,
        status=call_log.status,
        error_type=call_log.error_type,
        error_message=call_log.error_message,
        created_at=call_log.created_at.isoformat(timespec="seconds"),
    )


def to_failure_sample_item(sample) -> AiFailureSampleItem:
    return AiFailureSampleItem(
        sample_id=sample.sample_id,
        trace_id=sample.trace_id,
        task_id=sample.task_id,
        session_id=sample.session_id,
        message_id=sample.message_id,
        call_type=sample.call_type,
        model=sample.model,
        schema_type=sample.schema_type,
        schema_version=sample.schema_version,
        error_type=sample.error_type,
        error_message=sample.error_message,
        raw_text=sample.raw_text,
        validation_error=sample.validation_error,
        created_at=sample.created_at.isoformat(timespec="seconds"),
    )


def parse_json_or_none(json_text: str | None) -> dict | list | None:
    if not json_text:
        return None
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {"raw": json_text}


def to_eval_run_item(eval_run) -> AiEvalRunItem:
    return AiEvalRunItem(
        run_id=eval_run.run_id,
        prompt_name=eval_run.prompt_name,
        prompt_version=eval_run.prompt_version,
        dataset_version=eval_run.dataset_version,
        sample_count=eval_run.sample_count,
        schema_valid_rate=eval_run.schema_valid_rate,
        category_accuracy=eval_run.category_accuracy,
        risk_level_accuracy=eval_run.risk_level_accuracy,
        human_review_accuracy=eval_run.human_review_accuracy,
        avg_total_tokens=eval_run.avg_total_tokens,
        avg_cost_ms=eval_run.avg_cost_ms,
        metrics=parse_json_or_none(eval_run.metrics_json),
        created_at=eval_run.created_at.isoformat(timespec="seconds"),
    )


def to_eval_gate_decision_item(gate_decision) -> AiEvalGateDecisionItem:
    return AiEvalGateDecisionItem(
        gate_id=gate_decision.gate_id,
        baseline_run_id=gate_decision.baseline_run_id,
        candidate_run_id=gate_decision.candidate_run_id,
        prompt_name=gate_decision.prompt_name,
        dataset_version=gate_decision.dataset_version,
        decision=gate_decision.decision,
        comparison=parse_json_or_none(gate_decision.comparison_json) or {},
        reasons=parse_json_or_none(gate_decision.reason_json) or [],
        rule_snapshot=parse_json_or_none(gate_decision.rule_snapshot_json) or {},
        created_at=gate_decision.created_at.isoformat(timespec="seconds"),
    )


def to_eval_case_result_item(case_result) -> AiEvalCaseResultItem:
    return AiEvalCaseResultItem(
        run_id=case_result.run_id,
        sample_id=case_result.sample_id,
        schema_valid=case_result.schema_valid == 1,
        category_match=case_result.category_match == 1,
        risk_level_match=case_result.risk_level_match == 1,
        human_review_match=case_result.human_review_match == 1,
        total_tokens=case_result.total_tokens,
        cost_ms=case_result.cost_ms,
        error_type=case_result.error_type,
        error_message=case_result.error_message,
        expected=parse_json_or_none(case_result.expected_json),
        actual=parse_json_or_none(case_result.actual_json),
        row=parse_json_or_none(case_result.row_json),
        created_at=case_result.created_at.isoformat(timespec="seconds"),
    )


def to_prompt_version_item(prompt) -> AiPromptVersionItem:
    return AiPromptVersionItem(
        prompt_id=prompt.prompt_id,
        prompt_name=prompt.prompt_name,
        prompt_version=prompt.prompt_version,
        description=prompt.description,
        system_prompt=prompt.system_prompt,
        user_prompt_template=prompt.user_prompt_template,
        model=prompt.model,
        temperature=prompt.temperature,
        max_tokens=prompt.max_tokens,
        status=prompt.status,
        created_by=prompt.created_by,
        created_at=prompt.created_at.isoformat(timespec="seconds"),
        updated_at=prompt.updated_at.isoformat(timespec="seconds"),
    )


def to_prompt_publish_audit_item(audit) -> AiPromptPublishAuditItem:
    return AiPromptPublishAuditItem(
        publish_id=audit.publish_id,
        gate_id=audit.gate_id,
        prompt_id=audit.prompt_id,
        prompt_name=audit.prompt_name,
        candidate_prompt_version=audit.candidate_prompt_version,
        previous_prompt_version=audit.previous_prompt_version,
        gate_decision=audit.gate_decision,
        approval_note=audit.approval_note,
        approved_by=audit.approved_by,
        published_at=audit.published_at.isoformat(timespec="seconds"),
    )


def to_prompt_rollback_audit_item(audit) -> AiPromptRollbackAuditItem:
    return AiPromptRollbackAuditItem(
        rollback_id=audit.rollback_id,
        publish_id=audit.publish_id,
        prompt_name=audit.prompt_name,
        rolled_back_prompt_version=audit.rolled_back_prompt_version,
        restored_prompt_version=audit.restored_prompt_version,
        rollback_reason=audit.rollback_reason,
        rolled_back_by=audit.rolled_back_by,
        rolled_back_at=audit.rolled_back_at.isoformat(timespec="seconds"),
    )


def to_eval_dataset_item(dataset) -> AiEvalDatasetItem:
    return AiEvalDatasetItem(
        dataset_id=dataset.dataset_id,
        dataset_name=dataset.dataset_name,
        dataset_version=dataset.dataset_version,
        description=dataset.description,
        sample_count=dataset.sample_count,
        status=dataset.status,
        created_by=dataset.created_by,
        created_at=dataset.created_at.isoformat(timespec="seconds"),
        updated_at=dataset.updated_at.isoformat(timespec="seconds"),
    )


def to_eval_sample_item(sample) -> AiEvalSampleItem:
    return AiEvalSampleItem(
        sample_id=sample.sample_id,
        dataset_id=sample.dataset_id,
        dataset_version=sample.dataset_version,
        sample_type=sample.sample_type,
        input_text=sample.input_text,
        expected=parse_json_or_none(sample.expected_json) or {},
        source_type=sample.source_type,
        source_ref_id=sample.source_ref_id,
        status=sample.status,
        created_by=sample.created_by,
        created_at=sample.created_at.isoformat(timespec="seconds"),
        updated_at=sample.updated_at.isoformat(timespec="seconds"),
    )


def to_async_task_status(task, structured_result: dict | None = None) -> AsyncTaskStatusResponse:
    return AsyncTaskStatusResponse(
        task_id=task.task_id,
        broker_task_id=task.broker_task_id,
        trace_id=task.trace_id,
        session_id=task.session_id,
        message_id=task.message_id,
        task_type=task.task_type,
        status=task.status,
        input_text=task.input_text,
        result_text=task.result_text,
        structured_result=structured_result,
        model=task.model,
        prompt_tokens=task.prompt_tokens,
        completion_tokens=task.completion_tokens,
        total_tokens=task.total_tokens,
        cost_ms=task.cost_ms,
        retry_count=task.retry_count,
        max_retries=task.max_retries,
        error_type=task.error_type,
        error_message=task.error_message,
        created_at=task.created_at.isoformat(timespec="seconds"),
        updated_at=task.updated_at.isoformat(timespec="seconds"),
    )


@router.post("", response_model=ApiResponse[ChatResponse], summary="普通单轮聊天")
def chat(request_body: ChatRequest, request: Request) -> ApiResponse[ChatResponse]:
    result = safe_chat(request_body.message)
    return success(result, trace_id=request.state.trace_id)


@router.post(
    "/structured/work-order/analyze",
    response_model=ApiResponse[WorkOrderAnalysisResponse],
    summary="结构化分析工单内容",
)
def analyze_work_order(
    request_body: WorkOrderAnalysisRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[WorkOrderAnalysisResponse]:
    trace_id = request.state.trace_id
    start_time = time.perf_counter()
    try:
        # Day14 当前只验证“模型输出 JSON + Pydantic 校验”，暂时不走异步任务。
        result = analyze_work_order_structured(request_body.content)
    except ModelCallException as exc:
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        create_call_log(
            db,
            call_type="structured_work_order_analyze",
            trace_id=trace_id,
            model=settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=exc.error_type,
            error_message=exc.message,
        )
        raise

    cost_ms = round((time.perf_counter() - start_time) * 1000)
    structured_result = create_structured_result(
        db,
        business_type="work_order",
        business_id=request_body.business_id,
        schema_type="work_order_analysis",
        schema_version="v1",
        result_data=result.analysis.model_dump(),
        trace_id=trace_id,
        session_id=request_body.session_id,
        status="success",
    )
    result.result_id = structured_result.result_id
    create_call_log(
        db,
        call_type="structured_work_order_analyze",
        trace_id=trace_id,
        model=settings.dashscope_model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        cost_ms=cost_ms,
        status="success",
    )
    return success(result, trace_id=trace_id)


@router.post(
    "/structured/work-order/parse-test",
    response_model=ApiResponse[WorkOrderAnalysisResponse],
    summary="测试结构化输出解析与错误类型",
)
def parse_work_order_analysis_test(
    request_body: WorkOrderAnalysisParseTestRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[WorkOrderAnalysisResponse]:
    trace_id = request.state.trace_id
    try:
        # 这个接口不调模型，只模拟“模型原始输出 -> DTO 校验”，方便稳定验证 error_type。
        analysis = parse_work_order_analysis(request_body.raw_text)
    except ValidationError as exc:
        error_message = f"模型结构化输出字段不合法：{exc.errors()[0]['msg']}"
        validation_error = json.dumps(exc.errors(), ensure_ascii=False)
        create_call_log(
            db,
            call_type="structured_work_order_parse_test",
            trace_id=trace_id,
            model=settings.dashscope_model,
            status="error",
            error_type=ERROR_TYPE_STRUCTURED_FIELD_INVALID,
            error_message=error_message,
        )
        create_failure_sample(
            db,
            call_type="structured_work_order_parse_test",
            schema_type="work_order_analysis",
            schema_version="v1",
            error_type=ERROR_TYPE_STRUCTURED_FIELD_INVALID,
            error_message=error_message,
            raw_text=request_body.raw_text,
            validation_error=validation_error,
            trace_id=trace_id,
            model=settings.dashscope_model,
        )
        raise ModelCallException(
            message=error_message,
            error_type=ERROR_TYPE_STRUCTURED_FIELD_INVALID,
        ) from exc
    except json.JSONDecodeError as exc:
        error_message = "模型结构化输出不是合法 JSON"
        validation_error = str(exc)
        create_call_log(
            db,
            call_type="structured_work_order_parse_test",
            trace_id=trace_id,
            model=settings.dashscope_model,
            status="error",
            error_type=ERROR_TYPE_STRUCTURED_JSON_INVALID,
            error_message=error_message,
        )
        create_failure_sample(
            db,
            call_type="structured_work_order_parse_test",
            schema_type="work_order_analysis",
            schema_version="v1",
            error_type=ERROR_TYPE_STRUCTURED_JSON_INVALID,
            error_message=error_message,
            raw_text=request_body.raw_text,
            validation_error=validation_error,
            trace_id=trace_id,
            model=settings.dashscope_model,
        )
        raise ModelCallException(
            message=error_message,
            error_type=ERROR_TYPE_STRUCTURED_JSON_INVALID,
        ) from exc

    return success(
        WorkOrderAnalysisResponse(
            analysis=analysis,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
        trace_id=trace_id,
    )


@router.get(
    "/failure-samples",
    response_model=ApiResponse[AiFailureSamplePageResponse],
    summary="分页查询 AI 失败样本",
)
def list_ai_failure_samples(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    trace_id: str | None = Query(None, description="请求链路 ID，可选筛选条件"),
    schema_type: str | None = Query(None, description="结构化结果类型，例如 work_order_analysis"),
    error_type: str | None = Query(None, description="错误类型，例如 STRUCTURED_JSON_INVALID"),
    db: Session = Depends(get_db),
) -> ApiResponse[AiFailureSamplePageResponse]:
    samples, total = list_failure_samples(
        db,
        page=page,
        page_size=page_size,
        trace_id=trace_id,
        schema_type=schema_type,
        error_type=error_type,
    )
    return success(
        AiFailureSamplePageResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[to_failure_sample_item(sample) for sample in samples],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/failure-samples/{sample_id}/convert-to-eval-sample",
    response_model=ApiResponse[AiEvalSampleItem],
    summary="将失败样本转入评测样本库",
)
def convert_ai_failure_sample_to_eval_sample(
    sample_id: str,
    request_body: ConvertFailureSampleToEvalSampleRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[AiEvalSampleItem]:
    # expected 必须来自人工标注，不能让模型自动生成，否则评测会失去“标准答案”的意义。
    eval_sample = convert_failure_sample_to_eval_sample(
        db,
        failure_sample_id=sample_id,
        dataset_id=request_body.dataset_id,
        dataset_version=request_body.dataset_version,
        sample_type=request_body.sample_type,
        input_text=request_body.input_text,
        expected=request_body.expected.model_dump(),
    )
    return success(
        to_eval_sample_item(eval_sample),
        message="失败样本已转入评测样本库",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/prompt-versions",
    response_model=ApiResponse[AiPromptVersionPageResponse],
    summary="分页查询 AI Prompt 版本",
)
def list_ai_prompt_versions(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    prompt_name: str | None = Query(None, description="Prompt 名称，例如 work_order_analysis"),
    status: str | None = Query(None, description="Prompt 状态，例如 active/draft"),
    db: Session = Depends(get_db),
) -> ApiResponse[AiPromptVersionPageResponse]:
    prompts, total = list_prompt_versions(
        db,
        page=page,
        page_size=page_size,
        prompt_name=prompt_name,
        status=status,
    )
    return success(
        AiPromptVersionPageResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[to_prompt_version_item(prompt) for prompt in prompts],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/prompt-versions/{prompt_id}/publish",
    response_model=ApiResponse[AiPromptPublishAuditItem],
    summary="人工批准发布候选 Prompt 版本",
)
def publish_ai_prompt_version(
    prompt_id: str,
    request_body: PublishPromptVersionRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[AiPromptPublishAuditItem]:
    audit = publish_prompt_version(
        db,
        prompt_id=prompt_id,
        gate_id=request_body.gate_id,
        approval_note=request_body.approval_note,
        approved_by=request_body.approved_by,
    )
    return success(
        to_prompt_publish_audit_item(audit),
        message="Prompt 已人工批准发布，旧 active 版本已归档",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/prompt-publish-audits",
    response_model=ApiResponse[AiPromptPublishAuditPageResponse],
    summary="分页查询 Prompt 发布审计记录",
)
def list_ai_prompt_publish_audits(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    prompt_name: str | None = Query(None, description="Prompt 名称，例如 work_order_analysis"),
    db: Session = Depends(get_db),
) -> ApiResponse[AiPromptPublishAuditPageResponse]:
    audits, total = list_prompt_publish_audits(
        db,
        page=page,
        page_size=page_size,
        prompt_name=prompt_name,
    )
    return success(
        AiPromptPublishAuditPageResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[to_prompt_publish_audit_item(audit) for audit in audits],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/prompt-publish-audits/{publish_id}/rollback",
    response_model=ApiResponse[AiPromptRollbackAuditItem],
    summary="人工回滚 Prompt 到原线上版本",
)
def rollback_ai_prompt_version(
    publish_id: str,
    request_body: RollbackPromptVersionRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[AiPromptRollbackAuditItem]:
    audit = rollback_prompt_version(
        db,
        publish_id=publish_id,
        rollback_reason=request_body.rollback_reason,
        rolled_back_by=request_body.rolled_back_by,
    )
    return success(
        to_prompt_rollback_audit_item(audit),
        message="Prompt 已回滚到原线上版本，当前版本已归档",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/prompt-rollback-audits",
    response_model=ApiResponse[AiPromptRollbackAuditPageResponse],
    summary="分页查询 Prompt 回滚审计记录",
)
def list_ai_prompt_rollback_audits(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    prompt_name: str | None = Query(None, description="Prompt 名称，例如 work_order_analysis"),
    db: Session = Depends(get_db),
) -> ApiResponse[AiPromptRollbackAuditPageResponse]:
    audits, total = list_prompt_rollback_audits(
        db,
        page=page,
        page_size=page_size,
        prompt_name=prompt_name,
    )
    return success(
        AiPromptRollbackAuditPageResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[to_prompt_rollback_audit_item(audit) for audit in audits],
        ),
        trace_id=request.state.trace_id,
    )


@router.get(
    "/eval-datasets",
    response_model=ApiResponse[AiEvalDatasetPageResponse],
    summary="分页查询 AI 评测数据集",
)
def list_ai_eval_datasets(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    dataset_name: str | None = Query(None, description="数据集名称，例如 work_order_analysis"),
    status: str | None = Query(None, description="数据集状态，例如 active"),
    db: Session = Depends(get_db),
) -> ApiResponse[AiEvalDatasetPageResponse]:
    datasets, total = list_eval_datasets(
        db,
        page=page,
        page_size=page_size,
        dataset_name=dataset_name,
        status=status,
    )
    return success(
        AiEvalDatasetPageResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[to_eval_dataset_item(dataset) for dataset in datasets],
        ),
        trace_id=request.state.trace_id,
    )


@router.get(
    "/eval-samples",
    response_model=ApiResponse[AiEvalSamplePageResponse],
    summary="分页查询 AI 评测样本",
)
def list_ai_eval_samples(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    dataset_version: str | None = Query(None, description="数据集版本，例如 work_order_analysis_v1"),
    sample_type: str | None = Query(None, description="样本类型，例如 normal/boundary/error"),
    status: str | None = Query(None, description="样本状态，例如 active"),
    db: Session = Depends(get_db),
) -> ApiResponse[AiEvalSamplePageResponse]:
    samples, total = list_eval_samples_page(
        db,
        page=page,
        page_size=page_size,
        dataset_version=dataset_version,
        sample_type=sample_type,
        status=status,
    )
    return success(
        AiEvalSamplePageResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[to_eval_sample_item(sample) for sample in samples],
        ),
        trace_id=request.state.trace_id,
    )


@router.get(
    "/eval-runs",
    response_model=ApiResponse[AiEvalRunPageResponse],
    summary="分页查询 AI 评测运行记录",
)
def list_ai_eval_runs(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    prompt_name: str | None = Query(None, description="Prompt 名称，例如 work_order_analysis"),
    prompt_version: str | None = Query(None, description="Prompt 版本，例如 v2"),
    dataset_version: str | None = Query(None, description="数据集版本，例如 work_order_analysis_v1"),
    db: Session = Depends(get_db),
) -> ApiResponse[AiEvalRunPageResponse]:
    eval_runs, total = list_eval_runs(
        db,
        page=page,
        page_size=page_size,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        dataset_version=dataset_version,
    )
    return success(
        AiEvalRunPageResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[to_eval_run_item(eval_run) for eval_run in eval_runs],
        ),
        trace_id=request.state.trace_id,
    )


@router.get(
    "/eval-runs/{run_id}/cases",
    response_model=ApiResponse[AiEvalCaseResultPageResponse],
    summary="分页查询某次 AI 评测的样本明细",
)
def list_ai_eval_case_results(
    run_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(50, ge=1, le=200, description="每页数量"),
    only_failed: bool = Query(False, description="是否只查看未命中的样本"),
    db: Session = Depends(get_db),
) -> ApiResponse[AiEvalCaseResultPageResponse]:
    case_results, total = list_eval_case_results(
        db,
        run_id=run_id,
        page=page,
        page_size=page_size,
        only_failed=only_failed,
    )
    return success(
        AiEvalCaseResultPageResponse(
            run_id=run_id,
            total=total,
            page=page,
            page_size=page_size,
            items=[to_eval_case_result_item(case_result) for case_result in case_results],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/eval-gates/compare",
    response_model=ApiResponse[AiEvalGateDecisionItem],
    summary="比较基线与候选 Prompt 的评测准入结果",
)
def compare_ai_eval_gate(
    request_body: EvalGateCompareRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[AiEvalGateDecisionItem]:
    # Gate 只比较已有 Harness 报告，不重新调用模型，也不会重复消耗 Token。
    gate_decision = create_eval_gate_decision(
        db,
        baseline_run_id=request_body.baseline_run_id,
        candidate_run_id=request_body.candidate_run_id,
    )
    return success(
        to_eval_gate_decision_item(gate_decision),
        message="评测准入门禁已完成",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/eval-gates",
    response_model=ApiResponse[AiEvalGateDecisionPageResponse],
    summary="分页查询 Prompt 评测准入记录",
)
def list_ai_eval_gate_records(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    prompt_name: str | None = Query(None, description="Prompt 名称，例如 work_order_analysis"),
    decision: str | None = Query(None, description="门禁结论，例如 pass/reject/manual_review"),
    db: Session = Depends(get_db),
) -> ApiResponse[AiEvalGateDecisionPageResponse]:
    gate_decisions, total = list_eval_gate_decisions(
        db,
        page=page,
        page_size=page_size,
        prompt_name=prompt_name,
        decision=decision,
    )
    return success(
        AiEvalGateDecisionPageResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[to_eval_gate_decision_item(item) for item in gate_decisions],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/evals/work-order/run/async",
    response_model=ApiResponse[AsyncTaskSubmitResponse],
    summary="异步触发工单结构化分析评测",
)
def submit_async_work_order_eval(
    request_body: AsyncWorkOrderEvalTaskRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[AsyncTaskSubmitResponse]:
    trace_id = request.state.trace_id
    task, outbox_event = create_async_work_order_eval_task(
        db,
        trace_id=trace_id,
        prompt_name=request_body.prompt_name,
        prompt_version=request_body.prompt_version,
        dataset_version=request_body.dataset_version,
        max_retries=settings.async_task_max_retries,
    )
    # 投递失败时 Outbox 保持 pending，由 Celery Beat 后续补发。
    dispatch_outbox_event(db, outbox_event.event_id)
    return success(
        AsyncTaskSubmitResponse(
            task_id=task.task_id,
            status=task.status,
        ),
        message="工单评测任务已提交",
        trace_id=trace_id,
    )


@router.post(
    "/sessions",
    response_model=ApiResponse[CreateSessionResponse],
    summary="创建聊天会话",
)
def create_chat_session(
    request: Request, db: Session = Depends(get_db)
) -> ApiResponse[CreateSessionResponse]:
    session_id = create_session(db)
    return success(
        CreateSessionResponse(session_id=session_id),
        message="会话创建成功",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/sessions",
    response_model=ApiResponse[SessionListResponse],
    summary="分页查询会话列表",
)
def list_chat_sessions(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(10, ge=1, le=50, description="每页数量"),
    user_id: str | None = Query(None, description="用户 ID，可选筛选条件"),
    db: Session = Depends(get_db),
) -> ApiResponse[SessionListResponse]:
    # 查询会话列表时只返回会话级信息，不返回消息明细，避免列表页数据过大。
    sessions, total = list_sessions(
        db,
        page=page,
        page_size=page_size,
        user_id=user_id,
    )
    return success(
        SessionListResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[to_session_item(session) for session in sessions],
        ),
        trace_id=request.state.trace_id,
    )


@router.patch(
    "/sessions/{session_id}/title",
    response_model=ApiResponse[SessionTitleResponse],
    summary="手动修改会话标题",
)
# 用户输入会话标题
def update_chat_session_title(
    session_id: str,
    request_body: UpdateSessionTitleRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[SessionTitleResponse]:
    # 手动标题以用户输入为准，适合前端提供“重命名会话”功能。
    chat_session = update_session_title(db, session_id, request_body.title)
    return success(
        SessionTitleResponse(
            session_id=chat_session.session_id,
            title=chat_session.title or "",
        ),
        message="会话标题更新成功",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/sessions/{session_id}/title/generate",
    response_model=ApiResponse[SessionTitleResponse],
    summary="自动生成会话标题",
)
# 根据会话历史自动生成标题
def generate_chat_session_title(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[SessionTitleResponse]:
    # 自动生成标题会读取会话历史，优先调用模型生成，失败时使用规则标题兜底。
    chat_session = generate_session_title(db, session_id)
    return success(
        SessionTitleResponse(
            session_id=chat_session.session_id,
            title=chat_session.title or "",
        ),
        message="会话标题生成成功",
        trace_id=request.state.trace_id,
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=ApiResponse[SessionStatusResponse],
    summary="归档会话",
)
def archive_chat_session(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[SessionStatusResponse]:
    # 对外表现为删除会话，底层只做逻辑归档，不物理删除聊天记录。
    chat_session = archive_session(db, session_id)
    return success(
        SessionStatusResponse(
            session_id=chat_session.session_id,
            status=chat_session.status,
        ),
        message="会话已归档",
        trace_id=request.state.trace_id,
    )


@router.patch(
    "/sessions/{session_id}/restore",
    response_model=ApiResponse[SessionStatusResponse],
    summary="恢复归档会话",
)
def restore_chat_session(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[SessionStatusResponse]:
    # 恢复归档会话后，它会重新出现在默认会话列表中。
    chat_session = restore_session(db, session_id)
    return success(
        SessionStatusResponse(
            session_id=chat_session.session_id,
            status=chat_session.status,
        ),
        message="会话已恢复",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=ApiResponse[SessionMessagesResponse],
    summary="查询会话全部消息",
)
def list_session_messages(
    session_id: str, request: Request, db: Session = Depends(get_db)
) -> ApiResponse[SessionMessagesResponse]:
    chat_session = get_session(db, session_id)
    messages = get_session_messages(db, session_id)
    return success(
        SessionMessagesResponse(
            session_id=session_id,
            summary=chat_session.summary,
            messages=[to_message_item(message) for message in messages],
        ),
        trace_id=request.state.trace_id,
    )


@router.get(
    "/sessions/{session_id}/messages/page",
    response_model=ApiResponse[SessionMessagesPageResponse],
    summary="分页查询会话消息",
)
def list_session_messages_page(
    session_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页消息数量"),
    db: Session = Depends(get_db),
) -> ApiResponse[SessionMessagesPageResponse]:
    chat_session = get_session(db, session_id)
    messages, total = get_session_messages_page(
        db,
        session_id=session_id,
        page=page,
        page_size=page_size,
    )
    return success(
        SessionMessagesPageResponse(
            session_id=session_id,
            summary=chat_session.summary,
            total=total,
            page=page,
            page_size=page_size,
            messages=[to_message_item(message) for message in messages],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/sessions/{session_id}/summary/refresh",
    response_model=ApiResponse[RefreshSessionSummaryResponse],
    summary="手动刷新会话摘要",
)
def refresh_summary(
    session_id: str, request: Request, db: Session = Depends(get_db)
) -> ApiResponse[RefreshSessionSummaryResponse]:
    summary_record = refresh_session_summary(db, session_id)
    return success(
        RefreshSessionSummaryResponse(
            session_id=session_id,
            summary_id=summary_record.summary_id,
            summary=summary_record.summary,
            summary_until_message_id=summary_record.summary_until_message_id,
            version=summary_record.version,
        ),
        message="会话摘要刷新成功",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/sessions/chat",
    response_model=ApiResponse[ChatResponse],
    summary="会话多轮聊天",
)
def session_chat(
    request_body: SessionChatRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[ChatResponse]:
    trace_id = request.state.trace_id
    messages = build_messages(
        db=db,
        session_id=request_body.session_id,
        current_question=request_body.message,
        history_limit=request_body.history_limit,
    )
    add_message(db, request_body.session_id, "user", request_body.message, trace_id=trace_id)
    assistant_message = add_message(
        db,
        request_body.session_id,
        "assistant",
        "AI 回答生成中",
        trace_id=trace_id,
        model=settings.dashscope_model,
        status="pending",
    )

    # 记录模型调用开始时间，后面用于统计本次 AI 调用耗时。
    start_time = time.perf_counter()
    try:
        result = safe_chat_with_messages(messages)
    except ModelCallException as exc:
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
            call_type="session_chat",
            trace_id=trace_id,
            session_id=request_body.session_id,
            message_id=assistant_message.message_id,
            model=settings.dashscope_model,
            cost_ms=cost_ms,
            status="error",
            error_type=exc.error_type,
            error_message=exc.message,
        )
        raise

    cost_ms = round((time.perf_counter() - start_time) * 1000)
    update_message(
        db,
        assistant_message.message_id,
        content=result.answer,
        status="success",
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
    )
    create_call_log(
        db,
        call_type="session_chat",
        trace_id=trace_id,
        session_id=request_body.session_id,
        message_id=assistant_message.message_id,
        model=settings.dashscope_model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        cost_ms=cost_ms,
        status="success",
    )
    if should_refresh_summary_for_session(db, request_body.session_id):
        refresh_session_summary(db, request_body.session_id)
    return success(result, trace_id=trace_id)


@router.get(
    "/call-logs",
    response_model=ApiResponse[AiCallLogPageResponse],
    summary="分页查询 AI 调用日志",
)
def list_ai_call_logs(
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    trace_id: str | None = Query(None, description="请求链路 ID，可选筛选条件"),
    session_id: str | None = Query(None, description="会话 ID，可选筛选条件"),
    status: str | None = Query(None, description="调用状态，例如 success/error"),
    error_type: str | None = Query(None, description="错误类型，例如 MODEL_CALL_FAILED"),
    db: Session = Depends(get_db),
) -> ApiResponse[AiCallLogPageResponse]:
    # 调用日志属于运维和排查视角，支持按 trace_id、session_id、status 过滤。
    call_logs, total = list_call_logs(
        db,
        page=page,
        page_size=page_size,
        trace_id=trace_id,
        session_id=session_id,
        status=status,
        error_type=error_type,
    )
    return success(
        AiCallLogPageResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=[to_call_log_item(call_log) for call_log in call_logs],
        ),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/sessions/chat/async",
    response_model=ApiResponse[AsyncTaskSubmitResponse],
    summary="提交异步会话聊天任务",
)
def submit_async_session_chat(
    request_body: AsyncSessionChatTaskRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[AsyncTaskSubmitResponse]:
    trace_id = request.state.trace_id

    # 先校验会话存在，再创建任务记录，避免后台任务拿到无效 session_id。
    get_session(db, request_body.session_id)
    task, outbox_event = create_async_session_chat_task(
        db,
        session_id=request_body.session_id,
        input_text=request_body.message,
        trace_id=trace_id,
        model=settings.dashscope_model,
        history_limit=request_body.history_limit,
        max_retries=settings.async_task_max_retries,
    )
    # Broker 暂时不可用时事件会留在 Outbox，Celery Beat 后续会自动补发。
    dispatch_outbox_event(db, outbox_event.event_id)
    return success(
        AsyncTaskSubmitResponse(
            task_id=task.task_id,
            status=task.status,
        ),
        message="异步任务已提交",
        trace_id=trace_id,
    )


@router.post(
    "/structured/work-order/analyze/async",
    response_model=ApiResponse[AsyncTaskSubmitResponse],
    summary="提交异步工单结构化分析任务",
)
def submit_async_work_order_analysis(
    request_body: AsyncWorkOrderAnalysisTaskRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[AsyncTaskSubmitResponse]:
    trace_id = request.state.trace_id

    # 结构化分析异步任务也必须归属会话，方便后续查询聊天历史和审计。
    get_session(db, request_body.session_id)
    task, outbox_event = create_async_work_order_analysis_task(
        db,
        session_id=request_body.session_id,
        content=request_body.content,
        trace_id=trace_id,
        model=settings.dashscope_model,
        business_id=request_body.business_id,
        max_retries=settings.async_task_max_retries,
    )
    dispatch_outbox_event(db, outbox_event.event_id)
    return success(
        AsyncTaskSubmitResponse(
            task_id=task.task_id,
            status=task.status,
        ),
        message="异步结构化分析任务已提交",
        trace_id=trace_id,
    )


@router.post(
    "/tasks/actions/timeout-scan",
    response_model=ApiResponse[AsyncTaskTimeoutScanResponse],
    summary="扫描并标记超时异步任务",
)
def scan_timeout_async_tasks(
    request: Request,
    timeout_minutes: int = Query(10, ge=1, le=1440, description="任务超时分钟数"),
    db: Session = Depends(get_db),
) -> ApiResponse[AsyncTaskTimeoutScanResponse]:
    # 学习阶段通过接口手动触发；企业项目通常由定时任务或独立 Worker 定期执行。
    timeout_tasks = mark_timeout_tasks_error(db, timeout_minutes=timeout_minutes)
    return success(
        AsyncTaskTimeoutScanResponse(
            timeout_count=len(timeout_tasks),
            task_ids=[task.task_id for task in timeout_tasks],
        ),
        message="超时任务扫描完成",
        trace_id=request.state.trace_id,
    )


@router.get(
    "/tasks/{task_id}",
    response_model=ApiResponse[AsyncTaskStatusResponse],
    summary="查询异步任务状态",
)
def get_async_task_status(
    task_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[AsyncTaskStatusResponse]:
    task = get_async_task(db, task_id)
    structured_result_json = None
    if task.status == "success" and task.task_type == "work_order_analysis":
        structured_result = get_structured_result_by_task_id(db, task_id)
        structured_result_json = load_result_json(structured_result)
    return success(
        to_async_task_status(task, structured_result=structured_result_json),
        trace_id=request.state.trace_id,
    )


@router.post(
    "/tasks/{task_id}/retry",
    response_model=ApiResponse[AsyncTaskSubmitResponse],
    summary="重试失败的异步任务",
)
def retry_async_task(
    task_id: str,
    request: Request,
    history_limit: int = Query(6, ge=0, le=20, description="重试时携带的历史消息数量"),
    db: Session = Depends(get_db),
) -> ApiResponse[AsyncTaskSubmitResponse]:
    existing_task = get_async_task(db, task_id)
    if existing_task.task_type == "session_rag":
        (
            document_id,
            retrieval_top_k,
            max_context_characters,
            use_reranker,
            rerank_top_n,
            score_threshold,
        ) = get_session_rag_retry_parameters(
            db,
            task_id,
        )
        task, outbox_event = prepare_session_rag_task_retry(
            db,
            task_id=task_id,
            document_id=document_id,
            retrieval_top_k=retrieval_top_k,
            max_context_characters=max_context_characters,
            use_reranker=use_reranker,
            rerank_top_n=rerank_top_n,
            score_threshold=score_threshold,
        )
    elif existing_task.task_type == "knowledge_contextual_index":
        version_id, context_model, context_max_tokens = get_contextual_index_retry_parameters(
            db,
            task_id,
        )
        task, outbox_event = prepare_contextual_index_task_retry(
            db,
            task_id=task_id,
            version_id=version_id,
            context_model=context_model,
            context_max_tokens=context_max_tokens,
        )
    else:
        task, outbox_event = prepare_task_retry(db, task_id, history_limit=history_limit)
    dispatch_outbox_event(db, outbox_event.event_id)
    return success(
        AsyncTaskSubmitResponse(
            task_id=task.task_id,
            status=task.status,
        ),
        message="异步任务已重新提交",
        trace_id=request.state.trace_id,
    )


@router.post("/stream", summary="普通流式聊天")
def stream_chat(request_body: ChatRequest, request: Request) -> StreamingResponse:
    trace_id = request.state.trace_id
    return StreamingResponse(
        stream_chat_events(request_body.message, trace_id),
        media_type="text/event-stream",
        headers={"X-Trace-Id": trace_id},
    )


@router.post("/sessions/stream", summary="会话流式聊天")
def stream_session_chat(
    request_body: SessionStreamChatRequest,
    request: Request,
) -> StreamingResponse:
    trace_id = request.state.trace_id

    # SSE 接口返回的是事件流，不适合包一层统一 ApiResponse。
    return StreamingResponse(
        stream_session_chat_events(
            session_id=request_body.session_id,
            message=request_body.message,
            trace_id=trace_id,
            history_limit=request_body.history_limit,
        ),
        media_type="text/event-stream",
        headers={"X-Trace-Id": trace_id},
    )


@router.get("/path-variable/{user_id}", summary="路径参数测试")
def path_variable(user_id: str, request: Request) -> ApiResponse[dict]:
    return success(
        {"user_id": user_id},
        message="路由参数",
        trace_id=request.state.trace_id,
    )
