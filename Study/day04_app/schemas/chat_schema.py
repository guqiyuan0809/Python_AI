"""
聊天接口请求体和响应体

类似 Java 项目里的 DTO / VO。
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入的问题")


class SessionChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="会话 ID")
    message: str = Field(..., min_length=1, description="用户输入的问题")
    history_limit: int = Field(6, ge=0, le=20, description="携带最近多少条历史消息")


class SessionStreamChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="会话 ID")
    message: str = Field(..., min_length=1, description="用户输入的问题")
    history_limit: int = Field(6, ge=0, le=20, description="携带最近多少条历史消息")


class ChatResponse(BaseModel):
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class WorkOrderAnalysisRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000, description="需要 AI 分析的工单或业务问题内容")
    session_id: str | None = Field(None, min_length=1, description="可选会话 ID，用于把结构化结果归属到某次会话")
    business_id: str | None = Field(None, min_length=1, description="可选业务 ID，例如工单 ID")


class WorkOrderAnalysisParseTestRequest(BaseModel):
    raw_text: str = Field(..., min_length=1, description="模拟模型返回的原始文本")


class AsyncWorkOrderAnalysisTaskRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="会话 ID")
    content: str = Field(..., min_length=1, max_length=2000, description="需要 AI 分析的工单或业务问题内容")
    business_id: str | None = Field(None, min_length=1, description="可选业务 ID，例如工单 ID")


class WorkOrderAnalysisResult(BaseModel):
    category: Literal["consult", "complaint", "repair", "other"] = Field(..., description="问题分类")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="风险等级")
    summary: str = Field(..., min_length=1, max_length=200, description="问题摘要")
    suggestions: list[str] = Field(..., min_length=1, max_length=5, description="处理建议")
    need_human_review: bool = Field(..., description="是否需要人工复核")
    confidence: float = Field(..., ge=0, le=1, description="模型对分析结果的置信度")


class WorkOrderAnalysisResponse(BaseModel):
    result_id: str | None = None
    analysis: WorkOrderAnalysisResult
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    repair_count: int = Field(0, ge=0, description="结构化输出修复次数")


class CreateSessionResponse(BaseModel):
    session_id: str


class UpdateSessionTitleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=30, description="会话标题")


class SessionTitleResponse(BaseModel):
    session_id: str
    title: str


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str


class ChatSessionItem(BaseModel):
    session_id: str
    user_id: str | None = None
    title: str | None = None
    summary: str | None = None
    status: str
    created_at: str
    updated_at: str


class SessionListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[ChatSessionItem]


class RefreshSessionSummaryResponse(BaseModel):
    session_id: str
    # 本次新生成的摘要版本 ID。
    summary_id: str | None = None
    # 最新摘要内容，会作为后续长对话的长期上下文。
    summary: str
    # 表示这份摘要已经覆盖到哪一条消息。
    summary_until_message_id: str | None = None
    # 当前会话下的摘要版本号。
    version: int | None = None


class ChatMessageItem(BaseModel):
    message_id: str
    session_id: str
    trace_id: str | None = None
    stream_id: str | None = None
    role: str
    content: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    status: str
    error_type: str | None = None
    error_message: str | None = None
    created_at: str


class SessionMessagesResponse(BaseModel):
    session_id: str
    summary: str | None = None
    messages: list[ChatMessageItem]


class SessionMessagesPageResponse(BaseModel):
    session_id: str
    summary: str | None = None
    total: int
    page: int
    page_size: int
    messages: list[ChatMessageItem]


class AiCallLogItem(BaseModel):
    call_id: str
    trace_id: str | None = None
    session_id: str | None = None
    message_id: str | None = None
    call_type: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_ms: int | None = None
    status: str
    error_type: str | None = None
    error_message: str | None = None
    created_at: str


class AiCallLogPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiCallLogItem]


class AiFailureSampleItem(BaseModel):
    sample_id: str
    trace_id: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    message_id: str | None = None
    call_type: str
    model: str | None = None
    schema_type: str
    schema_version: str
    error_type: str
    error_message: str
    raw_text: str | None = None
    validation_error: str | None = None
    created_at: str


class AiFailureSamplePageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiFailureSampleItem]


class AiEvalRunItem(BaseModel):
    run_id: str
    prompt_name: str
    prompt_version: str
    dataset_version: str
    sample_count: int
    schema_valid_rate: float
    category_accuracy: float
    risk_level_accuracy: float
    human_review_accuracy: float
    avg_total_tokens: float | None = None
    avg_cost_ms: float | None = None
    metrics: dict | None = None
    created_at: str


class AiEvalRunPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiEvalRunItem]


class AiEvalCaseResultItem(BaseModel):
    run_id: str
    sample_id: str
    schema_valid: bool
    category_match: bool
    risk_level_match: bool
    human_review_match: bool
    total_tokens: int | None = None
    cost_ms: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    expected: dict | None = None
    actual: dict | None = None
    row: dict | None = None
    created_at: str


class AiEvalCaseResultPageResponse(BaseModel):
    run_id: str
    total: int
    page: int
    page_size: int
    items: list[AiEvalCaseResultItem]


class AiPromptVersionItem(BaseModel):
    prompt_id: str
    prompt_name: str
    prompt_version: str
    description: str | None = None
    system_prompt: str
    user_prompt_template: str
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    status: str
    created_by: str | None = None
    created_at: str
    updated_at: str


class AiPromptVersionPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiPromptVersionItem]


class AiEvalDatasetItem(BaseModel):
    dataset_id: str
    dataset_name: str
    dataset_version: str
    description: str | None = None
    sample_count: int
    status: str
    created_by: str | None = None
    created_at: str
    updated_at: str


class AiEvalDatasetPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiEvalDatasetItem]


class AiEvalSampleItem(BaseModel):
    sample_id: str
    dataset_id: str
    dataset_version: str
    sample_type: str
    input_text: str
    expected: dict
    source_type: str
    source_ref_id: str | None = None
    status: str
    created_by: str | None = None
    created_at: str
    updated_at: str


class AiEvalSamplePageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiEvalSampleItem]


class AsyncSessionChatTaskRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="会话 ID")
    message: str = Field(..., min_length=1, description="用户输入的问题")
    history_limit: int = Field(6, ge=0, le=20, description="携带最近多少条历史消息")


class AsyncTaskSubmitResponse(BaseModel):
    task_id: str
    status: str


class AsyncWorkOrderEvalTaskRequest(BaseModel):
    prompt_name: str = Field("work_order_analysis", min_length=1, description="Prompt 名称")
    prompt_version: str = Field("v2", min_length=1, description="Prompt 版本")
    dataset_version: str = Field(
        "work_order_analysis_v1",
        min_length=1,
        description="评测数据集版本",
    )


class AsyncTaskStatusResponse(BaseModel):
    task_id: str
    broker_task_id: str | None = None
    trace_id: str | None = None
    session_id: str
    message_id: str | None = None
    task_type: str
    status: str
    input_text: str
    result_text: str | None = None
    structured_result: dict | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_ms: int | None = None
    retry_count: int
    max_retries: int
    error_type: str | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str


class AsyncTaskTimeoutScanResponse(BaseModel):
    timeout_count: int
    task_ids: list[str]
