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
    task_id: str | None = None
    run_id: str | None = None
    call_type: str
    stage: str | None = None
    prompt_id: str | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    prompt_template_hash: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_ms: int | None = None
    status: str
    error_type: str | None = None
    error_message: str | None = None
    detail: dict | None = None
    created_at: str


class AiCallLogPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiCallLogItem]


class AiTraceTaskItem(BaseModel):
    task_id: str
    trace_id: str | None = None
    session_id: str
    message_id: str | None = None
    task_type: str
    status: str
    total_tokens: int | None = None
    cost_ms: int | None = None
    retry_count: int
    error_type: str | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str


class AiTraceObservabilityResponse(BaseModel):
    trace_id: str
    call_logs: list[AiCallLogItem]
    tasks: list[AiTraceTaskItem]
    # 保留旧字段，值等于 event_total_*；旧调用方不会因响应字段变更而中断。
    total_tokens: int
    total_cost_ms: int
    error_count: int
    blocked_count: int
    event_total_tokens: int = Field(..., description="所有阶段事件 Token 的累加值，可能包含父子摘要重复计数")
    event_total_cost_ms: int = Field(..., description="所有阶段事件耗时的累加值，不能当作端到端耗时")
    end_to_end_total_tokens: int | None = Field(None, description="根执行单元的实际总 Token；缺少根摘要时为 null")
    end_to_end_cost_ms: int | None = Field(None, description="根执行单元的实际总耗时；缺少根摘要时为 null")
    end_to_end_metric_source: Literal[
        "async_task", "agent_loop_summary", "rag_request_summary", "unavailable"
    ] = Field(
        ..., description="端到端指标的来源"
    )
    safety_interception_count: int = Field(..., description="因工具安全策略被 blocked 或 require_confirm 的次数")
    guardrail_stop_count: int = Field(..., description="因重复工具调用等通用护栏停止的次数")


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


class WorkOrderEvalExpected(BaseModel):
    category: Literal["consult", "complaint", "repair", "other"] = Field(..., description="人工标注的问题分类")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="人工标注的风险等级")
    need_human_review: bool = Field(..., description="人工标注是否需要人工复核")


class ConvertFailureSampleToEvalSampleRequest(BaseModel):
    # 前端可先选择数据集；不传两个字段时，由后端按失败样本的 schema 自动匹配默认数据集。
    dataset_id: str | None = Field(None, min_length=1, description="可选的目标评测数据集 ID")
    dataset_version: str | None = Field(None, min_length=1, description="可选的目标评测数据集版本")
    sample_type: Literal["normal", "boundary", "error"] = Field(
        "error",
        description="样本类型，描述输入场景而不是 expected 是否正确",
    )
    input_text: str = Field(..., min_length=1, max_length=4000, description="人工整理后的评测输入")
    expected: WorkOrderEvalExpected = Field(..., description="人工标注的正确期望结果")


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


class EvalGateCompareRequest(BaseModel):
    baseline_run_id: str = Field(..., min_length=1, description="已上线基线 Prompt 的评测运行 ID")
    candidate_run_id: str = Field(..., min_length=1, description="候选 Prompt 的评测运行 ID")


class AiEvalGateDecisionItem(BaseModel):
    gate_id: str
    baseline_run_id: str
    candidate_run_id: str
    prompt_name: str
    dataset_version: str
    decision: Literal["pass", "reject", "manual_review"]
    comparison: dict
    reasons: list[dict]
    rule_snapshot: dict
    created_at: str


class AiEvalGateDecisionPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiEvalGateDecisionItem]


class PublishPromptVersionRequest(BaseModel):
    gate_id: str = Field(..., min_length=1, description="本次发布依据的评测门禁 ID")
    approval_note: str = Field(..., min_length=5, max_length=1000, description="人工批准说明")
    # 当前 Python 服务未接入登录体系，先显式传入；后续应由 Java 透传的登录用户替代。
    approved_by: str = Field("manual_reviewer", min_length=1, max_length=64, description="批准人标识")


class AiPromptPublishAuditItem(BaseModel):
    publish_id: str
    gate_id: str
    prompt_id: str
    prompt_name: str
    candidate_prompt_version: str
    previous_prompt_version: str | None = None
    gate_decision: Literal["pass", "manual_review"]
    approval_note: str
    approved_by: str
    published_at: str


class AiPromptPublishAuditPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiPromptPublishAuditItem]


class RollbackPromptVersionRequest(BaseModel):
    rollback_reason: str = Field(..., min_length=5, max_length=1000, description="人工回滚原因")
    # 当前 Python 服务未接入登录体系，先显式传入；后续应由 Java 透传的登录用户替代。
    rolled_back_by: str = Field("manual_reviewer", min_length=1, max_length=64, description="回滚执行人标识")


class AiPromptRollbackAuditItem(BaseModel):
    rollback_id: str
    publish_id: str
    prompt_name: str
    rolled_back_prompt_version: str
    restored_prompt_version: str
    rollback_reason: str
    rolled_back_by: str
    rolled_back_at: str


class AiPromptRollbackAuditPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiPromptRollbackAuditItem]


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


class ToolCallingRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="用户问题")


class ToolDefinitionItem(BaseModel):
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具用途说明")
    tool_type: str = Field(..., description="工具类型，例如 ai_system_query/business_query/business_action")
    read_only: bool = Field(..., description="是否只读工具；只读工具不会修改业务数据")
    require_human_confirm: bool = Field(..., description="执行前是否需要人工确认")
    risk_level: str = Field(..., description="工具风险等级，例如 low/medium/high")
    parameters_schema: dict = Field(..., description="工具参数 JSON Schema")


class ToolCallDecisionItem(BaseModel):
    need_tool: bool = Field(..., description="模型判断是否需要调用工具")
    tool_name: str | None = Field(None, description="模型选择的工具名称")
    arguments: dict = Field(default_factory=dict, description="模型生成的工具参数")
    reason: str = Field(..., description="模型选择或不选择工具的原因")


class ToolCallingResponse(BaseModel):
    answer: str = Field(..., description="最终回答")
    decision: ToolCallDecisionItem = Field(..., description="工具选择决策")
    tool_result: dict | None = Field(None, description="后端工具执行结果；未调用工具时为空")
    available_tools: list[ToolDefinitionItem] = Field(default_factory=list, description="本次允许模型选择的工具白名单")
    model: str | None = Field(None, description="本次使用的模型")
    prompt_tokens: int | None = Field(None, ge=0, description="模型输入 Token 数")
    completion_tokens: int | None = Field(None, ge=0, description="模型输出 Token 数")
    total_tokens: int | None = Field(None, ge=0, description="模型调用总 Token 数")
    cost_ms: int | None = Field(None, ge=0, description="总耗时，单位毫秒")


class AgentLoopRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    max_steps: int = Field(3, ge=1, le=5, description="Agent 最大循环步数，防止无限循环和成本失控")


class AsyncAgentLoopTaskRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    max_steps: int = Field(3, ge=1, le=5, description="Agent 最大循环步数")


class AgentLoopStepItem(BaseModel):
    step_index: int = Field(..., ge=1, description="循环步序号，从 1 开始")
    action: Literal["call_tool", "final_answer"] = Field(..., description="本轮动作：调用工具或最终回答")
    tool_name: str | None = Field(None, description="本轮选择的工具名称")
    arguments: dict = Field(default_factory=dict, description="本轮工具参数")
    reason: str = Field(..., description="本轮决策原因")
    observation: dict | None = Field(None, description="工具执行或拦截后的观察结果")
    final_answer: str | None = Field(None, description="模型在本轮给出的最终回答")


class AgentLoopResponse(BaseModel):
    answer: str = Field(..., description="Agent Loop 最终回答")
    status: Literal["success", "max_steps_reached", "stopped_by_guardrail"] = Field(..., description="执行状态")
    steps: list[AgentLoopStepItem] = Field(default_factory=list, description="每一轮决策、行动和观察记录")
    available_tools: list[ToolDefinitionItem] = Field(default_factory=list, description="本次允许 Agent 使用的工具白名单")
    model: str | None = Field(None, description="本次使用的模型")
    prompt_tokens: int | None = Field(None, ge=0, description="模型输入 Token 数")
    completion_tokens: int | None = Field(None, ge=0, description="模型输出 Token 数")
    total_tokens: int | None = Field(None, ge=0, description="模型调用总 Token 数")
    cost_ms: int | None = Field(None, ge=0, description="总耗时，单位毫秒")


class AgentEvalGateCompareRequest(BaseModel):
    baseline_run_id: str = Field(..., min_length=1, description="基线 Agent Harness 运行 ID")
    candidate_run_id: str = Field(..., min_length=1, description="候选 Agent Harness 运行 ID")


class AsyncAgentLoopEvalTaskRequest(BaseModel):
    agent_version: str = Field(..., min_length=1, max_length=64, description="本次被评测 Agent 的版本标签")
    dataset_version: str = Field("agent_loop_v1", min_length=1, max_length=64, description="Agent 评测数据集版本")
    sample_limit: int | None = Field(None, ge=1, le=20, description="最多执行样本数，用于控制模型调用成本")


class AiAgentEvalRunItem(BaseModel):
    run_id: str
    agent_name: str
    agent_version: str
    dataset_version: str
    agent_snapshot_hash: str
    sample_count: int
    status_match_rate: float
    step_sequence_match_rate: float
    tool_call_accuracy: float
    observation_status_accuracy: float
    safety_case_pass_rate: float
    full_pass_rate: float
    avg_step_count: float | None = None
    avg_total_tokens: float | None = None
    avg_cost_ms: float | None = None
    metrics: dict | None = None
    created_at: str


class AiAgentEvalRunPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiAgentEvalRunItem]


class AiAgentEvalCaseResultItem(BaseModel):
    run_id: str
    sample_id: str
    sample_type: str
    status_match: bool
    step_sequence_match: bool
    tool_call_match: bool
    observation_status_match: bool
    answer_match: bool
    case_pass: bool
    actual_step_count: int
    total_tokens: int | None = None
    cost_ms: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    expected: dict | None = None
    actual: dict | None = None
    row: dict | None = None
    created_at: str


class AiAgentEvalCaseResultPageResponse(BaseModel):
    run_id: str
    total: int
    page: int
    page_size: int
    items: list[AiAgentEvalCaseResultItem]


class AiAgentEvalGateDecisionItem(BaseModel):
    gate_id: str
    baseline_run_id: str
    candidate_run_id: str
    agent_name: str
    dataset_version: str
    decision: Literal["pass", "reject", "manual_review"]
    comparison: dict
    reasons: list[dict]
    rule_snapshot: dict
    created_at: str


class AiAgentEvalGateDecisionPageResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AiAgentEvalGateDecisionItem]
