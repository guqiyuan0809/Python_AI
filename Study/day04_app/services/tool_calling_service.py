"""Day23 Tool Calling：受控工具注册、模型决策和后端执行。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException, ModelCallException
from day04_app.security.permissions import PERMISSION_KNOWLEDGE_READ, PERMISSION_TOOL_EXECUTE
from day04_app.security.principal import SYSTEM_PRINCIPAL, SecurityPrincipal
from day04_app.models import AiAsyncTask, AiStructuredResult, ChatSession, KnowledgeDocument
from day04_app.schemas.chat_schema import (
    ToolCallDecisionItem,
    ToolCallingResponse,
    ToolDefinitionItem,
)
from day04_app.services.chat_service import call_chat_completion, create_client, extract_json_object
from day04_app.services.call_log_service import create_call_log
from day04_app.services.tool_execution_context import ToolExecutionContext
from settings import settings


TOOL_CALLING_SYSTEM_PROMPT = """你是企业 AI 工具路由器。
你必须先判断用户问题是否需要调用后端工具。
只能从【可用工具】中选择工具，不能编造工具名，不能直接输出 SQL，不能请求执行未授权动作。
工具元信息中的 read_only、require_human_confirm、risk_level 用于提醒你工具边界。
如果用户请求匹配高风险或需要人工确认的工具，你仍然可以选择该工具并抽取参数；后端会决定是否拦截或进入人工确认。
你不能因为看见高风险工具就声称已经执行，也不能在没有工具结果时说业务动作已经完成。
如果用户只是问概念、解释、建议或普通聊天，不需要工具。
如果用户要求查询会话、异步任务、知识库文档、工单分析结果等实时业务状态，应该选择匹配工具。
你必须只输出一个合法 JSON 对象，不能输出 Markdown、解释或多余文本。
JSON 格式：
{
  "need_tool": true,
  "tool_name": "工具名或 null",
  "arguments": {"参数名": "参数值"},
  "reason": "选择原因"
}
"""


TOOL_FINAL_ANSWER_SYSTEM_PROMPT = """你是企业 AI 助手。
请只基于【工具决策结果】和【工具执行结果】回答用户问题。
如果工具没有执行、被策略拦截、等待人工确认或执行结果为空，必须明确说明“尚未执行”，不能声称操作已经完成。
如果工具结果显示未找到、失败或数据不足，要明确说明，不能编造业务数据。
回答要简洁、准确，适合直接展示给前端用户。"""


class ToolDecision(BaseModel):
    need_tool: bool = Field(..., description="是否需要调用工具")
    tool_name: str | None = Field(None, description="工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    reason: str = Field(..., min_length=1, max_length=500, description="决策原因")


class GetSessionStatusArgs(BaseModel):
    session_id: str = Field(..., min_length=1, description="会话 ID")


class GetAsyncTaskStatusArgs(BaseModel):
    task_id: str = Field(..., min_length=1, description="异步任务 ID")


class GetKnowledgeDocumentSummaryArgs(BaseModel):
    document_id: str = Field(..., min_length=1, description="知识库文档 ID")


class KnowledgeSearchArgs(BaseModel):
    """模型只能提供问题，不能决定可以访问哪些文档或知识域。"""

    # StructuredTool 的 schema 是模型唯一可见的入参契约。拒绝额外字段，避免
    # ``document_ids`` / ``domain_id`` 等伪造范围被 Pydantic 静默忽略。
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=1000, description="需要基于已授权知识库检索的问题")


class GetWorkOrderAnalysisResultArgs(BaseModel):
    business_id: str = Field(..., min_length=1, description="工单业务 ID")


class CloseWorkOrderArgs(BaseModel):
    business_id: str = Field(..., min_length=1, description="工单业务 ID")
    close_reason: str = Field(..., min_length=5, max_length=500, description="关闭原因")


class ToolPolicyResult(BaseModel):
    decision: Literal["allow", "block", "require_confirm"] = Field(..., description="策略决策结果")
    reason: str = Field(..., description="策略命中原因")
    matched_rules: list[str] = Field(default_factory=list, description="命中的策略规则")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    tool_type: str
    read_only: bool
    require_human_confirm: bool
    risk_level: str
    args_model: type[BaseModel]
    executor: Callable[[Session, BaseModel], dict[str, Any]]
    # 默认权限保持与 Day23 工具兼容；知识检索必须额外具有知识库读取权限。
    required_permissions: tuple[str, ...] = (PERMISSION_TOOL_EXECUTE,)

    def to_item(self) -> ToolDefinitionItem:
        return ToolDefinitionItem(
            name=self.name,
            description=self.description,
            tool_type=self.tool_type,
            read_only=self.read_only,
            require_human_confirm=self.require_human_confirm,
            risk_level=self.risk_level,
            parameters_schema=self.args_model.model_json_schema(),
        )


class ToolPolicyChecker:
    """工具执行策略校验器。

    课程阶段先用代码规则表达；企业中可以替换为数据库配置、配置中心或规则引擎。
    """

    def check(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        principal: SecurityPrincipal,
    ) -> ToolPolicyResult:
        matched_rules: list[str] = []

        # Service 层再次校验工具权限，避免未来非 HTTP 调用绕过 Router 的 RBAC。
        # 先保留既有工具执行权限的审计语义；知识检索等工具在此基础上再检查专属权限。
        if not principal.has_permissions(PERMISSION_TOOL_EXECUTE):
            return ToolPolicyResult(
                decision="block",
                reason="当前调用者没有执行受控工具的权限",
                matched_rules=["MISSING_TOOL_EXECUTE_PERMISSION"],
            )
        if not principal.has_permissions(*tool.required_permissions):
            return ToolPolicyResult(
                decision="block",
                reason="当前调用者缺少执行该受控工具所需权限",
                matched_rules=["MISSING_TOOL_SPECIFIC_PERMISSION"],
            )

        if not tool.read_only:
            matched_rules.append("WRITE_TOOL_REQUIRE_CONFIRM")
        if tool.require_human_confirm:
            matched_rules.append("TOOL_MARKED_REQUIRE_HUMAN_CONFIRM")
        if tool.risk_level != "low":
            matched_rules.append("NON_LOW_RISK_TOOL_REQUIRE_CONFIRM")

        if matched_rules:
            return ToolPolicyResult(
                decision="require_confirm",
                reason="该工具不是低风险自动执行工具，需要人工确认或额外权限审批",
                matched_rules=matched_rules,
            )

        return ToolPolicyResult(
            decision="allow",
            reason="低风险只读工具，允许自动执行",
            matched_rules=["LOW_RISK_READ_ONLY_AUTO_ALLOW"],
        )


tool_policy_checker = ToolPolicyChecker()


def _format_dt(value) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


def execute_get_session_status(db: Session, args: BaseModel) -> dict[str, Any]:
    typed_args = GetSessionStatusArgs.model_validate(args)
    session = db.scalars(
        select(ChatSession).where(ChatSession.session_id == typed_args.session_id)
    ).first()
    if session is None:
        return {
            "found": False,
            "message": "会话不存在",
            "session_id": typed_args.session_id,
        }
    return {
        "found": True,
        "session_id": session.session_id,
        "user_id": session.user_id,
        "title": session.title,
        "summary": session.summary,
        "status": session.status,
        "created_at": _format_dt(session.created_at),
        "updated_at": _format_dt(session.updated_at),
    }


def execute_get_async_task_status(db: Session, args: BaseModel) -> dict[str, Any]:
    typed_args = GetAsyncTaskStatusArgs.model_validate(args)
    task = db.scalars(
        select(AiAsyncTask).where(AiAsyncTask.task_id == typed_args.task_id)
    ).first()
    if task is None:
        return {
            "found": False,
            "message": "异步任务不存在",
            "task_id": typed_args.task_id,
        }
    return {
        "found": True,
        "task_id": task.task_id,
        "trace_id": task.trace_id,
        "session_id": task.session_id,
        "message_id": task.message_id,
        "broker_task_id": task.broker_task_id,
        "task_type": task.task_type,
        "status": task.status,
        "input_text": task.input_text,
        "result_text": task.result_text,
        "model": task.model,
        "prompt_tokens": task.prompt_tokens,
        "completion_tokens": task.completion_tokens,
        "total_tokens": task.total_tokens,
        "cost_ms": task.cost_ms,
        "retry_count": task.retry_count,
        "max_retries": task.max_retries,
        "error_type": task.error_type,
        "error_message": task.error_message,
        "created_at": _format_dt(task.created_at),
        "updated_at": _format_dt(task.updated_at),
    }


def execute_get_knowledge_document_summary(db: Session, args: BaseModel) -> dict[str, Any]:
    typed_args = GetKnowledgeDocumentSummaryArgs.model_validate(args)
    document = db.scalars(
        select(KnowledgeDocument).where(KnowledgeDocument.document_id == typed_args.document_id)
    ).first()
    if document is None:
        return {
            "found": False,
            "message": "知识库文档不存在",
            "document_id": typed_args.document_id,
        }
    return {
        "found": True,
        "document_id": document.document_id,
        "active_version_id": document.active_version_id,
        "original_file_name": document.original_file_name,
        "file_type": document.file_type,
        "file_size": document.file_size,
        "status": document.status,
        "parser_name": document.parser_name,
        "parsed_segment_count": document.parsed_segment_count,
        "chunk_status": document.chunk_status,
        "chunk_count": document.chunk_count,
        "chunked_at": _format_dt(document.chunked_at),
        "error_message": document.error_message,
        "created_at": _format_dt(document.created_at),
        "updated_at": _format_dt(document.updated_at),
    }


def _load_json_or_none(json_text: str | None) -> dict[str, Any] | None:
    if not json_text:
        return None
    return json.loads(json_text)


def execute_get_work_order_analysis_result(db: Session, args: BaseModel) -> dict[str, Any]:
    typed_args = GetWorkOrderAnalysisResultArgs.model_validate(args)
    # 这是一个“业务查询工具”示例：模型只能提供 business_id，真正的 SQL 查询由后端白名单工具执行。
    # 这里固定查询 work_order + work_order_analysis，避免模型通过参数越权查询其他业务结果。
    structured_result = db.scalars(
        select(AiStructuredResult)
        .where(
            AiStructuredResult.business_type == "work_order",
            AiStructuredResult.schema_type == "work_order_analysis",
            AiStructuredResult.business_id == typed_args.business_id,
        )
        .order_by(AiStructuredResult.id.desc())
    ).first()
    if structured_result is None:
        return {
            "found": False,
            "message": "未找到该工单的结构化分析结果",
            "business_id": typed_args.business_id,
        }
    return {
        "found": True,
        "result_id": structured_result.result_id,
        "task_id": structured_result.task_id,
        "trace_id": structured_result.trace_id,
        "session_id": structured_result.session_id,
        "message_id": structured_result.message_id,
        "business_type": structured_result.business_type,
        "business_id": structured_result.business_id,
        "schema_type": structured_result.schema_type,
        "schema_version": structured_result.schema_version,
        "status": structured_result.status,
        "result": _load_json_or_none(structured_result.result_json),
        "error_message": structured_result.error_message,
        "created_at": _format_dt(structured_result.created_at),
        "updated_at": _format_dt(structured_result.updated_at),
    }


def execute_close_work_order_demo(db: Session, args: BaseModel) -> dict[str, Any]:
    # 这是高风险写操作工具的演示 executor。
    # 正常情况下不会执行到这里，因为 execute_registered_tool 会先根据工具元信息拦截。
    # 后续如果真的要开放写操作，应在这里接入权限校验、人工确认单、幂等键和审计表。
    typed_args = CloseWorkOrderArgs.model_validate(args)
    return {
        "executed": False,
        "business_id": typed_args.business_id,
        "close_reason": typed_args.close_reason,
        "message": "演示工具不会真正关闭工单",
    }


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "get_session_status": ToolDefinition(
        name="get_session_status",
        description="根据 session_id 查询会话标题、摘要、状态和更新时间。",
        tool_type="ai_system_query",
        read_only=True,
        require_human_confirm=False,
        risk_level="low",
        args_model=GetSessionStatusArgs,
        executor=execute_get_session_status,
    ),
    "get_async_task_status": ToolDefinition(
        name="get_async_task_status",
        description="根据 task_id 查询异步任务状态、结果、错误原因、token 和耗时。",
        tool_type="ai_system_query",
        read_only=True,
        require_human_confirm=False,
        risk_level="low",
        args_model=GetAsyncTaskStatusArgs,
        executor=execute_get_async_task_status,
    ),
    "get_knowledge_document_summary": ToolDefinition(
        name="get_knowledge_document_summary",
        description="根据 document_id 查询知识库文档解析、切块、active 版本等基础状态。",
        tool_type="ai_system_query",
        read_only=True,
        require_human_confirm=False,
        risk_level="low",
        args_model=GetKnowledgeDocumentSummaryArgs,
        executor=execute_get_knowledge_document_summary,
    ),
    "knowledge_search": ToolDefinition(
        name="knowledge_search",
        description=(
            "在当前登录用户已授权的园区安全知识域中检索法规、隐患整改标准和处置资料，"
            "返回带来源引用的证据与回答。模型只能传 question，不能指定或扩大文档范围。"
        ),
        tool_type="knowledge_retrieval",
        read_only=True,
        require_human_confirm=False,
        risk_level="low",
        args_model=KnowledgeSearchArgs,
        # 真正的 LlamaIndex executor 需要 Runtime 中的可信身份、数据范围与 Trace，
        # 因此不通过这个通用 executor 直接调用，而由 LangChain Tool 适配层显式绑定。
        executor=lambda _db, _args: (_ for _ in ()).throw(
            RuntimeError("knowledge_search 必须通过 LangChain 受治理适配器执行")
        ),
        required_permissions=(PERMISSION_TOOL_EXECUTE, PERMISSION_KNOWLEDGE_READ),
    ),
    "get_work_order_analysis_result": ToolDefinition(
        name="get_work_order_analysis_result",
        description="根据工单业务 ID 查询最新的工单结构化分析结果，包括分类、风险等级、摘要、建议和是否需要人工复核。",
        tool_type="business_query",
        read_only=True,
        require_human_confirm=False,
        risk_level="low",
        args_model=GetWorkOrderAnalysisResultArgs,
        executor=execute_get_work_order_analysis_result,
    ),
    "close_work_order_demo": ToolDefinition(
        name="close_work_order_demo",
        description="演示用高风险动作工具：根据工单业务 ID 和关闭原因关闭工单。该工具不是只读操作，必须人工确认，当前不会被自动执行。",
        tool_type="business_action",
        read_only=False,
        require_human_confirm=True,
        risk_level="high",
        args_model=CloseWorkOrderArgs,
        executor=execute_close_work_order_demo,
    ),
}


def list_available_tools() -> list[ToolDefinitionItem]:
    return [tool.to_item() for tool in TOOL_REGISTRY.values()]


def _build_tool_choice_messages(message: str) -> list[dict[str, str]]:
    tools_text = json.dumps(
        [tool.model_dump() for tool in list_available_tools()],
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": TOOL_CALLING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"【可用工具】\n{tools_text}\n\n【用户问题】\n{message}",
        },
    ]


def _parse_tool_decision(raw_text: str) -> ToolDecision:
    json_text = extract_json_object(raw_text)
    decision = ToolDecision.model_validate_json(json_text)
    if not decision.need_tool:
        return decision
    if not decision.tool_name:
        raise ModelCallException(message="模型要求调用工具但未提供 tool_name")
    if decision.tool_name not in TOOL_REGISTRY:
        raise ModelCallException(message=f"模型选择了未注册工具：{decision.tool_name}")
    return decision


def execute_registered_tool(
    db: Session,
    decision: ToolDecision,
    principal: SecurityPrincipal | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, Any] | None:
    if not decision.need_tool:
        return None
    if decision.tool_name not in TOOL_REGISTRY:
        raise BusinessException(code=40090, message="工具不存在或未授权")
    tool = TOOL_REGISTRY[decision.tool_name]
    # 执行工具前必须先过策略校验层。
    # 当前策略允许低风险只读工具自动执行；高风险或写操作工具返回 require_confirm，不直接执行。
    # Harness/Worker 内部调用没有 HTTP 请求时使用 system Principal；在线请求必须显式透传 Principal。
    effective_principal = principal or SYSTEM_PRINCIPAL
    policy_result = tool_policy_checker.check(tool, decision.arguments, effective_principal)
    if policy_result.decision != "allow":
        return {
            "tool_name": tool.name,
            "arguments": decision.arguments,
            "status": policy_result.decision,
            "blocked_reason": policy_result.reason,
            "matched_rules": policy_result.matched_rules,
            "tool_metadata": {
                "tool_type": tool.tool_type,
                "read_only": tool.read_only,
                "require_human_confirm": tool.require_human_confirm,
                "risk_level": tool.risk_level,
            },
            "data": None,
        }
    try:
        args = tool.args_model.model_validate(decision.arguments)
    except ValidationError as exc:
        raise BusinessException(code=40091, message=f"工具参数不合法：{exc.errors()[0]['msg']}") from exc
    if tool.name == "knowledge_search":
        # 此工具需要可信的数据范围和 Trace 关联，不能通过 ToolDefinition 的通用两参
        # executor 获得；在统一策略与 Pydantic 校验之后显式进入其受治理实现。
        from day04_app.services.knowledge_search_tool_service import execute_knowledge_search_tool

        return execute_knowledge_search_tool(
            db,
            args,
            effective_principal,
            execution_context or ToolExecutionContext(),
        )
    return {
        "tool_name": tool.name,
        "arguments": args.model_dump(),
        "data": tool.executor(db, args),
    }


def _generate_final_answer(
    message: str,
    decision: ToolDecision,
    tool_result: dict[str, Any] | None,
) -> tuple[str, int, int, int]:
    client = create_client(timeout=30.0)
    if tool_result is None:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是简洁专业的企业 AI 助手。"
                    "请结合【工具决策结果】回答用户问题。"
                    "如果模型决策说明不需要或不能调用工具，不能声称已经执行任何业务操作。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"【用户问题】\n{message}\n\n"
                    f"【工具决策结果】\n{decision.model_dump_json()}"
                ),
            },
        ]
    else:
        messages = [
            {"role": "system", "content": TOOL_FINAL_ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"【用户问题】\n{message}\n\n"
                    f"【工具决策结果】\n{decision.model_dump_json()}\n\n"
                    f"【工具执行结果】\n{json.dumps(tool_result, ensure_ascii=False, indent=2)}"
                ),
            },
        ]
    response = call_chat_completion(
        client,
        messages,
        model=settings.dashscope_model,
        temperature=0.1,
        max_tokens=500,
    )
    usage = response.usage
    return (
        (response.choices[0].message.content or "").strip(),
        getattr(usage, "prompt_tokens", 0),
        getattr(usage, "completion_tokens", 0),
        getattr(usage, "total_tokens", 0),
    )


def _safe_create_tool_calling_log(
    db: Session,
    *,
    trace_id: str | None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_ms: int | None = None,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """记录 Tool Calling 调用日志；日志失败不能反向影响主流程。"""
    try:
        create_call_log(
            db,
            call_type="tool_calling",
            trace_id=trace_id,
            model=settings.dashscope_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_ms=cost_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
        )
    except Exception:
        # 可观测日志是旁路能力，不能因为日志写入失败导致一次本来成功的工具调用失败。
        db.rollback()


def answer_with_tool_calling(
    db: Session,
    message: str,
    trace_id: str | None = None,
    principal: SecurityPrincipal | None = None,
) -> ToolCallingResponse:
    start_time = time.perf_counter()
    client = create_client(timeout=30.0)
    try:
        # 第一次调用模型：让模型只做“工具路由决策”，判断是否需要调用后端工具。
        # 注意这里还不执行业务逻辑，模型只负责输出 need_tool/tool_name/arguments。
        decision_response = call_chat_completion(
            client,
            _build_tool_choice_messages(message),
            model=settings.dashscope_model,
            temperature=0.0,
            max_tokens=300,
        )

        # 将模型返回的 JSON 字符串解析成 ToolDecision，并校验工具名必须在白名单中。
        # 这一步类似 Java 中把模型返回值反序列化成 DTO 后再做业务校验。
        decision = _parse_tool_decision(decision_response.choices[0].message.content or "")

        # 如果模型判断需要工具，则后端执行真正的工具方法；如果不需要工具，则返回 None。
        # 模型不能直接访问数据库，真实查询只能通过这里的白名单工具完成。
        tool_result = execute_registered_tool(db, decision, principal=principal)

        # 第二次调用模型：把“用户问题 + 工具执行结果”交给模型，生成最终给前端展示的自然语言回答。
        # 如果 tool_result 为 None，也会携带第一轮决策结果，避免模型误以为某个业务动作已经执行。
        final_answer, final_prompt_tokens, final_completion_tokens, final_total_tokens = (
            _generate_final_answer(message, decision, tool_result)
        )

        # 汇总两次模型调用的 token：
        # 1）第一次是工具路由决策；
        # 2）第二次是最终回答生成。
        # 企业里看成本时要把这两段都算进去，否则 Tool Calling 的真实成本会被低估。
        decision_usage = decision_response.usage
        prompt_tokens = (getattr(decision_usage, "prompt_tokens", 0) or 0) + final_prompt_tokens
        completion_tokens = (
            (getattr(decision_usage, "completion_tokens", 0) or 0)
            + final_completion_tokens
        )
        total_tokens = (getattr(decision_usage, "total_tokens", 0) or 0) + final_total_tokens

        # 统计完整链路耗时：包含工具路由模型调用、后端工具执行、最终回答模型调用。
        cost_ms = round((time.perf_counter() - start_time) * 1000)

        # 写入统一 AI 调用日志，方便后续通过 trace_id 排查本次 Tool Calling 的成本和耗时。
        _safe_create_tool_calling_log(
            db,
            trace_id=trace_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_ms=cost_ms,
            status="success",
        )
        return ToolCallingResponse(
            answer=final_answer,
            decision=ToolCallDecisionItem(
                need_tool=decision.need_tool,
                tool_name=decision.tool_name,
                arguments=decision.arguments,
                reason=decision.reason,
            ),
            tool_result=tool_result,
            available_tools=list_available_tools(),
            model=settings.dashscope_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_ms=cost_ms,
        )
    except (BusinessException, ModelCallException) as exc:
        # 已知业务异常或模型异常：保留原始异常类型，同时补充一条失败调用日志。
        _safe_create_tool_calling_log(
            db,
            trace_id=trace_id,
            cost_ms=round((time.perf_counter() - start_time) * 1000),
            status="error",
            error_type=getattr(exc, "error_type", type(exc).__name__),
            error_message=getattr(exc, "message", str(exc)),
        )
        raise
    except Exception as exc:
        # 未预期异常统一包装成 ModelCallException，避免把底层堆栈细节直接暴露给前端。
        _safe_create_tool_calling_log(
            db,
            trace_id=trace_id,
            cost_ms=round((time.perf_counter() - start_time) * 1000),
            status="error",
            error_type=type(exc).__name__,
            error_message=f"Tool Calling 执行失败：{type(exc).__name__}",
        )
        raise ModelCallException(message=f"Tool Calling 执行失败：{type(exc).__name__}") from exc
