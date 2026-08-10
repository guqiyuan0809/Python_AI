"""Day24 Agent Loop：受控的感知、决策、行动、观察反馈循环。"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException, ModelCallException
from day04_app.schemas.chat_schema import (
    AgentLoopResponse,
    AgentLoopStepItem,
)
from day04_app.services.call_log_service import create_call_log
from day04_app.services.chat_service import call_chat_completion, create_client, extract_json_object
from day04_app.services.eval_master_service import get_active_prompt_version_for_runtime
from day04_app.services.prompt_observability_service import (
    build_registry_prompt_identity,
    render_prompt_template,
)
from day04_app.services.tool_calling_service import (
    TOOL_REGISTRY,
    ToolDecision,
    execute_registered_tool,
    list_available_tools,
)
from day04_app.security.principal import SecurityPrincipal
from settings import settings


AGENT_LOOP_SYSTEM_PROMPT = """你是企业 AI Agent 的受控决策器。
你需要在有限循环内完成用户目标，但必须遵守后端工具边界。

你每一轮只能输出一个合法 JSON 对象，不能输出 Markdown 或额外解释。

可选 action：
1. call_tool：当你需要读取真实业务系统状态时，选择一个可用工具并给出参数。
2. final_answer：当已有信息足够回答，或工具被拦截/无法继续时，给出最终回答。

安全规则：
- 只能选择【可用工具】中的工具，不能编造工具名。
- 不能直接输出 SQL，不能请求执行未授权动作。
- 如果用户已经明确请求执行高风险动作，且目标工具的必填参数已齐全，应直接调用该高风险工具，由后端策略层统一拦截并返回 require_confirm；不要先调用额外的只读查询工具确认目标是否存在。
- 如果工具观察结果 status=blocked 或 status=require_confirm，必须停止继续执行高风险动作，并用 final_answer 告知用户需要人工确认。
- 如果工具观察结果 status=not_found，必须 final_answer 告知用户未找到匹配数据，不要重复调用相同工具。
- 如果工具观察结果 status=error，必须 final_answer 告知用户工具执行失败或稍后重试，不要编造结果。
- 不要重复调用相同工具和相同参数；如果观察结果已足够，应直接 final_answer。
- 你可以从上一轮 observation.data 中提取字段作为下一轮工具参数，例如先查任务得到 session_id，再用 session_id 查询会话状态。
- 如果用户目标需要多个信息来源，应该按顺序调用不同工具，并在信息足够后 final_answer 汇总回答。

JSON 格式：
调用工具时：
{
  "action": "call_tool",
  "tool_name": "get_async_task_status",
  "arguments": {"参数名": "参数值"},
  "reason": "本轮决策原因",
  "final_answer": null
}
最终回答时：
{
  "action": "final_answer",
  "tool_name": null,
  "arguments": {},
  "reason": "本轮决策原因",
  "final_answer": "最终回答内容"
}
如果要调用工具，action 必须等于 "call_tool"。
如果要最终回答，action 必须等于 "final_answer"。
"""


# Prompt 负责开放式决策；已识别的高风险写操作则必须走确定性路由，
# 确保模型不会通过额外查询绕开“请求 -> 策略拦截”的安全验证路径。
AGENT_DECISION_POLICY_VERSION = "agent-loop-policy-v3"
AGENT_DECISION_PROMPT_NAME = "agent_decision"
# 仅作为迁移初始化和代码审查基线；线上决策必须读取 ai_prompt_version 的 active 版本。
AGENT_DECISION_USER_PROMPT_TEMPLATE = (
    "【用户目标】\n{message}\n\n"
    "【最大循环步数】\n{max_steps}\n\n"
    "【可用工具】\n{tools}\n\n"
    "【已完成步骤和观察结果】\n{steps}"
)
_CLOSE_WORK_ORDER_INTENT_PATTERN = re.compile(
    r"^\s*(?:请|帮我|麻烦)?\s*关闭\s*工单\s+"
    r"(?P<business_id>[A-Za-z0-9_-]+)\s*[，,]?\s*"
    r"关闭原因\s*(?:是|为|:|：)\s*(?P<close_reason>.+?)\s*[。！？!?]?\s*$",
    re.DOTALL,
)


class AgentLoopDecision(BaseModel):
    action: Literal["call_tool", "final_answer"] = Field(..., description="本轮动作")
    tool_name: str | None = Field(None, description="工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    reason: str = Field(..., min_length=1, max_length=500, description="决策原因")
    final_answer: str | None = Field(None, description="最终回答")


def _normalize_tool_observation(
    *,
    tool_name: str | None,
    arguments: dict[str, Any],
    raw_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """把不同工具的原始返回包装成统一 observation。

    统一状态能降低下一轮模型理解成本，也方便前端展示和后续审计分析。
    """
    if raw_result is None:
        return {
            "status": "error",
            "tool_name": tool_name,
            "arguments": arguments,
            "message": "工具未返回结果",
            "data": None,
            "raw_result": None,
        }

    raw_status = raw_result.get("status")
    if raw_status in {"require_confirm", "blocked", "stopped_by_guardrail", "error"}:
        return {
            "status": raw_status,
            "tool_name": raw_result.get("tool_name", tool_name),
            "arguments": raw_result.get("arguments", arguments),
            "message": (
                raw_result.get("blocked_reason")
                or raw_result.get("message")
                or "工具未执行或被系统拦截"
            ),
            "data": raw_result.get("data"),
            "matched_rules": raw_result.get("matched_rules", []),
            "tool_metadata": raw_result.get("tool_metadata"),
            "raw_result": raw_result,
        }

    data = raw_result.get("data")
    if isinstance(data, dict) and data.get("found") is False:
        return {
            "status": "not_found",
            "tool_name": raw_result.get("tool_name", tool_name),
            "arguments": raw_result.get("arguments", arguments),
            "message": data.get("message") or "未找到匹配数据",
            "data": None,
            "raw_result": raw_result,
        }

    return {
        "status": "success",
        "tool_name": raw_result.get("tool_name", tool_name),
        "arguments": raw_result.get("arguments", arguments),
        "message": "工具执行成功",
        "data": data if data is not None else raw_result,
        "raw_result": raw_result,
    }


def _build_tool_error_observation(
    *,
    tool_name: str | None,
    arguments: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    message = getattr(exc, "message", str(exc))
    return {
        "status": "error",
        "tool_name": tool_name,
        "arguments": arguments,
        "message": message,
        "error_type": getattr(exc, "error_type", type(exc).__name__),
        "error_code": getattr(exc, "code", None),
        "data": None,
        "raw_result": None,
    }


def _is_terminal_observation(observation: dict[str, Any]) -> bool:
    # 这些状态已经足够决定“不能继续自动行动”，无需再烧一次模型让它做 final_answer。
    return observation.get("status") in {
        "not_found",
        "require_confirm",
        "blocked",
        "error",
        "stopped_by_guardrail",
    }


def _build_terminal_observation_answer(observation: dict[str, Any]) -> str:
    status = observation.get("status")
    message = observation.get("message") or "工具执行结果不足"
    tool_name = observation.get("tool_name") or "未知工具"

    if status == "not_found":
        return f"未找到匹配数据：{message}"
    if status == "require_confirm":
        return f"工具 {tool_name} 尚未执行：{message}"
    if status == "blocked":
        return f"工具 {tool_name} 已被系统拦截：{message}"
    if status == "error":
        return f"工具 {tool_name} 执行失败：{message}"
    if status == "stopped_by_guardrail":
        return f"系统已停止继续执行：{message}"
    return f"系统已停止继续执行：{message}"


def _make_tool_call_key(tool_name: str | None, arguments: dict[str, Any]) -> str:
    # 将“工具名 + 参数”转成稳定字符串，用于判断 Agent 是否重复调用同一个工具。
    # sort_keys=True 可以避免 {"a":1,"b":2} 和 {"b":2,"a":1} 被误认为不同调用。
    return json.dumps(
        {
            "tool_name": tool_name,
            "arguments": arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _build_agent_loop_messages(
    *,
    message: str,
    steps: list[AgentLoopStepItem],
    max_steps: int,
    prompt,
) -> list[dict[str, str]]:
    tools_text = json.dumps(
        [tool.model_dump() for tool in list_available_tools()],
        ensure_ascii=False,
        indent=2,
    )
    steps_text = json.dumps(
        [step.model_dump() for step in steps],
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": prompt.system_prompt},
        {
            "role": "user",
            "content": render_prompt_template(
                prompt.user_prompt_template,
                message=message,
                max_steps=max_steps,
                tools=tools_text,
                steps=steps_text,
            ),
        },
    ]


def get_active_agent_decision_prompt(db: Session):
    return get_active_prompt_version_for_runtime(db, AGENT_DECISION_PROMPT_NAME)


def _get_agent_decision_prompt_identity(prompt):
    return build_registry_prompt_identity(prompt)


def _get_tool_catalog_hash() -> str:
    """工具定义也会影响真实模型输入，记录哈希而不重复保存完整工具描述。"""
    canonical = json.dumps(
        [tool.model_dump() for tool in list_available_tools()],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_deterministic_high_risk_decision(message: str) -> AgentLoopDecision | None:
    """只为参数完整的关闭工单演示动作建立确定性路由。

    该路由不执行工具，仍由 execute_registered_tool 触发 ToolPolicyChecker，
    因此高风险动作最终只会得到 require_confirm，不能绕过人工确认。
    """
    tool = TOOL_REGISTRY.get("close_work_order_demo")
    if tool is None or tool.read_only or not tool.require_human_confirm:
        return None

    match = _CLOSE_WORK_ORDER_INTENT_PATTERN.match(message)
    if match is None:
        return None

    arguments = {
        "business_id": match.group("business_id"),
        "close_reason": match.group("close_reason").strip(),
    }
    try:
        validated_args = tool.args_model.model_validate(arguments)
    except ValidationError:
        # 不完整或不合法的请求保留给模型澄清，不能猜测业务参数。
        return None

    return AgentLoopDecision(
        action="call_tool",
        tool_name=tool.name,
        arguments=validated_args.model_dump(),
        reason="命中参数完整的高风险关闭工单请求，交由后端策略层进行人工确认拦截。",
        final_answer=None,
    )


def _parse_agent_loop_decision(raw_text: str) -> AgentLoopDecision:
    json_text = extract_json_object(raw_text)
    # 模型偶尔会把 action 输出成中文或带解释的值；先做一次轻量归一化，再交给 Pydantic 强校验。
    # 这类似 Java 里在反序列化 DTO 前先做字段兼容处理，但最终仍然靠枚举校验兜底。
    payload = json.loads(json_text)
    action = str(payload.get("action", "")).strip().lower()
    if action in {"调用工具", "tool", "use_tool", "call tool", "call_tool"}:
        payload["action"] = "call_tool"
    elif action in {"最终回答", "回答", "final", "answer", "final answer", "final_answer"}:
        payload["action"] = "final_answer"
    decision = AgentLoopDecision.model_validate(payload)
    if decision.action == "call_tool":
        if not decision.tool_name:
            raise ModelCallException(message="Agent 选择调用工具但未提供 tool_name")
    if decision.action == "final_answer" and not decision.final_answer:
        raise ModelCallException(message="Agent 选择最终回答但未提供 final_answer")
    return decision


def _safe_create_agent_loop_log(
    db: Session,
    *,
    trace_id: str | None,
    task_id: str | None = None,
    run_id: str | None = None,
    model: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_ms: int | None = None,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    try:
        create_call_log(
            db,
            call_type="agent_loop",
            trace_id=trace_id,
            task_id=task_id,
            run_id=run_id,
            stage="agent_loop_summary",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_ms=cost_ms,
            status=status,
            error_type=error_type,
            error_message=error_message,
            detail={"summary": True},
        )
    except Exception:
        # Agent 主流程不能因为日志写入失败而失败；回滚当前日志事务即可。
        db.rollback()


def run_agent_loop(
    db: Session,
    *,
    message: str,
    max_steps: int = 3,
    trace_id: str | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
    sample_id: str | None = None,
    decision_prompt=None,
    principal: SecurityPrincipal | None = None,
) -> AgentLoopResponse:
    start_time = time.perf_counter()
    steps: list[AgentLoopStepItem] = []
    called_tool_keys: set[str] = set()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    runtime_prompt = decision_prompt
    runtime_model = (
        (runtime_prompt.model or settings.dashscope_model)
        if runtime_prompt is not None
        else None
    )

    try:
        for step_index in range(1, max_steps + 1):
            # 已识别且参数完整的高风险动作不交给模型自由规划，确保先经过策略拦截。
            decision = (
                _build_deterministic_high_risk_decision(message)
                if step_index == 1
                else None
            )
            if decision is None:
                # 其他场景仍由模型决定下一步 action；真正执行工具仍然由后端安全门控制。
                if runtime_prompt is None:
                    # 同一 Loop 首次调用时固定 active Prompt，后续轮次不受并发发布影响。
                    runtime_prompt = get_active_agent_decision_prompt(db)
                runtime_model = runtime_prompt.model or settings.dashscope_model
                client = create_client(timeout=30.0)
                decision_start_time = time.perf_counter()
                try:
                    response = call_chat_completion(
                        client,
                        _build_agent_loop_messages(
                            message=message,
                            steps=steps,
                            max_steps=max_steps,
                            prompt=runtime_prompt,
                        ),
                        model=runtime_model,
                        temperature=(
                            runtime_prompt.temperature
                            if runtime_prompt.temperature is not None
                            else 0.0
                        ),
                        max_tokens=runtime_prompt.max_tokens or 500,
                    )
                    usage = response.usage
                    prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                    total_tokens += getattr(usage, "total_tokens", 0) or 0
                    decision = _parse_agent_loop_decision(response.choices[0].message.content or "")
                except Exception as exc:
                    create_call_log(
                        db,
                        call_type="agent_loop",
                        stage="agent_model_decision",
                        trace_id=trace_id,
                        task_id=task_id,
                        run_id=run_id,
                        model=runtime_model,
                        cost_ms=round((time.perf_counter() - decision_start_time) * 1000),
                        status="error",
                        error_type=getattr(exc, "error_type", type(exc).__name__),
                        error_message=getattr(
                            exc,
                            "message",
                            f"Agent 模型决策失败：{type(exc).__name__}",
                        ),
                        **_get_agent_decision_prompt_identity(runtime_prompt).as_call_log_fields(),
                        detail={
                            "step_index": step_index,
                            "sample_id": sample_id,
                            "decision_source": "model",
                            "prompt_source": "database",
                            "decision_policy_version": AGENT_DECISION_POLICY_VERSION,
                            "tool_catalog_hash": _get_tool_catalog_hash(),
                        },
                    )
                    raise
                create_call_log(
                    db,
                    call_type="agent_loop",
                    stage="agent_model_decision",
                    trace_id=trace_id,
                    task_id=task_id,
                    run_id=run_id,
                    model=runtime_model,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                    cost_ms=round((time.perf_counter() - decision_start_time) * 1000),
                    **_get_agent_decision_prompt_identity(runtime_prompt).as_call_log_fields(),
                    detail={
                        "step_index": step_index,
                        "sample_id": sample_id,
                        "decision_source": "model",
                        "action": decision.action,
                        "tool_name": decision.tool_name,
                        "prompt_source": "database",
                        "decision_policy_version": AGENT_DECISION_POLICY_VERSION,
                        "tool_catalog_hash": _get_tool_catalog_hash(),
                    },
                )
            else:
                create_call_log(
                    db,
                    call_type="agent_loop",
                    stage="agent_route_decision",
                    trace_id=trace_id,
                    task_id=task_id,
                    run_id=run_id,
                    detail={
                        "step_index": step_index,
                        "sample_id": sample_id,
                        "decision_source": "deterministic_high_risk_route",
                        "action": decision.action,
                        "tool_name": decision.tool_name,
                        "prompt_source": "none",
                    },
                )

            if decision.action == "final_answer":
                step = AgentLoopStepItem(
                    step_index=step_index,
                    action="final_answer",
                    tool_name=None,
                    arguments={},
                    reason=decision.reason,
                    observation=None,
                    final_answer=decision.final_answer,
                )
                steps.append(step)
                cost_ms = round((time.perf_counter() - start_time) * 1000)
                _safe_create_agent_loop_log(
                    db,
                    trace_id=trace_id,
                    task_id=task_id,
                    run_id=run_id,
                    model=runtime_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_ms=cost_ms,
                    status="success",
                )
                return AgentLoopResponse(
                    answer=decision.final_answer or "",
                    status="success",
                    steps=steps,
                    available_tools=list_available_tools(),
                    model=runtime_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_ms=cost_ms,
                )

            # call_tool 动作会被转换成 Day23 的 ToolDecision，复用同一套白名单、参数校验和风险拦截。
            tool_call_key = _make_tool_call_key(decision.tool_name, decision.arguments)
            if tool_call_key in called_tool_keys:
                # 后端护栏：即使 prompt 已要求不要重复调用，仍要在代码里防止 Agent 陷入重复动作。
                # 一旦发现相同工具和相同参数重复调用，立刻停止循环，避免浪费 token 和重复打业务接口。
                observation = {
                    "status": "stopped_by_guardrail",
                    "guardrail": "duplicate_tool_call",
                    "message": "检测到 Agent 重复调用相同工具和相同参数，系统已停止继续执行",
                    "tool_name": decision.tool_name,
                    "arguments": decision.arguments,
                }
                steps.append(
                    AgentLoopStepItem(
                        step_index=step_index,
                        action="call_tool",
                        tool_name=decision.tool_name,
                        arguments=decision.arguments,
                        reason=decision.reason,
                        observation=observation,
                        final_answer=None,
                    )
                )
                cost_ms = round((time.perf_counter() - start_time) * 1000)
                answer = "检测到 Agent 重复调用相同工具和相同参数，系统已停止继续执行。请根据已有观察结果人工确认下一步。"
                _safe_create_agent_loop_log(
                    db,
                    trace_id=trace_id,
                    task_id=task_id,
                    run_id=run_id,
                    model=runtime_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_ms=cost_ms,
                    status="success",
                )
                return AgentLoopResponse(
                    answer=answer,
                    status="stopped_by_guardrail",
                    steps=steps,
                    available_tools=list_available_tools(),
                    model=runtime_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_ms=cost_ms,
                )

            called_tool_keys.add(tool_call_key)
            tool_decision = ToolDecision(
                need_tool=True,
                tool_name=decision.tool_name,
                arguments=decision.arguments,
                reason=decision.reason,
            )
            try:
                tool_start_time = time.perf_counter()
                raw_observation = execute_registered_tool(
                    db,
                    tool_decision,
                    principal=principal,
                )
                observation = _normalize_tool_observation(
                    tool_name=decision.tool_name,
                    arguments=decision.arguments,
                    raw_result=raw_observation,
                )
            except (BusinessException, ModelCallException) as exc:
                # 工具执行失败属于本轮 action 的观察结果，不让整个 Agent Loop 直接炸掉。
                # 例如参数不合法、工具不存在、权限策略拒绝等，都转成 error observation 后确定性收口。
                observation = _build_tool_error_observation(
                    tool_name=decision.tool_name,
                    arguments=decision.arguments,
                    exc=exc,
                )
            create_call_log(
                db,
                call_type="agent_loop",
                stage="agent_tool_execution",
                trace_id=trace_id,
                task_id=task_id,
                run_id=run_id,
                cost_ms=round((time.perf_counter() - tool_start_time) * 1000),
                status="error" if observation["status"] == "error" else "success",
                error_type=observation.get("error_type"),
                error_message=observation.get("message") if observation["status"] == "error" else None,
                detail={
                    "step_index": step_index,
                    "sample_id": sample_id,
                    "tool_name": decision.tool_name,
                    "argument_names": sorted(decision.arguments),
                    "observation_status": observation["status"],
                    "matched_rules": observation.get("matched_rules", []),
                },
            )
            steps.append(
                AgentLoopStepItem(
                    step_index=step_index,
                    action="call_tool",
                    tool_name=decision.tool_name,
                    arguments=decision.arguments,
                    reason=decision.reason,
                    observation=observation,
                    final_answer=None,
                )
            )
            if _is_terminal_observation(observation):
                # 终止态 observation 由后端直接确定性收口，避免多调用一轮模型造成成本浪费或误回答。
                cost_ms = round((time.perf_counter() - start_time) * 1000)
                answer = _build_terminal_observation_answer(observation)
                _safe_create_agent_loop_log(
                    db,
                    trace_id=trace_id,
                    task_id=task_id,
                    run_id=run_id,
                    model=runtime_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_ms=cost_ms,
                    status="success",
                )
                return AgentLoopResponse(
                    answer=answer,
                    status="success",
                    steps=steps,
                    available_tools=list_available_tools(),
                    model=runtime_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_ms=cost_ms,
                )

        # 达到最大循环次数时强制停止，避免 Agent 无限制调用模型和工具。
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        answer = "已达到最大 Agent 循环步数，系统已停止继续执行。请根据已有步骤结果人工确认下一步。"
        _safe_create_agent_loop_log(
            db,
            trace_id=trace_id,
            task_id=task_id,
            run_id=run_id,
            model=runtime_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_ms=cost_ms,
            status="success",
        )
        return AgentLoopResponse(
            answer=answer,
            status="max_steps_reached",
            steps=steps,
            available_tools=list_available_tools(),
            model=runtime_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_ms=cost_ms,
        )
    except (BusinessException, ModelCallException) as exc:
        _safe_create_agent_loop_log(
            db,
            trace_id=trace_id,
            task_id=task_id,
            run_id=run_id,
            model=runtime_model,
            cost_ms=round((time.perf_counter() - start_time) * 1000),
            status="error",
            error_type=getattr(exc, "error_type", type(exc).__name__),
            error_message=getattr(exc, "message", str(exc)),
        )
        raise
    except Exception as exc:
        _safe_create_agent_loop_log(
            db,
            trace_id=trace_id,
            task_id=task_id,
            run_id=run_id,
            model=runtime_model,
            cost_ms=round((time.perf_counter() - start_time) * 1000),
            status="error",
            error_type=type(exc).__name__,
            error_message=f"Agent Loop 执行失败：{type(exc).__name__}",
        )
        raise ModelCallException(message=f"Agent Loop 执行失败：{type(exc).__name__}") from exc
