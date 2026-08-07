"""Day24 Agent Loop：受控的感知、决策、行动、观察反馈循环。"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException, ModelCallException
from day04_app.schemas.chat_schema import (
    AgentLoopResponse,
    AgentLoopStepItem,
)
from day04_app.services.call_log_service import create_call_log
from day04_app.services.chat_service import call_chat_completion, create_client, extract_json_object
from day04_app.services.tool_calling_service import (
    ToolDecision,
    execute_registered_tool,
    list_available_tools,
)
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
- 如果工具观察结果 status=blocked，必须停止继续执行高风险动作，并用 final_answer 告知用户需要人工确认。
- 如果工具结果显示未找到或数据不足，可以选择 final_answer 说明现有信息不足。
- 不要重复调用相同工具和相同参数；如果观察结果已足够，应直接 final_answer。

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


class AgentLoopDecision(BaseModel):
    action: Literal["call_tool", "final_answer"] = Field(..., description="本轮动作")
    tool_name: str | None = Field(None, description="工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    reason: str = Field(..., min_length=1, max_length=500, description="决策原因")
    final_answer: str | None = Field(None, description="最终回答")


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
        {"role": "system", "content": AGENT_LOOP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"【用户目标】\n{message}\n\n"
                f"【最大循环步数】\n{max_steps}\n\n"
                f"【可用工具】\n{tools_text}\n\n"
                f"【已完成步骤和观察结果】\n{steps_text}"
            ),
        },
    ]


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
        # Agent 主流程不能因为日志写入失败而失败；回滚当前日志事务即可。
        db.rollback()


def run_agent_loop(
    db: Session,
    *,
    message: str,
    max_steps: int = 3,
    trace_id: str | None = None,
) -> AgentLoopResponse:
    start_time = time.perf_counter()
    client = create_client(timeout=30.0)
    steps: list[AgentLoopStepItem] = []
    called_tool_keys: set[str] = set()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    try:
        for step_index in range(1, max_steps + 1):
            # 每一轮都把用户目标、工具白名单、历史步骤和观察结果交给模型。
            # 模型只负责决定下一步 action；真正执行工具仍然由后端安全门控制。
            response = call_chat_completion(
                client,
                _build_agent_loop_messages(
                    message=message,
                    steps=steps,
                    max_steps=max_steps,
                ),
                model=settings.dashscope_model,
                temperature=0.0,
                max_tokens=500,
            )
            usage = response.usage
            prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            total_tokens += getattr(usage, "total_tokens", 0) or 0

            decision = _parse_agent_loop_decision(response.choices[0].message.content or "")

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
                    model=settings.dashscope_model,
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
                    model=settings.dashscope_model,
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
            observation = execute_registered_tool(db, tool_decision)
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

        # 达到最大循环次数时强制停止，避免 Agent 无限制调用模型和工具。
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        answer = "已达到最大 Agent 循环步数，系统已停止继续执行。请根据已有步骤结果人工确认下一步。"
        _safe_create_agent_loop_log(
            db,
            trace_id=trace_id,
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
            model=settings.dashscope_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_ms=cost_ms,
        )
    except (BusinessException, ModelCallException) as exc:
        _safe_create_agent_loop_log(
            db,
            trace_id=trace_id,
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
            cost_ms=round((time.perf_counter() - start_time) * 1000),
            status="error",
            error_type=type(exc).__name__,
            error_message=f"Agent Loop 执行失败：{type(exc).__name__}",
        )
        raise ModelCallException(message=f"Agent Loop 执行失败：{type(exc).__name__}") from exc
