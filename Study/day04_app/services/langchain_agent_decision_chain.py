"""Day31：使用 LangChain PromptTemplate 与 Runnable 表达 Agent 的“决策”步骤。

项目已有的受控 Agent Loop 仍拥有循环次数、重复调用终止、高风险确定性路由、
策略拦截和审计。这里仅将其中可组合的一步标准化：

    输入（目标、步骤） -> PromptTemplate -> 项目 Qwen 调用 -> 原始决策 JSON

不要将该模块误解为 LangChain 默认 AgentExecutor。默认 AgentExecutor 会同时掌握
规划与工具执行；企业项目需要保留自己的状态机和策略层，不能让框架执行绕过护栏。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from day04_app.common.exceptions import ModelCallException
from day04_app.services.chat_service import call_chat_completion, create_client
from day04_app.services.tool_calling_service import list_available_tools
from settings import settings


@dataclass(frozen=True)
class LangChainModelUsage:
    """一次 Runnable 模型阶段的真实用量；循环总量仍由 Agent 主流程汇总。"""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class LangChainAgentDecisionExecution:
    """链的稳定输出；解析和执行刻意留给项目的受控 Agent Loop。"""

    raw_decision_text: str
    model: str
    usage: LangChainModelUsage


ModelInvoker = Callable[[list[dict[str, str]], str, float, int], LangChainAgentDecisionExecution]


def _message_role(message: BaseMessage) -> str:
    """LangChain 消息类型转为项目 OpenAI 兼容客户端的 role。"""

    role_mapping = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
    }
    return role_mapping.get(message.type, "user")


def _to_openai_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    return [
        {
            "role": _message_role(message),
            "content": str(message.content or ""),
        }
        for message in messages
    ]


def invoke_project_chat_model(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> LangChainAgentDecisionExecution:
    """Runnable 的模型适配器：继续使用项目统一的 DashScope/Qwen 配置。"""

    try:
        response = call_chat_completion(
            create_client(timeout=30.0),
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_decision_text = (response.choices[0].message.content or "").strip()
        if not raw_decision_text:
            raise ModelCallException(message="Agent 决策模型返回空内容")
        usage = response.usage
        return LangChainAgentDecisionExecution(
            raw_decision_text=raw_decision_text,
            model=model,
            usage=LangChainModelUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            ),
        )
    except ModelCallException:
        raise
    except Exception as exc:
        raise ModelCallException(
            message=f"LangChain Agent 决策模型调用失败：{type(exc).__name__}"
        ) from exc


def _build_template_variables(payload: dict[str, Any]) -> dict[str, Any]:
    """输入适配 Runnable：将项目对象序列化为 Prompt 可使用的稳定文本。"""

    message = str(payload.get("message") or "").strip()
    if not message:
        raise ValueError("Agent message 不能为空")
    max_steps = int(payload.get("max_steps") or 0)
    if max_steps < 1:
        raise ValueError("Agent max_steps 必须大于等于 1")
    steps = payload.get("steps") or []
    return {
        "message": message,
        "max_steps": max_steps,
        "tools": json.dumps(
            [tool.model_dump() for tool in list_available_tools()],
            ensure_ascii=False,
            indent=2,
        ),
        "steps": json.dumps(steps, ensure_ascii=False, indent=2),
    }


def build_langchain_agent_decision_chain(
    runtime_prompt,
    *,
    model_invoker: ModelInvoker = invoke_project_chat_model,
) -> Runnable[dict[str, Any], LangChainAgentDecisionExecution]:
    """构建可复用的 ``输入 | Prompt | 模型`` Runnable 链。

    runtime_prompt 仍是数据库中已固定的 ai_prompt_version 快照，框架没有自行管理
    Prompt 内容或版本。该链只返回模型原始决策，必须交由原 AgentLoopDecision 的
    Pydantic 校验和后端策略层继续处理。
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", runtime_prompt.system_prompt),
            ("human", runtime_prompt.user_prompt_template),
        ]
    )
    model = runtime_prompt.model or settings.dashscope_model
    temperature = runtime_prompt.temperature if runtime_prompt.temperature is not None else 0.0
    max_tokens = runtime_prompt.max_tokens or 500

    def call_model(prompt_value) -> LangChainAgentDecisionExecution:
        return model_invoker(
            _to_openai_messages(prompt_value.to_messages()),
            model,
            temperature,
            max_tokens,
        )

    return (
        RunnableLambda(_build_template_variables).with_config(run_name="agent_decision_input")
        | prompt_template.with_config(run_name="agent_decision_prompt")
        | RunnableLambda(call_model).with_config(run_name="agent_decision_model")
    )
