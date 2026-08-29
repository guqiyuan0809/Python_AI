"""Day31：把项目受控工具适配为 LangChain StructuredTool。

LangChain Tool 只是 Agent/Chain 可理解的标准工具契约，不能替代企业后端的
权限与策略决策。本模块的每次 Tool.invoke 都会回到 ``execute_registered_tool``：

    LangChain StructuredTool
        -> 项目 ToolDecision
        -> ToolPolicyChecker（RBAC / 风险 / 人工确认）
        -> 白名单 executor

因此框架即使被替换、模型即使错误选择高风险工具，也不能绕过已有安全边界。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from day04_app.security.principal import SecurityPrincipal
from day04_app.services.tool_calling_service import (
    TOOL_REGISTRY,
    ToolDecision,
    ToolDefinition,
    execute_registered_tool,
)
from day04_app.services.tool_execution_context import ToolExecutionContext
from day04_app.common.exceptions import BusinessException
from pydantic import ValidationError


@dataclass(frozen=True)
class LangChainToolCatalog:
    """框架工具与项目原始定义的映射，便于后续 Agent 记录与审计。"""

    tools: list[StructuredTool]
    definitions_by_name: dict[str, ToolDefinition]


def _build_governed_description(definition: ToolDefinition) -> str:
    """在模型可见描述中明确工具风险，但安全结论始终由后端执行时决定。"""

    return (
        f"{definition.description}\n\n"
        "【企业执行边界】"
        f"read_only={definition.read_only}；"
        f"risk_level={definition.risk_level}；"
        f"require_human_confirm={definition.require_human_confirm}。"
        "调用请求会先经过后端权限和策略校验；返回 require_confirm 或 block 时，"
        "表示动作尚未执行，必须如实告知用户。"
    )


def _to_tool_observation_text(result: dict[str, Any] | None) -> str:
    """LangChain Tool 的观察结果使用 JSON 字符串，便于后续 LLM/Agent 消费。"""

    return json.dumps(result or {"status": "error", "message": "工具未返回结果"}, ensure_ascii=False)


def build_langchain_tools(
    db: Session,
    *,
    principal: SecurityPrincipal,
    execution_context: ToolExecutionContext | None = None,
) -> LangChainToolCatalog:
    """基于当前白名单构造 LangChain StructuredTool。

    ``principal`` 必须由已认证 HTTP 请求或异步任务身份快照显式提供。此处不兜底为
    SYSTEM_PRINCIPAL，防止未来在线链路因为遗漏身份上下文而获得系统权限。Trace、会话等
    不可由模型伪造的关联信息同样由 ``execution_context`` 闭包传入。
    """

    tools: list[StructuredTool] = []
    definitions_by_name = dict(TOOL_REGISTRY)

    for definition in definitions_by_name.values():
        # default argument 固化当前循环的 definition，避免 Python 闭包最后绑定同一个工具。
        def invoke_governed_tool(
            _definition: ToolDefinition = definition,
            **arguments: Any,
        ) -> str:
            result = execute_registered_tool(
                db,
                ToolDecision(
                    need_tool=True,
                    tool_name=_definition.name,
                    arguments=arguments,
                    reason="由 LangChain StructuredTool 调用，实际执行仍由项目策略层决定。",
                ),
                principal=principal,
                execution_context=execution_context,
            )
            return _to_tool_observation_text(result)

        tools.append(
            StructuredTool.from_function(
                func=invoke_governed_tool,
                name=definition.name,
                description=_build_governed_description(definition),
                args_schema=definition.args_model,
            )
        )

    return LangChainToolCatalog(
        tools=tools,
        definitions_by_name=definitions_by_name,
    )


def get_langchain_tool(catalog: LangChainToolCatalog, tool_name: str) -> StructuredTool:
    """按项目白名单名称取得框架 Tool；未知工具一律报错，不能动态构造。"""

    for tool in catalog.tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"未注册的 LangChain Tool：{tool_name}")


def invoke_governed_langchain_tool(
    catalog: LangChainToolCatalog,
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """从受控目录调用一个 StructuredTool，并恢复项目统一的 observation 原始结构。

    Agent Loop 只有通过这个函数才会使用 LangChain Tool。函数仍不执行权限判断；
    权限和风险判断已经在 Tool 闭包内的 ``execute_registered_tool`` 发生。
    """

    tool = get_langchain_tool(catalog, tool_name)
    try:
        result_text = tool.invoke(arguments)
    except ValidationError as exc:
        raise BusinessException(code=40091, message=f"工具参数不合法：{exc.errors()[0]['msg']}") from exc
    try:
        result = json.loads(str(result_text))
    except (TypeError, json.JSONDecodeError) as exc:
        raise BusinessException(code=50091, message="LangChain 工具返回不是合法 JSON") from exc
    if not isinstance(result, dict):
        raise BusinessException(code=50091, message="LangChain 工具返回结构不合法")
    return result
