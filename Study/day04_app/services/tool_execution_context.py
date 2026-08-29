"""一次受控工具调用的可信运行上下文。

LangChain ``StructuredTool`` 的参数只能来自模型生成的 JSON，因此绝不能把当前用户身份、
Trace ID 或会话 ID 设计成 Tool 的可调用参数。本对象由 Router / LangGraph Runtime 在服务端
构造，再通过闭包传入 Tool executor，专门承载这些不可由模型伪造的关联信息。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """工具执行时的链路关联 ID；身份本身仍通过独立的 ``SecurityPrincipal`` 传递。"""

    trace_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    message_id: str | None = None
