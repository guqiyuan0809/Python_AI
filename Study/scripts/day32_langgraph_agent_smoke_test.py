"""Day32：验证 LangGraph 候选 Agent 的白盒节点、条件循环和节点级重试。

不访问 MySQL、Milvus、DashScope 或真实低风险工具：

* 第一部分让真实 Prompt Chain 使用 fake model，检查记忆/计划/短期历史进入 payload 的位置；
* 第二部分让真实 StateGraph 走“规划模型首次失败 -> LangGraph 重试 -> 决策最终回答”；
* 第三部分走参数完整的高风险路由，确认图仍返回 require_confirm，而非自动执行写操作。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.common.exceptions import ModelCallException
from day04_app.schemas.chat_schema import AgentPlanItem
from day04_app.services.langchain_session_memory_chain import (
    LangChainSessionMemoryExecution,
    LangChainSessionMemoryUsage,
)
import day04_app.services.langgraph_agent_service as graph_service


class DemoDecisionPrompt:
    system_prompt = "你是受控 Agent 决策器，只能输出 JSON。"
    user_prompt_template = (
        "目标：{message}\n最大步骤：{max_steps}\n可用工具：{tools}\n已完成步骤：{steps}"
    )
    model = "demo-qwen"
    temperature = 0.0
    max_tokens = 300


def _execution(answer: str, model: str = "demo-qwen") -> LangChainSessionMemoryExecution:
    return LangChainSessionMemoryExecution(
        answer=answer,
        model=model,
        usage=LangChainSessionMemoryUsage(31, 11, 42),
    )


def _state() -> dict:
    return {
        "message": "请说明当前会话的安全约束，并给出结论。",
        "max_steps": 3,
        "memory_context": {
            "session_summary": "用户正在学习企业 Agent，要求高风险动作必须人工确认。",
            "recent_history": [
                {"role": "user", "content": "之前高风险动作怎么处理？", "turn_no": 5},
                {"role": "assistant", "content": "后端返回 require_confirm，不会自动执行。", "turn_no": 5},
            ],
            "semantic_memories": [
                {
                    "memory_id": "memory-constraint-001",
                    "memory_type": "constraint",
                    "content": "高风险写操作必须人工确认。",
                }
            ],
        },
        "plan": graph_service.AgentExecutionPlan(
            objective="解释安全约束",
            steps=["读取已授权记忆", "给出结论"],
            success_criteria="明确不会自动执行高风险动作",
        ),
        "prompt_steps": [],
    }


def verify_prompt_payloads() -> None:
    planner_messages: list[dict] = []
    decision_messages: list[dict] = []

    def planner_invoker(messages, model, temperature, max_tokens):
        planner_messages.extend(messages)
        return _execution(
            json.dumps(
                {
                    "objective": "解释安全约束",
                    "steps": ["读取受治理记忆", "给出结论"],
                    "success_criteria": "说明人工确认边界",
                },
                ensure_ascii=False,
            ),
            model,
        )

    def decision_invoker(messages, model, temperature, max_tokens):
        decision_messages.extend(messages)
        return _execution(
            json.dumps(
                {
                    "action": "final_answer",
                    "tool_name": None,
                    "arguments": {},
                    "reason": "已有计划和记忆足够回答",
                    "final_answer": "高风险动作必须经过人工确认。",
                },
                ensure_ascii=False,
            ),
            model,
        )

    state = _state()
    planner = graph_service.build_langgraph_planner_chain(model_invoker=planner_invoker)
    assert "解释安全约束" in planner.invoke(state).answer
    # system(规划规则) -> system(摘要) -> system(长期记忆) -> user/assistant(短期历史) -> 当前问题
    assert [item["role"] for item in planner_messages] == ["system", "system", "system", "user", "assistant", "user"]
    assert "高风险动作必须人工确认" in planner_messages[1]["content"]
    assert "memory-constraint-001" not in planner_messages[2]["content"]
    assert "高风险写操作必须人工确认" in planner_messages[2]["content"]

    decision = graph_service.build_langgraph_decision_chain(
        DemoDecisionPrompt(),
        model_invoker=decision_invoker,
    )
    assert "final_answer" in decision.invoke(state).answer
    # 决策 Prompt 比规划多一个“计划”system 槽位；短期历史仍保持真实 user/assistant 角色。
    assert [item["role"] for item in decision_messages] == [
        "system", "system", "system", "system", "user", "assistant", "user"
    ]
    assert "解释安全约束" in decision_messages[1]["content"]
    assert "最大步骤：3" in decision_messages[-1]["content"]


def verify_graph_model_retry() -> None:
    # 替换两个模型 Chain，不触碰真实网络；StateGraph、RetryPolicy、条件边仍为真实实现。
    original_planner = graph_service.build_langgraph_planner_chain
    original_decision = graph_service.build_langgraph_decision_chain
    original_prompt_loader = graph_service.get_active_agent_decision_prompt
    original_identity = graph_service._get_agent_decision_prompt_identity
    original_logger = graph_service._log_graph_event
    planner_attempts = {"count": 0}

    class RetryPlannerChain:
        def invoke(self, state):
            planner_attempts["count"] += 1
            if planner_attempts["count"] == 1:
                raise ModelCallException("模拟模型瞬时超时")
            return _execution(
                json.dumps(
                    {
                        "objective": "解释 LangGraph 重试",
                        "steps": ["生成计划", "生成最终回答"],
                        "success_criteria": "用户看到重试后的回答",
                    },
                    ensure_ascii=False,
                )
            )

    class FinalDecisionChain:
        def invoke(self, state):
            return _execution(
                json.dumps(
                    {
                        "action": "final_answer",
                        "tool_name": None,
                        "arguments": {},
                        "reason": "计划已完成，不需要工具",
                        "final_answer": "LangGraph 已在模型节点失败后重试成功。",
                    },
                    ensure_ascii=False,
                )
            )

    try:
        graph_service.build_langgraph_planner_chain = lambda: RetryPlannerChain()
        graph_service.build_langgraph_decision_chain = lambda prompt: FinalDecisionChain()
        graph_service.get_active_agent_decision_prompt = lambda db: DemoDecisionPrompt()
        graph_service._get_agent_decision_prompt_identity = lambda prompt: SimpleNamespace(
            as_call_log_fields=lambda: {}
        )
        graph_service._log_graph_event = lambda *args, **kwargs: None

        result = graph_service.run_langgraph_agent_loop(
            object(),
            message="解释 LangGraph 模型失败后的重试机制",
            max_steps=3,
            include_semantic_memories=False,
        )
    finally:
        graph_service.build_langgraph_planner_chain = original_planner
        graph_service.build_langgraph_decision_chain = original_decision
        graph_service.get_active_agent_decision_prompt = original_prompt_loader
        graph_service._get_agent_decision_prompt_identity = original_identity
        graph_service._log_graph_event = original_logger

    assert planner_attempts["count"] == 2
    assert result.status == "success"
    assert result.plan.source == "langchain_planner"
    assert result.model_retry_count == 1
    assert len(result.steps) == 1
    assert result.steps[0].action == "final_answer"


def verify_high_risk_route() -> None:
    # 参数完整的高风险请求不应调用规划/决策模型；真实 StructuredTool 仍回项目策略层。
    original_logger = graph_service._log_graph_event
    try:
        graph_service._log_graph_event = lambda *args, **kwargs: None
        result = graph_service.run_langgraph_agent_loop(
            object(),
            # 该确定性路由的语法是“关闭工单 <ID>，关闭原因是 <原因>”。
            message="关闭工单 WO-1001，关闭原因是本次测试流程已完成。",
            max_steps=3,
            include_semantic_memories=False,
        )
    finally:
        graph_service._log_graph_event = original_logger

    assert result.plan.source == "deterministic_security_route"
    assert result.model_retry_count == 0
    assert len(result.steps) == 1
    assert result.steps[0].observation is not None
    assert result.steps[0].observation["status"] == "require_confirm"
    assert "尚未执行" in result.answer


def verify_read_only_tool_retry() -> None:
    """低风险只读工具可节点级重试；高风险工具已由上一个用例证明不会走到此路径。"""

    original_planner = graph_service.build_langgraph_planner_chain
    original_decision = graph_service.build_langgraph_decision_chain
    original_prompt_loader = graph_service.get_active_agent_decision_prompt
    original_identity = graph_service._get_agent_decision_prompt_identity
    original_catalog = graph_service.build_langchain_tools
    original_invoke_tool = graph_service.invoke_governed_langchain_tool
    original_logger = graph_service._log_graph_event
    tool_attempts = {"count": 0}

    class PlannerChain:
        def invoke(self, state):
            return _execution(
                json.dumps(
                    {
                        "objective": "查询会话状态",
                        "steps": ["调用只读会话查询工具", "根据返回状态回答"],
                        "success_criteria": "得到会话状态或明确未找到",
                    },
                    ensure_ascii=False,
                )
            )

    class ToolDecisionChain:
        def invoke(self, state):
            return _execution(
                json.dumps(
                    {
                        "action": "call_tool",
                        "tool_name": "get_session_status",
                        "arguments": {"session_id": "session-retry-demo"},
                        "reason": "需要读取可信会话状态",
                        "final_answer": None,
                    },
                    ensure_ascii=False,
                )
            )

    def retryable_tool(_catalog, *, tool_name, arguments):
        tool_attempts["count"] += 1
        if tool_attempts["count"] == 1:
            raise RuntimeError("模拟下游读取服务瞬时网络异常")
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "data": {"found": False, "message": "未找到该会话"},
        }

    try:
        graph_service.build_langgraph_planner_chain = lambda: PlannerChain()
        graph_service.build_langgraph_decision_chain = lambda prompt: ToolDecisionChain()
        graph_service.get_active_agent_decision_prompt = lambda db: DemoDecisionPrompt()
        graph_service._get_agent_decision_prompt_identity = lambda prompt: SimpleNamespace(
            as_call_log_fields=lambda: {}
        )
        graph_service.build_langchain_tools = lambda db, principal, execution_context=None: object()
        graph_service.invoke_governed_langchain_tool = retryable_tool
        graph_service._log_graph_event = lambda *args, **kwargs: None

        result = graph_service.run_langgraph_agent_loop(
            object(),
            message="查询会话 session-retry-demo 的状态",
            max_steps=3,
            include_semantic_memories=False,
        )
    finally:
        graph_service.build_langgraph_planner_chain = original_planner
        graph_service.build_langgraph_decision_chain = original_decision
        graph_service.get_active_agent_decision_prompt = original_prompt_loader
        graph_service._get_agent_decision_prompt_identity = original_identity
        graph_service.build_langchain_tools = original_catalog
        graph_service.invoke_governed_langchain_tool = original_invoke_tool
        graph_service._log_graph_event = original_logger

    assert tool_attempts["count"] == 2
    assert result.read_only_tool_retry_count == 1
    assert result.status == "success"
    assert result.steps[0].observation is not None
    assert result.steps[0].observation["status"] == "not_found"


def main() -> None:
    verify_prompt_payloads()
    verify_graph_model_retry()
    verify_high_risk_route()
    verify_read_only_tool_retry()
    print("DAY32_LANGGRAPH_AGENT_SMOKE_OK")
    print("graph=load_memory->policy_route->planner->model_decision->tool_guard->tool_execute->working_memory->conditional_loop")
    print("model_retry=1 read_only_tool_retry=1 high_risk=require_confirm loop_control=state.step_count<=max_steps")


if __name__ == "__main__":
    main()
