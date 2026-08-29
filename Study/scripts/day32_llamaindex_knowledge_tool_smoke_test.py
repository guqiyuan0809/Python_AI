"""Day32：验证 LlamaIndex RAG 已作为受治理 LangChain Tool 接入 Agent 工具目录。

不访问 MySQL、Milvus 或 DashScope：通过替换 LlamaIndex 工具内部入口，验证真正重要的
集成边界：模型只传 question；可信 Principal data_scope 决定知识域；缺少 knowledge:read
时在项目策略层拦截；LangGraph 的 tool_execute 可以把 Tool Observation 回写 State 后继续决策。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.common.exceptions import BusinessException
from day04_app.security.permissions import PERMISSION_KNOWLEDGE_READ, PERMISSION_TOOL_EXECUTE
from day04_app.security.principal import SecurityPrincipal
from day04_app.services.langchain_tool_adapter_service import build_langchain_tools, get_langchain_tool
from day04_app.services.tool_execution_context import ToolExecutionContext
import day04_app.services.knowledge_search_tool_service as knowledge_tool_service
import day04_app.services.langgraph_agent_service as graph_service
from day04_app.services.langchain_session_memory_chain import (
    LangChainSessionMemoryExecution,
    LangChainSessionMemoryUsage,
)


def _principal(*permissions: str) -> SecurityPrincipal:
    return SecurityPrincipal(
        actor_id="knowledge-tool-smoke",
        api_key_id="test-key",
        roles=("operator",),
        permissions=frozenset(permissions),
        auth_type="test",
        data_scope={
            "tenant_id": "park-001",
            "knowledge_domains": [
                {
                    "domain_id": "safety_compliance",
                    "description": "园区安全法规与隐患整改标准",
                    "document_ids": ["doc-safety-001", "doc-safety-002"],
                    "keywords": ["隐患", "整改", "安全"],
                }
            ],
            "knowledge_default_domain_id": "safety_compliance",
        },
    )


def _fake_rag_result():
    reference = SimpleNamespace(
        source_id="S1",
        document_id="doc-safety-001",
        version_id="version-active-001",
        chunk_id="chunk-001",
        chunk_index=2,
        score=0.91,
        locations=["Paragraph:12"],
    )
    answer_result = SimpleNamespace(
        answer="应先设置隔离区域并完成整改复核。[S1]",
        references=[reference],
        model="demo-qwen",
        prompt_tokens=111,
        completion_tokens=22,
        total_tokens=133,
        prompt_identity=None,
        retrieved_node_count=5,
        included_node_count=2,
        omitted_node_count=1,
        top_score=0.91,
        score_threshold=0.25,
        rejected_by_score_threshold=False,
    )
    route = SimpleNamespace(
        selected_domain_id="safety_compliance",
        selected_document_ids=["doc-safety-001", "doc-safety-002"],
        route_reason="deterministic_domain_keyword_match",
    )
    return SimpleNamespace(
        answer_result=answer_result,
        route=route,
        active_version_by_document_id={
            "doc-safety-001": "version-active-001",
            "doc-safety-002": "version-active-002",
        },
    )


def verify_structured_tool_boundary() -> None:
    original_answer = knowledge_tool_service.answer_multi_document_with_llamaindex
    original_log = knowledge_tool_service.create_call_log
    captured: dict = {}

    def fake_answer(_db, **kwargs):
        captured.update(kwargs)
        return _fake_rag_result()

    try:
        knowledge_tool_service.answer_multi_document_with_llamaindex = fake_answer
        knowledge_tool_service.create_call_log = lambda *args, **kwargs: None
        catalog = build_langchain_tools(
            object(),  # type: ignore[arg-type]
            principal=_principal(PERMISSION_TOOL_EXECUTE, PERMISSION_KNOWLEDGE_READ),
            execution_context=ToolExecutionContext(
                trace_id="trace-knowledge-tool",
                session_id="session-knowledge-tool",
            ),
        )
        tool = get_langchain_tool(catalog, "knowledge_search")
        payload = json.loads(tool.invoke({"question": "发现消防通道堵塞后如何整改？"}))
    finally:
        knowledge_tool_service.answer_multi_document_with_llamaindex = original_answer
        knowledge_tool_service.create_call_log = original_log

    assert payload["tool_name"] == "knowledge_search"
    assert payload["data"]["framework"] == "llamaindex"
    assert payload["data"]["references"][0]["source_id"] == "S1"
    assert captured["question"] == "发现消防通道堵塞后如何整改？"
    assert captured["trace_id"] == "trace-knowledge-tool"
    assert captured["session_id"] == "session-knowledge-tool"
    assert captured["domains"][0].document_ids == ("doc-safety-001", "doc-safety-002")

    try:
        tool.invoke(
            {
                "question": "测试",
                "document_ids": ["forged-document"],
            }
        )
    except Exception:
        pass
    else:
        raise AssertionError("knowledge_search 参数 Schema 不应允许模型伪造 document_ids")


def verify_permission_block() -> None:
    catalog = build_langchain_tools(
        object(),  # type: ignore[arg-type]
        principal=_principal(PERMISSION_TOOL_EXECUTE),
        execution_context=ToolExecutionContext(trace_id="trace-permission-block"),
    )
    result = json.loads(
        get_langchain_tool(catalog, "knowledge_search").invoke({"question": "查询隐患整改标准"})
    )
    assert result["status"] == "block"
    assert "MISSING_TOOL_SPECIFIC_PERMISSION" in result["matched_rules"]


def _execution(answer: str) -> LangChainSessionMemoryExecution:
    return LangChainSessionMemoryExecution(
        answer=answer,
        model="demo-qwen",
        usage=LangChainSessionMemoryUsage(20, 10, 30),
    )


class DemoDecisionPrompt:
    system_prompt = "受控 Agent 决策器"
    user_prompt_template = "目标：{message}\n工具：{tools}\n步骤：{steps}"
    model = "demo-qwen"
    temperature = 0.0
    max_tokens = 300


def verify_langgraph_tool_loop() -> None:
    """真实 StateGraph 走 knowledge_search Observation，再进入第二轮 final_answer。"""

    original_planner = graph_service.build_langgraph_planner_chain
    original_decision = graph_service.build_langgraph_decision_chain
    original_prompt_loader = graph_service.get_active_agent_decision_prompt
    original_identity = graph_service._get_agent_decision_prompt_identity
    original_catalog = graph_service.build_langchain_tools
    original_invoke_tool = graph_service.invoke_governed_langchain_tool
    original_logger = graph_service._log_graph_event
    calls = {"decision": 0}

    class PlannerChain:
        def invoke(self, _state):
            return _execution(
                json.dumps(
                    {
                        "objective": "根据安全知识库给出整改建议",
                        "steps": ["检索已授权知识库", "依据引用回答"],
                        "success_criteria": "给出带来源的整改建议",
                    },
                    ensure_ascii=False,
                )
            )

    class DecisionChain:
        def invoke(self, state):
            calls["decision"] += 1
            if not state.get("steps"):
                return _execution(
                    json.dumps(
                        {
                            "action": "call_tool",
                            "tool_name": "knowledge_search",
                            "arguments": {"question": state["message"]},
                            "reason": "需要已授权资料作为回答依据",
                            "final_answer": None,
                        },
                        ensure_ascii=False,
                    )
                )
            observation = state["steps"][0].observation or {}
            assert observation["data"]["framework"] == "llamaindex"
            return _execution(
                json.dumps(
                    {
                        "action": "final_answer",
                        "tool_name": None,
                        "arguments": {},
                        "reason": "已有检索证据",
                        "final_answer": "应先设置隔离区域并完成整改复核。[S1]",
                    },
                    ensure_ascii=False,
                )
            )

    def fake_tool(_catalog, *, tool_name, arguments):
        assert tool_name == "knowledge_search"
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "data": {
                "framework": "llamaindex",
                "answer": "应先设置隔离区域并完成整改复核。[S1]",
                "references": [{"source_id": "S1"}],
            },
            "usage": {"model": "demo-qwen", "prompt_tokens": 111, "completion_tokens": 22, "total_tokens": 133},
        }

    try:
        graph_service.build_langgraph_planner_chain = lambda: PlannerChain()
        graph_service.build_langgraph_decision_chain = lambda _prompt: DecisionChain()
        graph_service.get_active_agent_decision_prompt = lambda _db: DemoDecisionPrompt()
        graph_service._get_agent_decision_prompt_identity = lambda _prompt: SimpleNamespace(
            as_call_log_fields=lambda: {}
        )
        graph_service.build_langchain_tools = lambda _db, principal, execution_context=None: object()
        graph_service.invoke_governed_langchain_tool = fake_tool
        graph_service._log_graph_event = lambda *args, **kwargs: None
        result = graph_service.run_langgraph_agent_loop(
            object(),
            message="消防通道堵塞后如何整改？",
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

    assert result.status == "success"
    assert calls["decision"] == 2
    assert len(result.steps) == 2
    assert result.steps[0].tool_name == "knowledge_search"
    assert result.steps[0].observation is not None
    assert result.steps[0].observation["data"]["framework"] == "llamaindex"
    assert result.steps[1].action == "final_answer"
    # 第一轮规划/决策、RAG Tool 内 QueryEngine 和第二轮决策均会进入根成本。
    assert result.total_tokens == 223


def main() -> None:
    verify_structured_tool_boundary()
    verify_permission_block()
    verify_langgraph_tool_loop()
    print("DAY32_LLAMA_INDEX_KNOWLEDGE_TOOL_SMOKE_OK")
    print("tool=knowledge_search scope=trusted_principal_data_scope framework=llamaindex graph_loop=tool_observation_to_final_answer")


if __name__ == "__main__":
    main()
