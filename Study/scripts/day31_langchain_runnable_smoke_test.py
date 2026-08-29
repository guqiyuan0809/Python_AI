"""Day31：验证 LangChain PromptTemplate | Runnable 只负责决策链，不执行工具。"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.services.langchain_agent_decision_chain import (
    LangChainAgentDecisionExecution,
    LangChainModelUsage,
    build_langchain_agent_decision_chain,
)


class DemoPrompt:
    system_prompt = "你是受控 Agent 决策器，只能输出 JSON。"
    user_prompt_template = (
        "目标：{message}\n最大步骤：{max_steps}\n工具：{tools}\n已完成步骤：{steps}"
    )
    model = "demo-qwen"
    temperature = 0.0
    max_tokens = 500


def main() -> None:
    captured: dict[str, object] = {}

    def fake_model_invoker(messages, model, temperature, max_tokens):
        captured["messages"] = messages
        captured["model"] = model
        captured["temperature"] = temperature
        captured["max_tokens"] = max_tokens
        return LangChainAgentDecisionExecution(
            raw_decision_text=json.dumps(
                {
                    "action": "call_tool",
                    "tool_name": "get_session_status",
                    "arguments": {"session_id": "session-demo"},
                    "reason": "需要读取会话状态",
                    "final_answer": None,
                },
                ensure_ascii=False,
            ),
            model=model,
            usage=LangChainModelUsage(11, 7, 18),
        )

    chain = build_langchain_agent_decision_chain(
        DemoPrompt(),
        model_invoker=fake_model_invoker,
    )
    execution = chain.invoke(
        {
            "message": "查询会话 session-demo 的状态",
            "max_steps": 3,
            "steps": [],
        }
    )

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert "查询会话 session-demo 的状态" in messages[1]["content"]
    assert "get_session_status" in messages[1]["content"]
    assert execution.usage.total_tokens == 18
    assert json.loads(execution.raw_decision_text)["tool_name"] == "get_session_status"
    print("DAY31_LANGCHAIN_RUNNABLE_SMOKE_OK")
    print("chain=input_adapter|ChatPromptTemplate|project_qwen_adapter tool_executed=false")


if __name__ == "__main__":
    main()
