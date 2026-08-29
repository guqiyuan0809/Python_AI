"""Day32：验证 LangChain 只负责将项目治理的记忆 payload 组装成模型 messages。

不访问 MySQL、Milvus、DashScope；Memory Service 的数据库检索行为已由
``day32_memory_smoke_test.py`` 覆盖。这里聚焦观察短期、摘要、长期事实进入 Prompt 的位置。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.services.langchain_session_memory_chain import (
    LANGCHAIN_SESSION_MEMORY_CHAIN_NAME,
    LangChainSessionMemoryExecution,
    LangChainSessionMemoryUsage,
    build_langchain_session_memory_chain,
    get_langchain_session_memory_prompt_identity,
)


def main() -> None:
    captured: dict[str, object] = {}

    def fake_model_invoker(messages, model, temperature, max_tokens):
        captured["messages"] = messages
        captured["model"] = model
        captured["temperature"] = temperature
        captured["max_tokens"] = max_tokens
        return LangChainSessionMemoryExecution(
            answer="会话摘要、最近对话和已授权长期偏好已被用于回答。",
            model=model,
            usage=LangChainSessionMemoryUsage(101, 22, 123),
        )

    # 模拟项目 Memory Service 的输出；调用方不可以把该数据换成客户端 HTTP Body。
    governed_memory_context = {
        "session_summary": "用户正在学习企业级 AI 应用开发，强调 Agent 必须保留 max_steps 硬上限。",
        "recent_history": [
            {"role": "user", "content": "那工作记忆摘要是否会放开最大步数？", "turn_no": 11},
            {"role": "assistant", "content": "不会，摘要只压缩上下文，不改变执行上限。", "turn_no": 11},
        ],
        "semantic_memories": [
            {
                "memory_id": "memory-preference-001",
                "memory_type": "preference",
                "content": "用户偏好中文、按代码路径理解实现。",
            },
            {
                "memory_id": "memory-constraint-002",
                "memory_type": "constraint",
                "content": "不要将完整会话原文直接保存到 Milvus。",
            },
        ],
    }
    chain = build_langchain_session_memory_chain(
        model_invoker=fake_model_invoker,
        model="demo-qwen",
    )
    execution = chain.invoke(
        {
            "current_question": "请总结本次记忆是怎样进入模型 payload 的？",
            "memory_context": governed_memory_context,
        }
    )

    messages = captured["messages"]
    assert isinstance(messages, list)
    # 0=系统角色，1=会话摘要，2=经治理长期事实，3/4=MessagesPlaceholder 注入的短期历史，5=当前问题。
    assert [item["role"] for item in messages] == ["system", "system", "system", "user", "assistant", "user"]
    assert "max_steps 硬上限" in messages[1]["content"]
    assert "memory-preference-001" not in messages[2]["content"]
    assert "用户偏好中文" in messages[2]["content"]
    assert "工作记忆摘要" in messages[3]["content"]
    assert "不会，摘要只压缩" in messages[4]["content"]
    assert "怎样进入模型 payload" in messages[5]["content"]
    assert execution.usage.total_tokens == 123
    identity = get_langchain_session_memory_prompt_identity()
    assert identity["prompt_name"] == "langchain_session_memory"
    assert len(identity["prompt_template_hash"]) == 64

    print("DAY32_LANGCHAIN_MEMORY_CHAIN_SMOKE_OK")
    print(
        "chain=governed_memory_payload|ChatPromptTemplate|MessagesPlaceholder(recent_history)|project_qwen_adapter"
    )
    print("message_order=system_role+session_summary+semantic_memories+recent_history+current_question")
    print(f"chain_name={LANGCHAIN_SESSION_MEMORY_CHAIN_NAME} model_called=demo-qwen")


if __name__ == "__main__":
    main()
