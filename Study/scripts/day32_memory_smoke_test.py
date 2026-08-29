"""Day32：验证记忆分层和阈值行为，不访问 MySQL、Milvus 或真实模型。"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.schemas.chat_schema import AgentLoopStepItem
from day04_app.services.agent_memory_service import (
    build_memory_context,
    compact_agent_steps,
)


def _step(index: int, *, task_id: str | None = None) -> AgentLoopStepItem:
    data = {"found": True, "status": "success"}
    if task_id:
        data["task_id"] = task_id
        data["session_id"] = "session-memory-demo"
    return AgentLoopStepItem(
        step_index=index,
        action="call_tool",
        tool_name="get_async_task_status" if task_id else "get_session_status",
        arguments={"task_id": task_id} if task_id else {"session_id": "session-memory-demo"},
        reason="读取可信业务状态",
        observation={"status": "success", "data": data},
        final_answer=None,
    )


def main() -> None:
    steps = [_step(index, task_id="task-001" if index == 1 else None) for index in range(1, 7)]
    compaction = compact_agent_steps(
        steps,
        trigger_steps=6,
        keep_recent_steps=2,
        trigger_tokens=99999,
    )
    assert compaction.should_compact is True
    assert compaction.covered_step_from == 1
    assert compaction.covered_step_to == 4
    assert len(compaction.recent_steps) == 2
    assert compaction.summary["confirmed_facts"]["task_id"] == "task-001"
    assert compaction.summary["confirmed_facts"]["session_id"] == "session-memory-demo"

    # 少于阈值时保留全部原始 Agent 步骤；压缩不等于延长 max_steps。
    small = compact_agent_steps(steps[:5], trigger_steps=6, keep_recent_steps=2, trigger_tokens=99999)
    assert small.should_compact is False
    assert len(small.recent_steps) == 5

    # Prompt 上下文由项目服务准备：摘要、最近轮次和已治理语义记忆是三个独立槽位。
    # 这里不访问 MySQL/Milvus，只验证 LangChain 将消费的稳定数据契约。
    memory_context = build_memory_context(
        summary="用户正在学习 Agent Loop，要求保留最大步数上限。",
        recent_messages=[],
        semantic_memories=(),
    )
    assert memory_context["session_summary"].startswith("用户正在学习")
    assert memory_context["recent_history"] == []
    assert memory_context["semantic_memories"] == []

    print("DAY32_MEMORY_SMOKE_OK")
    print("agent_steps=6 compacted=1-4 retained=5-6 facts=task_id+session_id max_steps_control=separate")
    print("memory_context=summary+recent_history+governed_semantic_memories")


if __name__ == "__main__":
    main()
