"""
Day07 预习代码：会话 ID 与多轮历史消息

这个文件先用内存字典模拟数据库，帮助理解：
1. session_id 如何标识一段会话
2. user / assistant 消息如何保存
3. 如何把历史消息重新组装成 messages

明天会在这个基础上继续升级成 FastAPI 接口和数据库设计。
"""

from datetime import datetime
from uuid import uuid4


SYSTEM_PROMPT = "你是一个专业、简洁的 Python AI 应用开发老师。"

session_store: dict[str, list[dict]] = {}


def create_session() -> str:
    session_id = uuid4().hex
    session_store[session_id] = []
    return session_id


def add_message(session_id: str, role: str, content: str) -> None:
    if session_id not in session_store:
        raise ValueError("会话不存在")

    session_store[session_id].append(
        {
            "message_id": uuid4().hex,
            "role": role,
            "content": content,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )


def get_recent_messages(session_id: str, limit: int = 6) -> list[dict]:
    if session_id not in session_store:
        raise ValueError("会话不存在")

    return session_store[session_id][-limit:]


def build_messages(session_id: str, current_question: str, history_limit: int = 6) -> list[dict]:
    history = get_recent_messages(session_id, limit=history_limit)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in history:
        messages.append(
            {
                "role": item["role"],
                "content": item["content"],
            }
        )

    messages.append({"role": "user", "content": current_question})
    return messages


def demo() -> None:
    session_id = create_session()
    print(f"创建会话: {session_id}")

    add_message(session_id, "user", "什么是 SSE？")
    add_message(session_id, "assistant", "SSE 是 Server-Sent Events，用于服务端持续推送事件。")
    add_message(session_id, "user", "它适合 AI 流式输出吗？")
    add_message(session_id, "assistant", "适合，因为模型生成一段内容后可以立刻推送给前端。")

    messages = build_messages(session_id, "它和 WebSocket 有什么区别？")
    print("\n本次发给模型的 messages:")
    for message in messages:
        print(message)


if __name__ == "__main__":
    demo()
