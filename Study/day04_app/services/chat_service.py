"""
聊天业务服务层

类似 Java 项目里的 ChatService。
"""

import json
from collections.abc import Iterator
from uuid import uuid4

from openai import OpenAI

from day04_app.common.exceptions import ModelCallException
from day04_app.schemas.chat_schema import ChatResponse
from settings import settings


def create_client(timeout: float = 30.0) -> OpenAI:
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        timeout=timeout,
    )


def safe_chat(message: str) -> ChatResponse:
    client = create_client(timeout=30.0)
    try:
        response = client.chat.completions.create(
            model=settings.dashscope_model,
            messages=[
                {"role": "system", "content": "你是一个简洁专业的 Python AI 助手。"},
                {"role": "user", "content": message},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return ChatResponse(
            answer=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
    except Exception as exc:
        raise ModelCallException(message=f"模型调用失败：{type(exc).__name__}") from exc


def build_sse_event(data: dict) -> str:
    json_data = json.dumps(data, ensure_ascii=False)
    return f"data: {json_data}\n\n"


def stream_chat_events(message: str, trace_id: str) -> Iterator[str]:
    client = create_client(timeout=60.0)
    stream_id = uuid4().hex
    full_answer: list[str] = []

    yield build_sse_event(
        {
            "type": "start",
            "trace_id": trace_id,
            "stream_id": stream_id,
            "message": "stream started",
        }
    )

    try:
        stream = client.chat.completions.create(
            model=settings.dashscope_model,
            messages=[
                {"role": "system", "content": "你是一个简洁专业的 Python AI 助手。"},
                {"role": "user", "content": message},
            ],
            temperature=0.3,
            max_tokens=500,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue

            full_answer.append(delta)
            yield build_sse_event(
                {
                    "type": "delta",
                    "trace_id": trace_id,
                    "stream_id": stream_id,
                    "delta": delta,
                }
            )

        answer = "".join(full_answer)
        save_stream_answer(
            stream_id=stream_id,
            trace_id=trace_id,
            question=message,
            answer=answer,
        )
        yield build_sse_event(
            {
                "type": "done",
                "trace_id": trace_id,
                "stream_id": stream_id,
            }
        )
    except Exception as exc:
        yield build_sse_event(
            {
                "type": "error",
                "trace_id": trace_id,
                "stream_id": stream_id,
                "code": 50001,
                "message": f"模型流式调用失败：{type(exc).__name__}",
            }
        )


def save_stream_answer(
    stream_id: str,
    trace_id: str,
    question: str,
    answer: str,
) -> None:
    """
    这里是企业项目中的落库扩展点。

    真实项目里可以保存到 MySQL / Redis / MongoDB，
    用于会话历史、审计日志、上下文恢复和问题排查。
    """
    return None
