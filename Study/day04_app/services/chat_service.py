"""
聊天业务服务层

类似 Java 项目里的 ChatService。
"""

import json
import time
from collections.abc import Iterator
from uuid import uuid4

from openai import OpenAI
from pydantic import ValidationError

from day04_app.common.exceptions import ModelCallException
from day04_app.schemas.chat_schema import ChatResponse, WorkOrderAnalysisResponse, WorkOrderAnalysisResult
from settings import settings


def create_client(timeout: float = 30.0) -> OpenAI:
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        timeout=timeout,
    )


def safe_chat(message: str) -> ChatResponse:
    messages = [
        {"role": "system", "content": "你是一个简洁专业的 Python AI 助手。"},
        {"role": "user", "content": message},
    ]
    return safe_chat_with_messages(messages)


def safe_chat_with_messages(messages: list[dict]) -> ChatResponse:
    # 普通非流式调用：一次性拿到完整模型回答和 token 用量。
    client = create_client(timeout=30.0)
    try:
        response = client.chat.completions.create(
            model=settings.dashscope_model,
            messages=messages,
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


def analyze_work_order_structured(content: str) -> WorkOrderAnalysisResponse:
    messages = [
        {
            "role": "system",
            "content": (
                "你是企业工单分析助手。你必须只输出一个合法 JSON 对象，不能输出 Markdown、解释或多余文本。"
                "category 只能是 consult、complaint、repair、other；"
                "risk_level 只能是 low、medium、high；"
                "suggestions 必须是 1 到 5 条中文处理建议。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请分析下面的工单内容，并严格按这个 JSON 结构返回：\n"
                "{\n"
                '  "category": "consult|complaint|repair|other",\n'
                '  "risk_level": "low|medium|high",\n'
                '  "summary": "200字以内的问题摘要",\n'
                '  "suggestions": ["处理建议1", "处理建议2"],\n'
                '  "need_human_review": true,\n'
                '  "confidence": 0.85\n'
                "}\n\n"
                f"工单内容：{content}"
            ),
        },
    ]

    client = create_client(timeout=30.0)
    try:
        response = call_chat_completion(client, messages)
        raw_text = response.choices[0].message.content or ""
        try:
            # 模型本质仍返回字符串；这里先抽取 JSON 对象，再交给 Pydantic 做强类型校验。
            analysis = parse_work_order_analysis(raw_text)
            repair_count = 0
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
        except (ValidationError, json.JSONDecodeError) as exc:
            # 首次输出不合格时，只允许走一次修复，避免模型格式异常导致接口无限重试。
            repair_response = repair_work_order_analysis_output(client, raw_text, str(exc))
            analysis = parse_work_order_analysis(repair_response.choices[0].message.content or "")
            repair_count = 1
            prompt_tokens = response.usage.prompt_tokens + repair_response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens + repair_response.usage.completion_tokens
            total_tokens = response.usage.total_tokens + repair_response.usage.total_tokens
        return WorkOrderAnalysisResponse(
            analysis=analysis,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            repair_count=repair_count,
        )
    except ValidationError as exc:
        raise ModelCallException(message=f"模型结构化输出字段不合法：{exc.errors()[0]['msg']}") from exc
    except json.JSONDecodeError as exc:
        raise ModelCallException(message="模型结构化输出不是合法 JSON") from exc
    except ModelCallException:
        raise
    except Exception as exc:
        raise ModelCallException(message=f"模型结构化分析失败：{type(exc).__name__}") from exc


def call_chat_completion(client: OpenAI, messages: list[dict]):
    return client.chat.completions.create(
        model=settings.dashscope_model,
        messages=messages,
        temperature=0.1,
        max_tokens=600,
    )


def parse_work_order_analysis(raw_text: str) -> WorkOrderAnalysisResult:
    json_text = extract_json_object(raw_text)
    return WorkOrderAnalysisResult.model_validate_json(json_text)


def repair_work_order_analysis_output(client: OpenAI, raw_text: str, error_message: str):
    repair_messages = [
        {
            "role": "system",
            "content": (
                "你是 JSON 修复器。你只能输出一个合法 JSON 对象，不能输出解释、Markdown 或代码块。"
                "必须严格符合字段：category、risk_level、summary、suggestions、need_human_review、confidence。"
            ),
        },
        {
            "role": "user",
            "content": (
                "下面是上一次模型输出和校验错误，请修复为合法 JSON。\n"
                "枚举限制：category=consult|complaint|repair|other，risk_level=low|medium|high。\n"
                "confidence 必须是 0 到 1 之间的数字，suggestions 必须是 1 到 5 条中文建议。\n\n"
                f"校验错误：{error_message}\n\n"
                f"上一次输出：\n{raw_text}"
            ),
        },
    ]
    return call_chat_completion(client, repair_messages)


def extract_json_object(raw_text: str) -> str:
    stripped_text = raw_text.strip()
    start_index = stripped_text.find("{")
    end_index = stripped_text.rfind("}")
    if start_index < 0 or end_index < start_index:
        raise json.JSONDecodeError("未找到 JSON 对象", stripped_text, 0)
    return stripped_text[start_index : end_index + 1]


def summarize_messages_with_model(history_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个会话摘要助手。请把用户和 AI 的历史对话压缩成简洁摘要，"
                "保留用户目标、关键事实、已达成结论、未解决问题。不要添加历史中不存在的信息。"
            ),
        },
        {
            "role": "user",
            "content": f"请总结以下历史对话，控制在 300 字以内：\n{history_text}",
        },
    ]
    result = safe_chat_with_messages(messages)
    return result.answer


def generate_session_title_with_model(history_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个会话标题生成助手。请根据对话内容生成一个简短中文标题，"
                "控制在 6 到 18 个字，不要加书名号、引号、标点解释或多余前缀。"
            ),
        },
        {
            "role": "user",
            "content": f"请为下面这段对话生成标题：\n{history_text}",
        },
    ]
    result = safe_chat_with_messages(messages)
    return result.answer.strip().strip("《》\"'“”")


def build_sse_event(data: dict) -> str:
    json_data = json.dumps(data, ensure_ascii=False)
    return f"data: {json_data}\n\n"


def stream_chat_events(message: str, trace_id: str) -> Iterator[str]:
    client = create_client(timeout=60.0)
    stream_id = uuid4().hex
    full_answer: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

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
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            # 部分兼容模型会在最后一个流式 chunk 中返回 usage。
            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = choices[0].delta.content
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
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
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


def stream_session_chat_events(
    session_id: str,
    message: str,
    trace_id: str,
    history_limit: int = 6,
) -> Iterator[str]:
    stream_id = uuid4().hex
    full_answer: list[str] = []
    assistant_message_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # StreamingResponse 会在返回后继续迭代生成器，所以这里单独创建数据库连接。
    from day04_app.database import SessionLocal
    from day04_app.services.call_log_service import create_call_log
    from day04_app.services.session_service import (
        add_message,
        build_messages,
        refresh_session_summary,
        should_refresh_summary_for_session,
        update_message,
    )

    db = SessionLocal()
    try:
        # 先基于当前会话历史构造模型上下文，此时 build_messages 会把本次问题追加到 messages 末尾。
        messages = build_messages(
            db=db,
            session_id=session_id,
            current_question=message,
            history_limit=history_limit,
        )

        # 用户消息先落库，保证即使模型后续失败，也能查到用户这次问了什么。
        add_message(
            db,
            session_id,
            "user",
            message,
            trace_id=trace_id,
            stream_id=stream_id,
        )
        assistant_message = add_message(
            db,
            session_id,
            "assistant",
            "AI 回答生成中",
            trace_id=trace_id,
            stream_id=stream_id,
            model=settings.dashscope_model,
            status="pending",
        )
        assistant_message_id = assistant_message.message_id

        yield build_sse_event(
            {
                "type": "start",
                "session_id": session_id,
                "trace_id": trace_id,
                "stream_id": stream_id,
                "message_id": assistant_message_id,
            }
        )

        # 真正开始推流后，把 assistant 消息从 pending 更新为 streaming。
        update_message(
            db,
            assistant_message_id,
            status="streaming",
        )

        # 只统计真正调用模型的耗时，不把前面的数据库准备时间算进去。
        start_time = time.perf_counter()
        client = create_client(timeout=60.0)
        stream = client.chat.completions.create(
            model=settings.dashscope_model,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            # 部分兼容模型会在最后一个流式 chunk 中返回 usage，用于统计 token 消耗。
            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

            # usage chunk 可能没有 choices，所以不能直接 chunk.choices[0]。
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = choices[0].delta.content
            if not delta:
                continue

            # 边返回给前端，边在服务端拼接完整回答，方便流结束后一次性落库。
            full_answer.append(delta)
            yield build_sse_event(
                {
                    "type": "delta",
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "stream_id": stream_id,
                    "message_id": assistant_message_id,
                    "delta": delta,
                }
            )

        answer = "".join(full_answer)
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        update_message(
            db,
            assistant_message_id,
            content=answer,
            status="success",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        create_call_log(
            db,
            call_type="session_stream_chat",
            trace_id=trace_id,
            session_id=session_id,
            message_id=assistant_message_id,
            model=settings.dashscope_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_ms=cost_ms,
            status="success",
        )

        # 流式回答落库后，再按原有规则判断是否需要刷新会话摘要。
        if should_refresh_summary_for_session(db, session_id):
            refresh_session_summary(db, session_id)

        yield build_sse_event(
            {
                "type": "done",
                "session_id": session_id,
                "trace_id": trace_id,
                "stream_id": stream_id,
                "message_id": assistant_message_id,
            }
        )
    except Exception as exc:
        error_message = f"模型流式调用失败：{type(exc).__name__}"
        cost_ms = None
        if "start_time" in locals():
            cost_ms = round((time.perf_counter() - start_time) * 1000)

        # 中途失败时优先保存已经输出的半截内容；没有半截内容才保存错误提示。
        partial_answer = "".join(full_answer)
        failed_content = partial_answer if partial_answer else error_message

        # 如果 assistant 占位消息已经创建，就把它标记为 error，方便历史记录和排查。
        if assistant_message_id:
            update_message(
                db,
                assistant_message_id,
                content=failed_content,
                status="error",
                error_message=error_message,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            create_call_log(
                db,
                call_type="session_stream_chat",
                trace_id=trace_id,
                session_id=session_id,
                message_id=assistant_message_id,
                model=settings.dashscope_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_ms=cost_ms,
                status="error",
                error_message=error_message,
            )

        yield build_sse_event(
            {
                "type": "error",
                "session_id": session_id,
                "trace_id": trace_id,
                "stream_id": stream_id,
                "message_id": assistant_message_id,
                "code": 50001,
                "message": error_message,
                "partial_answer": partial_answer,
            }
        )
    finally:
        db.close()


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
