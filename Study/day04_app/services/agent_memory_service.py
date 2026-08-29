"""Day32：会话记忆与 Agent 工作记忆的项目治理层。

这层故意不使用 LangChain 的内存对象保存事实：

* MySQL 是完整消息、摘要版本、Agent 快照和长期记忆的事实源；
* Agent 工作记忆是一次 run 内的有序状态，压缩后仍只能服务本次 run；
* 只有经过筛选的稳定事实才进入 ``SessionMemory``，之后才允许异步向量化到
  独立的 Milvus collection；
* LangChain 未来只消费这里已经完成权限、去重和 Token 预算的上下文。
"""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Sequence
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from day04_app.models import AgentWorkingMemorySnapshot, ChatMessage, ChatSession, SessionMemory
from day04_app.services.knowledge_embedding_service import generate_text_embeddings
from day04_app.services.milvus_vector_store_service import search_session_memory_vectors
from day04_app.services.session_service import (
    estimate_text_tokens,
    get_context_messages_after_summary,
    get_latest_session_summary,
    get_session,
)
from settings import settings


@dataclass(frozen=True)
class WorkingMemoryCompaction:
    """当前 Agent Loop 的窗口化结果，不改变原始 steps 审计记录。"""

    summary: dict[str, Any]
    recent_steps: list[dict[str, Any]]
    covered_step_from: int | None
    covered_step_to: int | None
    estimated_tokens: int
    should_compact: bool


def estimate_steps_tokens(steps: Sequence[Any]) -> int:
    """用与会话预算一致的粗略估算，生产可替换为模型 tokenizer。"""

    payload = [
        step.model_dump(mode="json") if hasattr(step, "model_dump") else step
        for step in steps
    ]
    return estimate_text_tokens(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) if payload else 0


def _step_payload(step: Any) -> dict[str, Any]:
    if hasattr(step, "model_dump"):
        return step.model_dump(mode="json")
    return dict(step)


def compact_agent_steps(
    steps: Sequence[Any],
    *,
    force: bool = False,
    trigger_steps: int | None = None,
    keep_recent_steps: int | None = None,
    trigger_tokens: int | None = None,
) -> WorkingMemoryCompaction:
    """把已完成的旧 steps 压成结构化事实，并保留最近若干原始 steps。

    注意这里不调用 LLM：工具 observation 是安全决策事实，第一版先采用确定性提炼，
    防止摘要模型编造一个不存在的 task_id、工具结果或授权状态。
    """

    payloads = [_step_payload(step) for step in steps]
    trigger_steps = trigger_steps or settings.agent_working_memory_trigger_steps
    keep_recent_steps = keep_recent_steps or settings.agent_working_memory_keep_recent_steps
    trigger_tokens = trigger_tokens or settings.agent_working_memory_trigger_tokens
    estimated_tokens = estimate_steps_tokens(payloads)
    should_compact = bool(
        force
        or len(payloads) >= trigger_steps
        or estimated_tokens >= trigger_tokens
    )
    if not should_compact or len(payloads) <= keep_recent_steps:
        return WorkingMemoryCompaction(
            summary={"covered_steps": [], "completed_actions": [], "confirmed_facts": {}, "terminal_constraints": []},
            recent_steps=payloads,
            covered_step_from=None,
            covered_step_to=None,
            estimated_tokens=estimated_tokens,
            should_compact=False,
        )

    old_steps = payloads[:-keep_recent_steps]
    recent_steps = payloads[-keep_recent_steps:]
    completed_actions: list[str] = []
    confirmed_facts: dict[str, Any] = {}
    terminal_constraints: list[dict[str, Any]] = []
    failed_or_blocked_actions: list[dict[str, Any]] = []
    for step in old_steps:
        tool_name = step.get("tool_name")
        if tool_name:
            completed_actions.append(str(tool_name))
        observation = step.get("observation") or {}
        if not isinstance(observation, dict):
            continue
        status = observation.get("status")
        if status in {"require_confirm", "blocked", "error", "stopped_by_guardrail"}:
            terminal_constraints.append(
                {
                    "step_index": step.get("step_index"),
                    "tool_name": tool_name,
                    "status": status,
                    "message": observation.get("message"),
                }
            )
        if status in {"blocked", "error"}:
            failed_or_blocked_actions.append(
                {"tool_name": tool_name, "status": status}
            )
        data = observation.get("data")
        # 只提炼结构化、非长文本字段；完整 observation 仍留在原始 steps/日志。
        if isinstance(data, dict):
            for key in ("task_id", "session_id", "status", "found", "trace_id"):
                if key in data and isinstance(data[key], (str, int, float, bool)):
                    confirmed_facts[key] = data[key]

    indexes = [int(step.get("step_index", 0)) for step in old_steps if step.get("step_index") is not None]
    summary = {
        "covered_steps": [min(indexes), max(indexes)] if indexes else [],
        "completed_actions": list(dict.fromkeys(completed_actions)),
        "confirmed_facts": confirmed_facts,
        "failed_or_blocked_actions": failed_or_blocked_actions,
        "terminal_constraints": terminal_constraints,
    }
    return WorkingMemoryCompaction(
        summary=summary,
        recent_steps=recent_steps,
        covered_step_from=min(indexes) if indexes else None,
        covered_step_to=max(indexes) if indexes else None,
        estimated_tokens=estimate_steps_tokens([summary, *recent_steps]),
        should_compact=True,
    )


def create_working_memory_snapshot(
    db: Session,
    *,
    run_id: str,
    session_id: str | None,
    trace_id: str | None,
    compaction: WorkingMemoryCompaction,
) -> AgentWorkingMemorySnapshot | None:
    """保存一次压缩检查点；原始 steps 不被删除。"""

    if not compaction.should_compact or compaction.covered_step_from is None:
        return None
    latest_version = db.scalar(
        select(func.max(AgentWorkingMemorySnapshot.snapshot_version)).where(
            AgentWorkingMemorySnapshot.run_id == run_id
        )
    ) or 0
    record = AgentWorkingMemorySnapshot(
        snapshot_id=uuid4().hex,
        run_id=run_id,
        session_id=session_id,
        trace_id=trace_id,
        covered_step_from=compaction.covered_step_from,
        covered_step_to=compaction.covered_step_to or compaction.covered_step_from,
        retained_step_from=(compaction.recent_steps[0].get("step_index") if compaction.recent_steps else None),
        state_json=json.dumps(compaction.summary, ensure_ascii=False, sort_keys=True),
        summary_text=json.dumps(compaction.summary, ensure_ascii=False),
        estimated_tokens=compaction.estimated_tokens,
        snapshot_version=int(latest_version) + 1,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def create_active_summary_memory(
    db: Session,
    *,
    session_id: str,
    summary_id: str,
    summary: str,
    source_message_ids: list[str],
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> SessionMemory:
    """创建/更新一条“当前摘要”长期记忆事实。

    历史摘要版本继续保存在 MySQL，但 Milvus 只应保留最新 active 摘要向量，避免 v1/v2/v3
    重复召回。实际向量写入由异步任务完成，状态先为 pending。
    """

    record = db.scalar(
        select(SessionMemory).where(
            SessionMemory.session_id == session_id,
            SessionMemory.memory_type == "session_summary",
            SessionMemory.status == "active",
        )
    )
    if record is None:
        record = SessionMemory(
            memory_id=uuid4().hex,
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            memory_type="session_summary",
            content=summary,
            source_summary_id=summary_id,
            source_message_ids_json=json.dumps(source_message_ids, ensure_ascii=False),
            embedding_status="pending",
            status="active",
        )
        db.add(record)
    else:
        record.content = summary
        record.source_summary_id = summary_id
        record.source_message_ids_json = json.dumps(source_message_ids, ensure_ascii=False)
        record.embedding_status = "pending"
    db.commit()
    db.refresh(record)
    return record


def create_session_fact_memory(
    db: Session,
    *,
    session_id: str,
    content: str,
    memory_type: str,
    source_message_ids: list[str],
    user_id: str | None,
    tenant_id: str | None = None,
    source_run_id: str | None = None,
) -> SessionMemory:
    """写入一条明确治理过的会话事实，供异步 Embedding 与后续语义检索使用。

    这不是“把整段聊天丢进向量库”的快捷入口。调用方必须先完成权限、脱敏、事实提炼
    和生命周期判定，再显式选择 ``preference``、``constraint`` 或 ``business_fact``。
    当前 Day32 仅提供服务层能力，不开放让客户端直接写记忆的 HTTP 接口。
    """

    allowed_types = {"preference", "constraint", "business_fact"}
    normalized_content = content.strip()
    if memory_type not in allowed_types:
        raise ValueError(f"不支持的长期记忆类型：{memory_type}")
    if not normalized_content:
        raise ValueError("长期记忆内容不能为空")
    record = SessionMemory(
        memory_id=uuid4().hex,
        session_id=session_id,
        user_id=user_id,
        tenant_id=tenant_id,
        memory_type=memory_type,
        content=normalized_content,
        source_message_ids_json=json.dumps(source_message_ids, ensure_ascii=False),
        source_run_id=source_run_id,
        status="active",
        embedding_status="pending",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_active_memory_for_embedding(db: Session, memory_id: str) -> SessionMemory:
    """读取待向量化记忆；Worker 只处理 active/pending 记录。"""

    record = db.scalar(
        select(SessionMemory).where(
            SessionMemory.memory_id == memory_id,
            SessionMemory.status == "active",
        )
    )
    if record is None:
        raise ValueError("长期记忆不存在或已失效")
    return record


def _get_hit_memory_id(hit: dict[str, Any]) -> str | None:
    """兼容 PyMilvus 返回扁平字段或 ``entity`` 字段两种形式。"""

    entity = hit.get("entity")
    value = entity.get("memory_id") if isinstance(entity, dict) else hit.get("memory_id")
    return str(value) if value else None


def load_governed_semantic_memories(
    db: Session,
    *,
    session_id: str,
    question: str,
    actor_id: str | None,
    tenant_id: str | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
) -> list[SessionMemory]:
    """按会话归属、Milvus 向量召回、MySQL 复核和 Token 预算取得长期记忆。

    Milvus 只返回 ``memory_id`` 候选，绝不直接把向量库 metadata 当作 Prompt 正文。
    从 MySQL 回填时再次确认 active、会话范围、所属用户、过期时间，避免索引延迟或
    未来跨会话扩展时发生越权读取。
    """

    if not question.strip():
        return []
    chat_session = get_session(db, session_id)
    if actor_id is not None and chat_session.user_id not in {None, actor_id}:
        # 正常 HTTP 层已做过授权；服务层仍兜底，防止未来内部调用绕过控制器。
        return []

    # 没有已经成功建索引的长期记忆时，不要为了“查空结果”白白调用 Embedding。
    indexed_exists = db.scalar(
        select(SessionMemory.id).where(
            SessionMemory.session_id == session_id,
            SessionMemory.status == "active",
            SessionMemory.embedding_status == "indexed",
        ).limit(1)
    )
    if indexed_exists is None:
        return []

    _, vectors = generate_text_embeddings([question])
    hits = search_session_memory_vectors(
        question_vector=vectors[0],
        session_id=session_id,
        user_id=chat_session.user_id,
        tenant_id=tenant_id,
        top_k=top_k or settings.session_memory_retrieval_top_k,
    )
    candidate_ids = [memory_id for hit in hits if (memory_id := _get_hit_memory_id(hit))]
    if not candidate_ids:
        return []

    records = list(
        db.scalars(
            select(SessionMemory).where(
                SessionMemory.memory_id.in_(candidate_ids),
                SessionMemory.session_id == session_id,
                SessionMemory.status == "active",
                SessionMemory.embedding_status == "indexed",
            )
        ).all()
    )
    memory_by_id = {record.memory_id: record for record in records}
    token_budget = max_tokens or settings.session_memory_context_max_tokens
    selected: list[SessionMemory] = []
    used_tokens = 0
    for memory_id in candidate_ids:
        record = memory_by_id.get(memory_id)
        if record is None:
            continue
        # MySQL 是最后一跳授权事实源，Milvus 只是可丢失、可延迟的索引。
        if chat_session.user_id is not None and record.user_id != chat_session.user_id:
            continue
        if tenant_id is not None and record.tenant_id != tenant_id:
            continue
        if record.expires_at is not None and record.expires_at <= datetime.now():
            continue
        record_tokens = estimate_text_tokens(record.content)
        if selected and used_tokens + record_tokens > token_budget:
            continue
        selected.append(record)
        used_tokens += record_tokens
    return selected


def build_governed_memory_context(
    db: Session,
    *,
    session_id: str,
    current_question: str,
    actor_id: str | None,
    tenant_id: str | None = None,
    history_limit: int = 6,
    include_semantic_memories: bool = True,
    suppress_retrieval_errors: bool = True,
) -> dict[str, Any]:
    """项目 Memory Service 的统一输出，LangChain 只能消费该输出而不能自行持久化。"""

    session = get_session(db, session_id)
    latest_summary = get_latest_session_summary(db, session_id)
    semantic_memories: list[SessionMemory] = []
    if include_semantic_memories:
        try:
            semantic_memories = load_governed_semantic_memories(
                db,
                session_id=session_id,
                question=current_question,
                actor_id=actor_id,
                tenant_id=tenant_id,
            )
        except Exception:
            # 长期记忆是增强项，Embedding/Milvus 故障不能让普通会话回答不可用。
            # 真实异常应由调用层写监控日志；这里的上下文退回 MySQL 摘要 + 最近原文。
            if not suppress_retrieval_errors:
                raise
            semantic_memories = []
    # 最新会话摘要已无条件注入，不能再把同源 ``session_summary`` 从 Milvus 回填一次。
    # 语义检索的真正价值是后续 preference / constraint / business_fact 等按问题取用的
    # 长期事实；当前 first release 只有摘要记忆时，检索结果会在这里被主动去重。
    if latest_summary is not None:
        semantic_memories = [
            memory
            for memory in semantic_memories
            if memory.source_summary_id != latest_summary.summary_id
        ]
    return build_memory_context(
        summary=(latest_summary.summary if latest_summary else session.summary),
        recent_messages=get_context_messages_after_summary(
            db,
            session_id,
            latest_summary,
            limit=history_limit,
        ),
        semantic_memories=semantic_memories,
    )


def build_memory_context(
    *,
    summary: str | None,
    recent_messages: Sequence[ChatMessage],
    semantic_memories: Sequence[SessionMemory] = (),
) -> dict[str, Any]:
    """给 LangChain Runnable 的受治理输入；不负责查询权限或数据库。"""

    return {
        "session_summary": summary or "",
        "recent_history": [
            {"role": message.role, "content": message.content, "turn_no": message.turn_no}
            for message in recent_messages
        ],
        "semantic_memories": [
            {
                "memory_id": memory.memory_id,
                "memory_type": memory.memory_type,
                "source_summary_id": memory.source_summary_id,
                "content": memory.content,
            }
            for memory in semantic_memories
            if memory.status == "active"
        ],
    }
