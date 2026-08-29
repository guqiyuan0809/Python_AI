"""将受治理的 LlamaIndex 多文档 RAG 暴露为 Agent 可调用的知识检索工具。

这个模块故意不接受模型生成的 ``document_id``、``domain_id``、租户或身份参数：

* 模型只可以提出 ``question``；
* 可信 Java 身份透传形成的 ``SecurityPrincipal.data_scope`` 决定可访问知识域；
* ``ToolExecutionContext`` 由 LangGraph Runtime 注入 Trace / 会话关联信息；
* LlamaIndex 只处理已经授权的知识域，继续复用项目的 active 版本、Milvus、重排和引用治理。

因此 ``knowledge_search`` 不是“让 Agent 自由访问数据库”的工具，而是一个受数据范围约束的
只读 RAG 能力。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException, ModelCallException
from day04_app.security.permissions import PERMISSION_KNOWLEDGE_READ
from day04_app.security.principal import SecurityPrincipal
from day04_app.services.call_log_service import create_call_log
from day04_app.services.llamaindex_multi_document_rag_service import (
    KnowledgeDomain,
    answer_multi_document_with_llamaindex,
)
from day04_app.services.tool_execution_context import ToolExecutionContext
from settings import settings


KNOWLEDGE_DOMAINS_SCOPE_KEY = "knowledge_domains"
KNOWLEDGE_DEFAULT_DOMAIN_SCOPE_KEY = "knowledge_default_domain_id"


class KnowledgeSearchArgs(BaseModel):
    """模型可见的唯一参数；禁止通过 extra 字段伪造知识范围。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=1000, description="需要基于已授权知识库回答的问题")


@dataclass(frozen=True)
class AuthorizedKnowledgeDomains:
    """从可信 Principal 数据范围解析出的知识域，不来自模型或浏览器请求体。"""

    domains: tuple[KnowledgeDomain, ...]
    domain_keywords: dict[str, tuple[str, ...]]
    default_domain_id: str | None


def _normalize_text_sequence(value: Any, *, field_name: str, max_items: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise BusinessException(code=40320, message=f"可信知识数据范围字段 {field_name} 必须是字符串数组")
    items = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    if not items:
        raise BusinessException(code=40320, message=f"可信知识数据范围字段 {field_name} 不能为空")
    if len(items) > max_items:
        raise BusinessException(code=40320, message=f"可信知识数据范围字段 {field_name} 超过数量上限")
    return items


def resolve_authorized_knowledge_domains(principal: SecurityPrincipal) -> AuthorizedKnowledgeDomains:
    """将可信身份中的知识域范围转成 LlamaIndex RouterRetriever 所需契约。

    该函数是模型与数据库之间的数据权限边界。生产中这些 domains 应由 Java/Python 按
    tenant/park/enterprise 查询 ``knowledge_domain_document`` 生成；课程项目暂以已认证
    Java 服务签名透传的 ``data_scope`` 演示同一边界，不能读取客户端 body 中的文档 ID。
    """

    if not principal.has_permissions(PERMISSION_KNOWLEDGE_READ):
        # ToolPolicyChecker 也会检查；这里是 RAG 服务的纵深防御，防止未来被其他入口直接调用。
        raise BusinessException(code=40321, message="当前调用者没有读取企业知识库的权限")

    raw_domains = principal.data_scope.get(KNOWLEDGE_DOMAINS_SCOPE_KEY)
    if not isinstance(raw_domains, list) or not raw_domains:
        raise BusinessException(
            code=40320,
            message="当前身份未配置可访问知识域，拒绝执行知识检索",
        )
    if len(raw_domains) > 8:
        raise BusinessException(code=40320, message="当前身份允许的知识域超过服务上限")

    domains: list[KnowledgeDomain] = []
    keywords_by_domain: dict[str, tuple[str, ...]] = {}
    for raw_domain in raw_domains:
        if not isinstance(raw_domain, Mapping):
            raise BusinessException(code=40320, message="可信知识域配置必须是对象数组")
        domain_id = str(raw_domain.get("domain_id") or "").strip()
        description = str(raw_domain.get("description") or "").strip()
        if not domain_id or len(domain_id) > 64 or not description or len(description) > 300:
            raise BusinessException(code=40320, message="可信知识域缺少合法 domain_id 或 description")
        document_ids = _normalize_text_sequence(
            raw_domain.get("document_ids"),
            field_name=f"{domain_id}.document_ids",
            max_items=30,
        )
        raw_keywords = raw_domain.get("keywords", [])
        if not isinstance(raw_keywords, (list, tuple)):
            raise BusinessException(code=40320, message=f"可信知识域字段 {domain_id}.keywords 必须是字符串数组")
        keywords = tuple(
            dict.fromkeys(str(item).strip() for item in raw_keywords if str(item).strip())
        )
        if len(keywords) > 30:
            raise BusinessException(code=40320, message=f"可信知识域字段 {domain_id}.keywords 超过数量上限")
        domains.append(
            KnowledgeDomain(
                domain_id=domain_id,
                description=description,
                document_ids=document_ids,
            )
        )
        keywords_by_domain[domain_id] = keywords

    if len({domain.domain_id for domain in domains}) != len(domains):
        raise BusinessException(code=40320, message="可信知识域 domain_id 不能重复")

    raw_default_domain_id = principal.data_scope.get(KNOWLEDGE_DEFAULT_DOMAIN_SCOPE_KEY)
    default_domain_id = str(raw_default_domain_id).strip() if raw_default_domain_id is not None else None
    if default_domain_id and default_domain_id not in {domain.domain_id for domain in domains}:
        raise BusinessException(code=40320, message="可信知识域默认领域不属于允许范围")

    return AuthorizedKnowledgeDomains(
        domains=tuple(domains),
        domain_keywords=keywords_by_domain,
        default_domain_id=default_domain_id or domains[0].domain_id,
    )


def _reference_payload(reference: Any) -> dict[str, Any]:
    if hasattr(reference, "model_dump"):
        return reference.model_dump(mode="json")
    return {
        "source_id": str(getattr(reference, "source_id", "")),
        "document_id": str(getattr(reference, "document_id", "")),
        "version_id": str(getattr(reference, "version_id", "")),
        "chunk_id": str(getattr(reference, "chunk_id", "")),
        "chunk_index": int(getattr(reference, "chunk_index", 0)),
        "score": float(getattr(reference, "score", 0.0)),
        "locations": list(getattr(reference, "locations", []) or []),
    }


def execute_knowledge_search_tool(
    db: Session,
    args: BaseModel,
    principal: SecurityPrincipal,
    execution_context: ToolExecutionContext,
) -> dict[str, Any]:
    """执行知识搜索并投影为统一的 Agent Observation 原始结果。

    Agent 的下一轮决策只读取这个返回值；它不会得到数据库会话、全部文档正文或可伪造的
    授权参数。底层 LlamaIndex Retriever/QueryEngine 的分阶段日志和本工具摘要共享同一
    ``trace_id``，可经 ai_call_log 还原整个检索过程。
    """

    # ToolDefinition 与此模块各自持有 Pydantic 参数契约，不能要求二者是同一个 Python
    # 类；统一先投影为普通 dict，再按知识检索自身的 extra=forbid 契约重新校验。
    typed_args = KnowledgeSearchArgs.model_validate(
        args.model_dump() if isinstance(args, BaseModel) else args
    )
    authorized_domains = resolve_authorized_knowledge_domains(principal)
    start_time = time.perf_counter()
    try:
        result = answer_multi_document_with_llamaindex(
            db,
            domains=authorized_domains.domains,
            domain_keywords=authorized_domains.domain_keywords,
            default_domain_id=authorized_domains.default_domain_id,
            question=typed_args.question,
            retrieval_top_k=5,
            max_context_characters=4000,
            # Agent 内知识查询固定由后端选定的质量策略执行，模型不得自行改写这些参数。
            use_reranker=True,
            rerank_top_n=20,
            score_threshold=settings.rag_min_relevance_score,
            trace_id=execution_context.trace_id,
            task_id=execution_context.task_id,
            session_id=execution_context.session_id,
            message_id=execution_context.message_id,
        )
    except (BusinessException, ModelCallException):
        raise
    except Exception as exc:
        raise BusinessException(code=50093, message=f"知识检索工具执行失败：{type(exc).__name__}") from exc

    answer_result = result.answer_result
    payload = {
        "framework": "llamaindex",
        "orchestration": "RouterRetriever + RetrieverQueryEngine",
        "retrieval_backend": "project_milvus",
        "answer": answer_result.answer,
        "references": [_reference_payload(item) for item in answer_result.references],
        "selected_domain_id": result.route.selected_domain_id,
        # 返回实际路由到的允许集合，供下一轮模型解释；不返回其他未授权领域。
        "selected_document_ids": result.route.selected_document_ids,
        "route_reason": result.route.route_reason,
        "active_version_by_document_id": result.active_version_by_document_id,
        "retrieved_node_count": answer_result.retrieved_node_count,
        "included_node_count": answer_result.included_node_count,
        "omitted_node_count": answer_result.omitted_node_count,
        "top_score": answer_result.top_score,
        "score_threshold": answer_result.score_threshold,
        "rejected_by_score_threshold": answer_result.rejected_by_score_threshold,
    }
    usage = {
        "model": answer_result.model,
        "prompt_tokens": answer_result.prompt_tokens or 0,
        "completion_tokens": answer_result.completion_tokens or 0,
        "total_tokens": answer_result.total_tokens or 0,
    }
    create_call_log(
        db,
        call_type="agent_loop",
        stage="llamaindex_knowledge_search",
        trace_id=execution_context.trace_id,
        task_id=execution_context.task_id,
        run_id=execution_context.run_id,
        session_id=execution_context.session_id,
        message_id=execution_context.message_id,
        model=answer_result.model,
        prompt_tokens=answer_result.prompt_tokens,
        completion_tokens=answer_result.completion_tokens,
        total_tokens=answer_result.total_tokens,
        cost_ms=round((time.perf_counter() - start_time) * 1000),
        status="success",
        **(
            answer_result.prompt_identity.as_call_log_fields()
            if answer_result.prompt_identity is not None
            else {}
        ),
        detail={
            "framework": "llamaindex",
            "orchestration": "RouterRetriever + RetrieverQueryEngine",
            "tool_name": "knowledge_search",
            "selected_domain_id": result.route.selected_domain_id,
            "selected_document_count": len(result.route.selected_document_ids),
            "route_reason": result.route.route_reason,
            "used_reference_count": len(answer_result.references),
            "retrieved_node_count": answer_result.retrieved_node_count,
            "included_node_count": answer_result.included_node_count,
            "rejected_by_score_threshold": answer_result.rejected_by_score_threshold,
        },
        # 外层 tool_execute 的节点日志统一提交，避免工具内部提前拆开 Agent 状态与阶段日志事务。
        commit=False,
    )
    return {
        "tool_name": "knowledge_search",
        "arguments": {"question": typed_args.question},
        "data": payload,
        "usage": usage,
    }
