"""Day31 LlamaIndex 法规检索适配层。

这一层刻意不让 LlamaIndex 直接接管现有 Milvus Collection：现有 Collection、文档版本、
Embedding 模型、Reranker 与来源审计已经是线上治理契约。适配层把项目已经验证过的
Milvus 命中转换为 LlamaIndex ``NodeWithScore``，让后续的 LlamaIndex Query Engine 或
LangChain Tool 可以消费标准节点，而不绕过这些治理边界。
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from sqlalchemy.orm import Session

from day04_app.schemas.knowledge_schema import MilvusChunkSearchItem
from day04_app.services.knowledge_milvus_search_service import search_active_document_chunks


@dataclass(frozen=True)
class LlamaIndexRetrievalResult:
    """框架节点和现有索引治理元数据的统一返回值。"""

    embedding_model: str
    vector_dimension: int
    active_version_id: str
    source_items: list[MilvusChunkSearchItem]
    nodes: list[NodeWithScore]


def to_llamaindex_node(item: MilvusChunkSearchItem) -> NodeWithScore:
    """将项目检索 DTO 转为 LlamaIndex 的标准 NodeWithScore。

    ``TextNode`` 对应 LlamaIndex 的“可被后续 Query Engine/Agent 消费的知识节点”；
    ``NodeWithScore`` 对应当前这次召回中该节点及其相关性分数。原始来源位置和版本 ID
    保留在 metadata，不能因框架适配而丢失可审计性。
    """

    source_references = item.parent_source_references or item.source_references
    node = TextNode(
        # 父子切块时保留现有“子块负责命中、父块负责回答上下文”的策略。
        text=item.parent_content or item.content,
        id_=item.chunk_id,
        metadata={
            "document_id": item.document_id,
            "version_id": item.version_id,
            "chunk_id": item.chunk_id,
            "chunk_index": item.chunk_index,
            "parent_chunk_id": item.parent_chunk_id,
            "source_locations": [reference.location for reference in source_references],
            "vector_score": item.vector_score,
            "rerank_score": item.rerank_score,
            "retrieval_backend": "project_milvus",
        },
    )
    return NodeWithScore(node=node, score=item.score)


class ExistingKnowledgeMilvusRetriever(BaseRetriever):
    """使用现有 active 文档版本与 Milvus 检索的 LlamaIndex Retriever。

    它的价值不是再写一份向量检索，而是把项目的 ``MilvusChunkSearchItem`` 转成
    LlamaIndex 标准节点。这样框架后续负责编排，项目仍负责版本过滤、Embedding 契约、
    Reranker、Trace 与来源回填。
    """

    def __init__(
        self,
        *,
        db: Session,
        document_id: str,
        top_k: int,
        use_reranker: bool,
        rerank_top_n: int | None,
        trace_id: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> None:
        super().__init__()
        self._db = db
        self._document_id = document_id
        self._top_k = top_k
        self._use_reranker = use_reranker
        self._rerank_top_n = rerank_top_n
        self._trace_id = trace_id
        self._task_id = task_id
        self._session_id = session_id
        self._message_id = message_id
        self.last_result: LlamaIndexRetrievalResult | None = None
        # QueryEngine 前置安全门禁会先检索一次，用于在“没有可用资料”时跳过模型调用。
        # 同一个 QueryEngine 随后再次调用 Retriever 时，必须复用同一批节点；否则会产生
        # 双倍 Embedding/Milvus 成本和两组重复审计日志。
        self._cached_query: str | None = None
        self._cached_nodes: list[NodeWithScore] | None = None
        self._cached_result: LlamaIndexRetrievalResult | None = None

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if self._cached_query == query_bundle.query_str and self._cached_nodes is not None:
            # NodePostprocessor 对节点使用 deep copy，因此返回缓存原件不会被引用编号、
            # metadata 过滤等后续处理污染。
            self.last_result = self._cached_result
            return list(self._cached_nodes)
        model, dimension, active_version_id, items = search_active_document_chunks(
            self._db,
            document_id=self._document_id,
            question=query_bundle.query_str,
            top_k=self._top_k,
            use_reranker=self._use_reranker,
            rerank_top_n=self._rerank_top_n,
            trace_id=self._trace_id,
            task_id=self._task_id,
            session_id=self._session_id,
            message_id=self._message_id,
        )
        nodes = [to_llamaindex_node(item) for item in items]
        self.last_result = LlamaIndexRetrievalResult(
            embedding_model=model,
            vector_dimension=dimension,
            active_version_id=active_version_id,
            source_items=items,
            nodes=nodes,
        )
        self._cached_query = query_bundle.query_str
        self._cached_nodes = list(nodes)
        self._cached_result = self.last_result
        return nodes


def retrieve_active_document_as_llamaindex_nodes(
    db: Session,
    *,
    document_id: str,
    question: str,
    top_k: int,
    use_reranker: bool = False,
    rerank_top_n: int | None = None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
) -> LlamaIndexRetrievalResult:
    """对外服务函数：执行框架 Retriever 并返回可观察的转换结果。"""

    retriever = ExistingKnowledgeMilvusRetriever(
        db=db,
        document_id=document_id,
        top_k=top_k,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
    )
    retriever.retrieve(question)
    if retriever.last_result is None:  # 防御式检查，BaseRetriever 正常执行后一定会赋值。
        raise RuntimeError("LlamaIndex Retriever 未返回检索结果")
    return retriever.last_result
