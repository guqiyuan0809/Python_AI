"""Day31：多文档知识库的 LlamaIndex 路由与受治理全局检索。

这里刻意区分两种常被混淆的能力：

* ``RouterRetriever``：根据问题在“法规、企业制度、应急预案”等知识域中选择应查的
  文档集合，避免每个问题都检索全部资料；
* ``MultiDocumentMilvusRetriever``：在已选集合的 *active* 版本中一次执行全局 Top-K，
  用统一排序的跨文档证据回答问题。

后者不能由 RouterRetriever 自动替代。若逐文档检索再拼接，会重复生成 query embedding，
也会丢失跨文档的全局相关度排序。权限、文档准入和版本状态先在项目层完成，LlamaIndex
只消费已经允许访问的 Retriever Tool。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from llama_index.core.base.base_selector import BaseSelector, SelectorResult, SingleSelection
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import BaseRetriever, RouterRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.tools import RetrieverTool
from sqlalchemy.orm import Session

from day04_app.common.exceptions import BusinessException
from day04_app.services.knowledge_milvus_search_service import search_active_documents_chunks
from day04_app.services.llamaindex_law_retrieval_service import (
    LlamaIndexRetrievalResult,
    to_llamaindex_node,
)
from day04_app.services.llamaindex_rag_query_service import (
    GovernedRagNodePostprocessor,
    LlamaIndexContextPreview,
    LlamaIndexRagAnswerResult,
    LlamaIndexRagPreparation,
    answer_prepared_document_with_llamaindex,
    preview_governed_llamaindex_context,
)
from day04_app.services.rag_context_service import get_active_rag_answer_prompt


@dataclass(frozen=True)
class KnowledgeDomain:
    """调用方明确允许访问的一组文档，而不是由模型生成 document_id。"""

    domain_id: str
    description: str
    document_ids: tuple[str, ...]


class KeywordKnowledgeDomainSelector(BaseSelector):
    """LlamaIndex Router 的确定性 Selector，作为企业知识路由首个安全版本。

    ``RouterRetriever`` 支持用 LLM Selector 自动选工具，但生产首版不能让模型自由决定
    数据范围。Java/RBAC 已过滤出可访问 domain 后，这个 Selector 只基于调用方配置的
    领域关键词选择一项；没有命中则默认选择全域工具。它仍完全走 LlamaIndex 的
    RouterRetriever 协议，后续可在同一边界替换为受评测的 LLM Selector。
    """

    def __init__(
        self,
        *,
        domain_keywords: dict[str, tuple[str, ...]],
        default_index: int = 0,
    ) -> None:
        self.domain_keywords = domain_keywords
        self.default_index = default_index
        # RouterRetriever 不会把 SelectorResult 暴露给调用方。项目需要把“为什么选了
        # 此知识域”写入审计日志和 HTTP 响应，所以只保存本次确定性选择的事实；不保存
        # 用户问题正文，也不把它当作权限依据。
        self.last_result: SelectorResult | None = None

    # 这个确定性 Selector 没有模型 Prompt；实现 PromptMixin 的空钩子以满足
    # LlamaIndex BaseSelector 协议，并保留未来替换为 LLM Selector 的位置。
    def _get_prompts(self):  # type: ignore[override]
        return {}

    def _update_prompts(self, prompts_dict) -> None:  # type: ignore[override]
        if prompts_dict:
            raise ValueError("KeywordKnowledgeDomainSelector 不支持更新 Prompt")

    def _select(self, choices, query: QueryBundle) -> SelectorResult:  # type: ignore[override]
        query_text = query.query_str.lower()
        for index, choice in enumerate(choices):
            keywords = self.domain_keywords.get(choice.name or "", ())
            if any(keyword.lower() in query_text for keyword in keywords):
                result = SelectorResult(
                    selections=[
                        SingleSelection(
                            index=index,
                            reason="deterministic_domain_keyword_match",
                        )
                    ]
                )
                self.last_result = result
                return result
        result = SelectorResult(
            selections=[
                SingleSelection(index=self.default_index, reason="deterministic_default_domain")
            ]
        )
        self.last_result = result
        return result

    async def _aselect(self, choices, query: QueryBundle) -> SelectorResult:  # type: ignore[override]
        return self._select(choices, query)


class MultiDocumentMilvusRetriever(BaseRetriever):
    """将项目受治理的跨文档 Milvus 检索适配为一个 LlamaIndex Retriever。"""

    def __init__(
        self,
        *,
        db: Session,
        document_ids: Sequence[str],
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
        self._document_ids = tuple(dict.fromkeys(document_ids))
        self._top_k = top_k
        self._use_reranker = use_reranker
        self._rerank_top_n = rerank_top_n
        self._trace_id = trace_id
        self._task_id = task_id
        self._session_id = session_id
        self._message_id = message_id
        self.last_result: LlamaIndexRetrievalResult | None = None
        self.active_version_by_document_id: dict[str, str] = {}
        self._cached_query: str | None = None
        self._cached_nodes: list[NodeWithScore] | None = None

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if self._cached_query == query_bundle.query_str and self._cached_nodes is not None:
            return list(self._cached_nodes)
        model, dimension, active_versions, items = search_active_documents_chunks(
            self._db,
            document_ids=list(self._document_ids),
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
        # LlamaIndexRagAnswerResult 使用单一 active_version_id；多文档场景将完整映射保存
        # 在 Retriever 上，HTTP DTO 和审计日志再显式返回它，避免伪造一个“唯一版本”。
        self.active_version_by_document_id = active_versions
        self.last_result = LlamaIndexRetrievalResult(
            embedding_model=model,
            vector_dimension=dimension,
            active_version_id="multi_document",
            source_items=items,
            nodes=nodes,
        )
        self._cached_query = query_bundle.query_str
        self._cached_nodes = list(nodes)
        return nodes


@dataclass(frozen=True)
class MultiDocumentRouteResult:
    """路由结果：保留选中知识域和真正允许检索的文档，供日志与 HTTP 响应使用。"""

    selected_domain_id: str
    selected_document_ids: list[str]
    route_reason: str


@dataclass(frozen=True)
class MultiDocumentRagAnswerResult:
    """多文档回答的稳定项目投影，不伪造单一 document/version。"""

    answer_result: LlamaIndexRagAnswerResult
    route: MultiDocumentRouteResult
    active_version_by_document_id: dict[str, str]


@dataclass(frozen=True)
class MultiDocumentRagContextPreview:
    """多文档模型前预览，同时展示路由、版本映射和实际资料节点。"""

    preview: LlamaIndexContextPreview
    route: MultiDocumentRouteResult
    active_version_by_document_id: dict[str, str]


@dataclass(frozen=True)
class MultiDocumentRouter:
    """项目对 LlamaIndex ``RouterRetriever`` 的可观察包装。

    LlamaIndex 负责按照 Selector 协议把问题分发到一个子 Retriever；项目包装层只补回
    企业接口必须返回的路由事实（领域 ID、允许文档及理由）。它不绕过 RouterRetriever
    的私有字段，也不自行实现第二套路由/检索逻辑。
    """

    router_retriever: RouterRetriever
    selector: KeywordKnowledgeDomainSelector
    domains: tuple[KnowledgeDomain, ...]
    child_retrievers: tuple[MultiDocumentMilvusRetriever, ...]

    def selected_route(self) -> MultiDocumentRouteResult:
        """读取最近一次 RouterRetriever 真实执行留下的确定性选择结果。"""

        result = self.selector.last_result
        if result is None or len(result.inds) != 1:
            raise BusinessException(code=50057, message="知识域路由必须且只能选择一个领域")
        selected_index = result.ind
        if selected_index < 0 or selected_index >= len(self.domains):
            raise BusinessException(code=50057, message="知识域路由返回了无效领域")
        selected_domain = self.domains[selected_index]
        return MultiDocumentRouteResult(
            selected_domain_id=selected_domain.domain_id,
            selected_document_ids=list(selected_domain.document_ids),
            route_reason=result.reason,
        )

    def selected_retrieval_result(self) -> LlamaIndexRetrievalResult | None:
        """取得 Router 实际选中子 Retriever 的受治理检索投影。"""

        result = self.selector.last_result
        if result is None or len(result.inds) != 1:
            return None
        selected_index = result.ind
        if selected_index < 0 or selected_index >= len(self.child_retrievers):
            return None
        return self.child_retrievers[selected_index].last_result


def build_multi_document_router_retriever(
    *,
    db: Session,
    domains: Sequence[KnowledgeDomain],
    top_k: int,
    use_reranker: bool,
    rerank_top_n: int | None,
    trace_id: str | None = None,
    domain_keywords: dict[str, tuple[str, ...]] | None = None,
) -> tuple[RouterRetriever, list[KnowledgeDomain]]:
    """构造 LlamaIndex RouterRetriever；所有 Tool 都来自调用方允许的知识域。

    保留旧返回契约给已写好的教学调用；正式多文档 QueryEngine 使用下方的
    :func:`build_multi_document_router`，以便获取选中领域的可观察事实。
    """

    router = build_multi_document_router(
        db=db,
        domains=domains,
        top_k=top_k,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
        trace_id=trace_id,
        domain_keywords=domain_keywords,
    )
    return router.router_retriever, list(router.domains)


def build_multi_document_router(
    *,
    db: Session,
    domains: Sequence[KnowledgeDomain],
    top_k: int,
    use_reranker: bool,
    rerank_top_n: int | None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
    domain_keywords: dict[str, tuple[str, ...]] | None = None,
    default_domain_id: str | None = None,
) -> MultiDocumentRouter:
    """创建正式多文档路由器，并保存每个领域对应的子 Retriever。

    ``document_ids`` 必须来自业务层按租户、园区和角色过滤后的结果。当前教学项目
    还没有 ``knowledge_domain`` 数据表，因此 HTTP 预览显式传入允许集合；接金汤令时
    应由 Java/Python 服务端根据数据范围构造它，绝不能信任浏览器自报的文档范围。
    """

    normalized_domains = [domain for domain in domains if domain.document_ids]
    if not normalized_domains:
        raise BusinessException(code=40074, message="至少提供一个包含文档的知识域")
    if len({domain.domain_id for domain in normalized_domains}) != len(normalized_domains):
        raise BusinessException(code=40075, message="知识域 domain_id 不能重复")

    tools: list[RetrieverTool] = []
    child_retrievers: list[MultiDocumentMilvusRetriever] = []
    for domain in normalized_domains:
        retriever = MultiDocumentMilvusRetriever(
            db=db,
            document_ids=domain.document_ids,
            top_k=top_k,
            use_reranker=use_reranker,
            rerank_top_n=rerank_top_n,
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=message_id,
        )
        child_retrievers.append(retriever)
        tools.append(
            RetrieverTool.from_defaults(
                retriever=retriever,
                name=domain.domain_id,
                description=domain.description,
            )
        )
    default_index = 0
    if default_domain_id is not None:
        matching_indexes = [
            index
            for index, domain in enumerate(normalized_domains)
            if domain.domain_id == default_domain_id
        ]
        if not matching_indexes:
            raise BusinessException(code=40076, message="默认知识域不在允许的领域集合中")
        default_index = matching_indexes[0]
    selector = KeywordKnowledgeDomainSelector(
        domain_keywords=domain_keywords or {},
        default_index=default_index,
    )
    return MultiDocumentRouter(
        # 当前 Selector 完全确定性，不会调用 LLM；但 RouterRetriever 构造时仍会读取
        # ``Settings.llm``。显式给一个永不执行的 MockLLM，避免框架为了默认 OpenAI LLM
        # 适配器引入无关依赖。真正的回答模型仍在 RetrieverQueryEngine 中注入项目适配器。
        router_retriever=RouterRetriever(
            selector=selector,
            retriever_tools=tools,
            llm=MockLLM(),
        ),
        selector=selector,
        domains=tuple(normalized_domains),
        child_retrievers=tuple(child_retrievers),
    )


def route_and_retrieve_multi_document_context(
    *,
    router: MultiDocumentRouter,
    question: str,
) -> tuple[MultiDocumentRouteResult, list[NodeWithScore]]:
    """执行 RouterRetriever，并从其选择结果恢复项目需要的可观察路由事实。"""

    # 这里必须真正调用 RouterRetriever，而不是读取它的 _selector/_retrievers 私有字段
    # 后手工调子 Retriever。这样 callbacks、未来 LLM Selector 与框架行为都保持一致。
    nodes = router.router_retriever.retrieve(question)
    return router.selected_route(), nodes


def prepare_multi_document_llamaindex_rag(
    db: Session,
    *,
    domains: Sequence[KnowledgeDomain],
    domain_keywords: dict[str, tuple[str, ...]],
    default_domain_id: str | None,
    retrieval_top_k: int,
    max_context_characters: int,
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    score_threshold: float | None = None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
) -> tuple[MultiDocumentRouter, LlamaIndexRagPreparation]:
    """准备多文档 QueryEngine，但不调用模型。

    与单文档 ``prepare_governed_llamaindex_rag`` 的差异只有 Retriever：这里交给
    ``RouterRetriever``，而 Prompt 版本、NodePostprocessor 和拒答策略仍完全复用。
    因而“多文档”不是新造一条绕过治理的 RAG 链路。
    """

    router = build_multi_document_router(
        db=db,
        domains=domains,
        top_k=retrieval_top_k,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
        domain_keywords=domain_keywords,
        default_domain_id=default_domain_id,
    )
    preparation = LlamaIndexRagPreparation(
        retriever=router.router_retriever,
        retrieval_result_getter=router.selected_retrieval_result,
        postprocessor=GovernedRagNodePostprocessor(
            max_context_characters=max_context_characters,
            score_threshold=score_threshold,
        ),
        runtime_prompt=get_active_rag_answer_prompt(db),
    )
    return router, preparation


def answer_multi_document_with_llamaindex(
    db: Session,
    *,
    domains: Sequence[KnowledgeDomain],
    domain_keywords: dict[str, tuple[str, ...]],
    default_domain_id: str | None,
    question: str,
    retrieval_top_k: int,
    max_context_characters: int,
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    score_threshold: float | None = None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
) -> MultiDocumentRagAnswerResult:
    """按已授权知识域路由后，使用同一 QueryEngine 生成跨文档带引用回答。"""

    router, preparation = prepare_multi_document_llamaindex_rag(
        db,
        domains=domains,
        domain_keywords=domain_keywords,
        default_domain_id=default_domain_id,
        retrieval_top_k=retrieval_top_k,
        max_context_characters=max_context_characters,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
        score_threshold=score_threshold,
        trace_id=trace_id,
        task_id=task_id,
        session_id=session_id,
        message_id=message_id,
    )
    answer_result = answer_prepared_document_with_llamaindex(
        question=question,
        preparation=preparation,
    )
    route = router.selected_route()
    selected_retrieval = router.selected_retrieval_result()
    if selected_retrieval is None:
        raise RuntimeError("多文档 RouterRetriever 未产生检索结果")
    selected_index = router.selector.last_result.ind if router.selector.last_result else -1
    active_versions = dict(router.child_retrievers[selected_index].active_version_by_document_id)
    return MultiDocumentRagAnswerResult(
        answer_result=answer_result,
        route=route,
        active_version_by_document_id=active_versions,
    )


def preview_multi_document_llamaindex_context(
    db: Session,
    *,
    domains: Sequence[KnowledgeDomain],
    domain_keywords: dict[str, tuple[str, ...]],
    default_domain_id: str | None,
    question: str,
    retrieval_top_k: int,
    max_context_characters: int,
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    score_threshold: float | None = None,
    trace_id: str | None = None,
) -> MultiDocumentRagContextPreview:
    """在模型调用前停止，供开发验证“路由后哪些跨文档证据会进入 Prompt”。"""

    router, preparation = prepare_multi_document_llamaindex_rag(
        db,
        domains=domains,
        domain_keywords=domain_keywords,
        default_domain_id=default_domain_id,
        retrieval_top_k=retrieval_top_k,
        max_context_characters=max_context_characters,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
        score_threshold=score_threshold,
        trace_id=trace_id,
    )
    preview = preview_governed_llamaindex_context(preparation, question=question)
    route = router.selected_route()
    selected_index = router.selector.last_result.ind if router.selector.last_result else -1
    if selected_index < 0:
        raise RuntimeError("多文档 RouterRetriever 未产生路由结果")
    return MultiDocumentRagContextPreview(
        preview=preview,
        route=route,
        active_version_by_document_id=dict(
            router.child_retrievers[selected_index].active_version_by_document_id
        ),
    )
