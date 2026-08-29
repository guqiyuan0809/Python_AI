"""Day31：以 LlamaIndex QueryEngine 编排既有企业 RAG 链路。

这不是 ``VectorStoreIndex.from_documents`` 的玩具式接入。项目已经拥有经过版本、
评测和审计治理的 Milvus 检索能力，因此保留：

* 文档版本与 active 指针；
* Query Embedding、Milvus 召回与可选 Reranker；
* Prompt 版本、引用校验和调用日志。

本模块让 LlamaIndex 实际接管的是“Retriever 返回 NodeWithScore 后，如何过滤节点、
组织上下文并驱动模型回答”的 QueryEngine 编排层。这样后续可以继续增加
Node Postprocessor、Response Synthesizer 或 Agent Tool，而无需重写底层数据治理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CustomLLM,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from pydantic import Field, PrivateAttr
from sqlalchemy.orm import Session

from day04_app.common.exceptions import ModelCallException
from day04_app.schemas.knowledge_schema import RagContextReference
from day04_app.services.chat_service import call_chat_completion, create_client
from day04_app.services.llamaindex_law_retrieval_service import (
    ExistingKnowledgeMilvusRetriever,
    LlamaIndexRetrievalResult,
)
from day04_app.services.prompt_observability_service import PromptIdentity
from day04_app.services.rag_context_service import (
    NO_ANSWER_FALLBACK_TEXT,
    extract_cited_source_ids,
    get_active_rag_answer_prompt,
    get_rag_answer_prompt_identity,
)
from settings import settings


@dataclass(frozen=True)
class LlamaIndexModelUsage:
    """LlamaIndex LLM 适配器捕获到的真实模型用量。"""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class LlamaIndexRagAnswerResult:
    """QueryEngine 执行结果的项目稳定投影，不把框架对象直接暴露到 HTTP 层。"""

    answer: str
    references: list[RagContextReference]
    retrieval: LlamaIndexRetrievalResult
    retrieved_node_count: int
    included_node_count: int
    omitted_node_count: int
    top_score: float | None
    score_threshold: float | None
    rejected_by_score_threshold: bool
    context_char_count: int
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prompt_identity: PromptIdentity | None


@dataclass(frozen=True)
class LlamaIndexRagPreparation:
    """QueryEngine 调用前的受治理检索结果。

    会话同步入口和 Celery Worker 都需要分别记录“检索/上下文”与“模型回答”阶段日志。
    无框架实现靠 ``build_rag_context`` 返回一个手写字符串；框架化后不提前拼字符串，
    而是保留 ``retriever + postprocessor``，在 QueryEngine 执行时按同一规则生成上下文。
    """

    # 单文档时是 ExistingKnowledgeMilvusRetriever；多文档时是 RouterRetriever。
    # 二者都是 LlamaIndex 标准 BaseRetriever，因此同一个 QueryEngine 和后处理器可以
    # 复用，而不需要复制一份 Prompt、引用校验或拒答代码。
    retriever: BaseRetriever
    retrieval_result_getter: Callable[[], LlamaIndexRetrievalResult | None]
    postprocessor: "GovernedRagNodePostprocessor"
    runtime_prompt: object


@dataclass(frozen=True)
class LlamaIndexContextPreview:
    """模型调用前的框架节点预览，HTTP 层只取这个稳定投影。"""

    retrieval: LlamaIndexRetrievalResult
    source_nodes: list[NodeWithScore]
    references: list[RagContextReference]
    context: str
    omitted_node_count: int
    top_score: float | None
    score_threshold: float | None
    rejected_by_score_threshold: bool


class GovernedRagNodePostprocessor(BaseNodePostprocessor):
    """把项目已有的上下文治理规则挂到 LlamaIndex 的标准后处理扩展点。

    它替代旧 ``build_rag_context`` 中与“选择哪些资料进入模型”有关的那一段逻辑：
    分数阈值拒答、父子块去重、上下文预算和引用编号。真正的上下文拼装由
    ``RetrieverQueryEngine`` / Response Synthesizer 完成。
    """

    max_context_characters: int = Field(..., ge=1)
    score_threshold: float | None = Field(default=None, ge=-1, le=1)
    _top_score: float | None = PrivateAttr(default=None)
    _rejected_by_score_threshold: bool = PrivateAttr(default=False)
    _omitted_node_count: int = PrivateAttr(default=0)
    _context_char_count: int = PrivateAttr(default=0)

    @property
    def top_score(self) -> float | None:
        return self._top_score

    @property
    def rejected_by_score_threshold(self) -> bool:
        return self._rejected_by_score_threshold

    @property
    def omitted_node_count(self) -> int:
        return self._omitted_node_count

    @property
    def context_char_count(self) -> int:
        return self._context_char_count

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        del query_bundle
        self._top_score = nodes[0].score if nodes else None
        self._rejected_by_score_threshold = bool(
            self.score_threshold is not None
            and (self._top_score is None or self._top_score < self.score_threshold)
        )
        self._omitted_node_count = 0
        self._context_char_count = 0
        if self._rejected_by_score_threshold:
            self._omitted_node_count = len(nodes)
            return []

        selected_nodes: list[NodeWithScore] = []
        seen_parent_keys: set[str] = set()
        for index, source_node in enumerate(nodes):
            # QueryEngine 后处理器可能会被复用；必须复制节点，不能污染原始检索结果。
            node_with_score = source_node.model_copy(deep=True)
            metadata = node_with_score.node.metadata
            parent_key = str(metadata.get("parent_chunk_id") or node_with_score.node.node_id)
            if parent_key in seen_parent_keys:
                self._omitted_node_count += 1
                continue
            seen_parent_keys.add(parent_key)

            # 当前 source_id 才是要写入 Prompt、供模型输出 [S1] 的引用编号。
            source_id = f"S{len(selected_nodes) + 1}"
            locations = metadata.get("source_locations") or []
            metadata["citation"] = f"[{source_id}]"
            metadata["source_locations"] = "、".join(str(item) for item in locations) or "未提供位置"
            # 给 LLM 的上下文只保留引用编号和位置；document/version/chunk ID 继续保留在
            # 节点 metadata，供 Python 侧审计和返回，不浪费模型上下文。
            node_with_score.node.excluded_llm_metadata_keys = [
                key
                for key in metadata
                if key not in {"citation", "source_locations"}
            ]
            estimated_characters = len(
                node_with_score.node.get_content(metadata_mode=MetadataMode.LLM)
            )
            if self._context_char_count + estimated_characters > self.max_context_characters:
                # 与旧实现一致：不能跳过高排名资料，再塞入排名更低的资料。
                self._omitted_node_count += len(nodes) - index
                break

            selected_nodes.append(node_with_score)
            self._context_char_count += estimated_characters

        return selected_nodes


def prepare_governed_llamaindex_rag(
    db: Session,
    *,
    document_id: str,
    retrieval_top_k: int,
    max_context_characters: int,
    use_reranker: bool = False,
    rerank_top_n: int = 20,
    score_threshold: float | None = None,
    trace_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    message_id: str | None = None,
) -> LlamaIndexRagPreparation:
    """构造一条正式 RAG 的 LlamaIndex 编排契约，但不执行模型调用。

    这相当于旧 ``search_active_document_chunks + build_rag_context`` 的替代入口：
    Retriever 仍受项目的 active 版本、Milvus、Reranker 和 Trace 约束；Postprocessor
    仍执行企业阈值、父块去重、预算和引用编号。真正执行时由 QueryEngine 统一调度，
    从而避免同步 API 与异步 Worker 分叉出不同的上下文策略。
    """

    retriever = ExistingKnowledgeMilvusRetriever(
            db=db,
            document_id=document_id,
            top_k=retrieval_top_k,
            use_reranker=use_reranker,
            rerank_top_n=rerank_top_n,
            trace_id=trace_id,
            task_id=task_id,
            session_id=session_id,
            message_id=message_id,
        )
    return LlamaIndexRagPreparation(
        retriever=retriever,
        retrieval_result_getter=lambda: retriever.last_result,
        postprocessor=GovernedRagNodePostprocessor(
            max_context_characters=max_context_characters,
            score_threshold=score_threshold,
        ),
        runtime_prompt=get_active_rag_answer_prompt(db),
    )


def retrieve_governed_llamaindex_context(
    preparation: LlamaIndexRagPreparation,
    *,
    question: str,
) -> tuple[LlamaIndexRetrievalResult, list[NodeWithScore]]:
    """执行框架 Retriever 与 NodePostprocessor，但不调用聊天模型。

    这是开发预览、检索评测和正式 QueryEngine 共用的“模型前”阶段。旧实现直接返回
    手工字符串；这里返回标准 ``NodeWithScore``，需要展示时再由框架 Node 生成内容，
    这样预览结果与真正送入模型的节点不会出现两套规则。
    """

    retrieved_nodes = preparation.retriever.retrieve(question)
    selected_nodes = preparation.postprocessor.postprocess_nodes(retrieved_nodes, QueryBundle(question))
    retrieval = preparation.retrieval_result_getter()
    if retrieval is None:
        raise RuntimeError("LlamaIndex Retriever 未产生检索结果")
    return retrieval, selected_nodes


def preview_governed_llamaindex_context(
    preparation: LlamaIndexRagPreparation,
    *,
    question: str,
) -> LlamaIndexContextPreview:
    """生成与 QueryEngine 使用同一套 Retriever/Postprocessor 的开发预览。

    旧预览接口在 Router 中直接调用 ``build_rag_context``；那会让预览和正式框架回答有
    两份去重/预算逻辑。现在只把框架最终选中的 Node 内容拼成展示文本，真正的 Prompt
    仍由 QueryEngine 的 Response Synthesizer 负责。
    """

    retrieval, source_nodes = retrieve_governed_llamaindex_context(
        preparation,
        question=question,
    )
    context = "\n\n".join(
        node.node.get_content(metadata_mode=MetadataMode.LLM)
        for node in source_nodes
    ).strip()
    return LlamaIndexContextPreview(
        retrieval=retrieval,
        source_nodes=source_nodes,
        references=_references_from_source_nodes(source_nodes),
        context=context,
        omitted_node_count=preparation.postprocessor.omitted_node_count,
        top_score=preparation.postprocessor.top_score,
        score_threshold=preparation.postprocessor.score_threshold,
        rejected_by_score_threshold=preparation.postprocessor.rejected_by_score_threshold,
    )


class ProjectChatCompletionLlamaIndexLLM(CustomLLM):
    """把项目统一的 OpenAI 兼容模型调用封装为 LlamaIndex LLM。

    DashScope/Qwen 的调用、模型名、temperature 和 max_tokens 仍沿用项目原有治理；
    LlamaIndex 只通过该适配器调用它，避免框架私自读取另一套 API Key 或模型配置。
    """

    runtime_model: str
    temperature: float = 0.1
    max_tokens: int = 800
    _last_usage: LlamaIndexModelUsage | None = PrivateAttr(default=None)

    @property
    def metadata(self) -> LLMMetadata:
        # 这里声明 chat model，QueryEngine 才会把 ChatPromptTemplate 的 system/user
        # 消息原样交给项目的 Qwen 调用，而不是拼成一大段不可审计字符串。
        return LLMMetadata(
            context_window=16000,
            num_output=self.max_tokens,
            is_chat_model=True,
            model_name=self.runtime_model,
        )

    @property
    def last_usage(self) -> LlamaIndexModelUsage | None:
        return self._last_usage

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        del kwargs
        openai_messages = [
            {
                "role": message.role.value,
                "content": message.content or "",
            }
            for message in messages
        ]
        try:
            response = call_chat_completion(
                create_client(timeout=45.0),
                openai_messages,
                model=self.runtime_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                raise ModelCallException(message="RAG 模型返回了空回答")
            usage = response.usage
            self._last_usage = LlamaIndexModelUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=answer),
                raw=response,
            )
        except ModelCallException:
            raise
        except Exception as exc:
            raise ModelCallException(
                message=f"LlamaIndex RAG 模型生成失败：{type(exc).__name__}"
            ) from exc

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        """兼容 LlamaIndex 的 completion 调用路径；当前 QueryEngine 实际使用 chat。"""
        del formatted, kwargs
        response = self.chat([ChatMessage(role=MessageRole.USER, content=prompt)])
        return CompletionResponse(text=response.message.content or "", raw=response.raw)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs,
    ) -> Iterator[CompletionResponse]:
        # 当前 Day31 只接入非流式 QueryEngine；实现该抽象方法以便后续自然升级 SSE。
        completion = self.complete(prompt, formatted=formatted, **kwargs)
        yield CompletionResponse(text=completion.text, delta=completion.text, raw=completion.raw)


def build_llamaindex_rag_chat_prompt(runtime_prompt) -> ChatPromptTemplate:
    """把 ai_prompt_version 的当前 Prompt 转成 LlamaIndex QueryEngine 模板。

    Prompt 的真实内容、版本和哈希没有换；仅把项目的 ``{context}`` / ``{question}``
    占位符映射为 LlamaIndex Response Synthesizer 规定的变量名。
    """

    user_template = runtime_prompt.user_prompt_template
    if "{context}" not in user_template or "{question}" not in user_template:
        raise RuntimeError("RAG active Prompt 必须包含 {context} 和 {question} 占位符")
    user_template = (
        user_template.replace("{context}", "{context_str}")
        .replace("{question}", "{query_str}")
    )
    return ChatPromptTemplate(
        message_templates=[
            ChatMessage(role=MessageRole.SYSTEM, content=runtime_prompt.system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_template),
        ]
    )


def _references_from_source_nodes(
    source_nodes: list[NodeWithScore],
) -> list[RagContextReference]:
    """将 QueryEngine 最终实际送入模型的节点转换为项目原有引用契约。"""

    references: list[RagContextReference] = []
    for node_with_score in source_nodes:
        metadata = node_with_score.node.metadata
        citation = str(metadata.get("citation", ""))
        if not citation.startswith("[S") or not citation.endswith("]"):
            raise RuntimeError("LlamaIndex 节点缺少受控引用编号")
        source_locations = metadata.get("source_locations") or ""
        locations = (
            [item for item in str(source_locations).split("、") if item]
            if source_locations != "未提供位置"
            else []
        )
        references.append(
            RagContextReference(
                source_id=citation[1:-1],
                document_id=str(metadata["document_id"]),
                version_id=str(metadata["version_id"]),
                chunk_id=str(metadata["chunk_id"]),
                chunk_index=int(metadata["chunk_index"]),
                score=float(node_with_score.score or 0.0),
                locations=locations,
            )
        )
    return references


def _validate_answer_citations(
    answer: str,
    references: list[RagContextReference],
) -> list[RagContextReference]:
    """保留项目已有的回答引用校验，框架不能绕过事实可追溯边界。"""

    cited_source_ids = [f"S{number}" for number in extract_cited_source_ids(answer)]
    reference_by_source_id = {reference.source_id: reference for reference in references}
    unknown_source_ids = [
        source_id for source_id in cited_source_ids if source_id not in reference_by_source_id
    ]
    if unknown_source_ids:
        raise ModelCallException(message="RAG 回答引用了不存在的资料编号")
    if not cited_source_ids and NO_ANSWER_FALLBACK_TEXT.rstrip("。") not in answer:
        raise ModelCallException(message="RAG 回答缺少资料引用")
    return [reference_by_source_id[source_id] for source_id in cited_source_ids]


def answer_active_document_with_llamaindex(
    db: Session,
    *,
    document_id: str,
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
) -> LlamaIndexRagAnswerResult:
    """执行真实 LlamaIndex QueryEngine RAG，同时复用项目底层检索治理。"""

    preparation = prepare_governed_llamaindex_rag(
        db=db,
        document_id=document_id,
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
    return answer_prepared_document_with_llamaindex(
        question=question,
        preparation=preparation,
    )


def answer_prepared_document_with_llamaindex(
    *,
    question: str,
    preparation: LlamaIndexRagPreparation,
) -> LlamaIndexRagAnswerResult:
    """执行已准备好的 QueryEngine，并把框架结果投影回项目稳定 DTO。

    ``preparation`` 被单独抽出后，外层可以在回答前后记录阶段日志；但回答本身只有这一
    个框架入口，避免同步 API、异步 Worker 与单文档接口各自拼 Prompt/上下文。
    """

    retriever = preparation.retriever
    postprocessor = preparation.postprocessor
    runtime_prompt = preparation.runtime_prompt
    # QueryEngine 默认可能在空节点时仍调用模型。企业 RAG 必须先做一次“资料存在性
    # 门禁”，否则低相关问题会落到通用模型知识，绕过项目的拒答策略。Retriever 内部
    # 会缓存同一问题的节点，后续 QueryEngine 调度不会再次请求 Milvus。
    preflight_retrieval, preflight_nodes = retrieve_governed_llamaindex_context(
        preparation,
        question=question,
    )
    if postprocessor.rejected_by_score_threshold or not preflight_nodes:
        return LlamaIndexRagAnswerResult(
            answer=NO_ANSWER_FALLBACK_TEXT,
            references=[],
            retrieval=preflight_retrieval,
            retrieved_node_count=len(preflight_retrieval.nodes),
            included_node_count=0,
            omitted_node_count=postprocessor.omitted_node_count,
            top_score=postprocessor.top_score,
            score_threshold=postprocessor.score_threshold,
            rejected_by_score_threshold=postprocessor.rejected_by_score_threshold,
            context_char_count=postprocessor.context_char_count,
            model=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            prompt_identity=None,
        )
    llm = ProjectChatCompletionLlamaIndexLLM(
        runtime_model=runtime_prompt.model or settings.dashscope_model,
        temperature=(runtime_prompt.temperature if runtime_prompt.temperature is not None else 0.1),
        max_tokens=runtime_prompt.max_tokens or 800,
    )
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        text_qa_template=build_llamaindex_rag_chat_prompt(runtime_prompt),
        response_mode="compact",
        node_postprocessors=[postprocessor],
    )
    response = query_engine.query(question)
    retrieval = preparation.retrieval_result_getter()
    if retrieval is None:
        raise RuntimeError("LlamaIndex QueryEngine 未产生检索结果")

    source_nodes = list(response.source_nodes)
    if postprocessor.rejected_by_score_threshold or not source_nodes:
        return LlamaIndexRagAnswerResult(
            answer=NO_ANSWER_FALLBACK_TEXT,
            references=[],
            retrieval=retrieval,
            retrieved_node_count=len(retrieval.nodes),
            included_node_count=0,
            omitted_node_count=postprocessor.omitted_node_count,
            top_score=postprocessor.top_score,
            score_threshold=postprocessor.score_threshold,
            rejected_by_score_threshold=postprocessor.rejected_by_score_threshold,
            context_char_count=postprocessor.context_char_count,
            model=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            prompt_identity=None,
        )

    answer = str(response).strip()
    references = _references_from_source_nodes(source_nodes)
    used_references = _validate_answer_citations(answer, references)
    usage = llm.last_usage
    return LlamaIndexRagAnswerResult(
        answer=answer,
        references=used_references,
        retrieval=retrieval,
        retrieved_node_count=len(retrieval.nodes),
        included_node_count=len(source_nodes),
        omitted_node_count=postprocessor.omitted_node_count,
        top_score=postprocessor.top_score,
        score_threshold=postprocessor.score_threshold,
        rejected_by_score_threshold=False,
        context_char_count=postprocessor.context_char_count,
        model=llm.runtime_model,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        prompt_identity=get_rag_answer_prompt_identity(runtime_prompt),
    )
