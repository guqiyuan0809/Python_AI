"""LlamaIndex 在企业知识库离线链路中的落地适配。

本模块有意把职责分成两层：

* **LlamaIndex 层**：``Document``、``IngestionPipeline``、Node Parser、Embedding
  Transform 和 VectorStore 协议。这一层解决“把不同文档变成标准节点并编排处理”的
  通用问题；
* **项目治理层**：候选版本状态、MySQL 原文/来源快照、Milvus Collection 字段契约、
  向量数量校验和 active 发布。这一层决定数据能否进入生产，不应交给框架默认行为。

因此，本文件不是调用 ``VectorStoreIndex.from_documents`` 后把数据直接写进向量库的
Demo。框架产出标准 Node 和 embedding，项目仍以 ``version_id`` 为事务边界写入既有
Milvus，并由外层服务完成发布与审计。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter, get_leaf_nodes
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore, VectorStoreQuery, VectorStoreQueryResult
from pydantic import Field, PrivateAttr

from day04_app.schemas.knowledge_schema import (
    ChunkSourceReference,
    KnowledgeParentTextChunk,
    KnowledgeTextChunk,
    ParentChildTextChunkBuildResult,
    ParsedDocumentSegment,
)
from day04_app.services.knowledge_embedding_service import generate_text_embeddings
from day04_app.services.milvus_vector_store_service import upsert_chunk_vectors
from settings import settings


@dataclass(frozen=True)
class _SourceSpan:
    """拼接后的 LlamaIndex Document 中一个原始段的字符范围。

    LlamaIndex Node 的 ``start_char_idx/end_char_idx`` 指向完整 Document。项目需要把这
    两个框架内部坐标重新投影到原始 ``segment_index/location``，否则数据库中的回答
    引用将无法精确追溯到 Word/PDF 的段落。
    """

    start: int
    end: int
    reference: ChunkSourceReference


@dataclass(frozen=True)
class LlamaIndexNodeBuildResult:
    """离线节点构建结果，供版本化持久化层消费而不暴露框架对象给 HTTP 层。"""

    parent_chunks: list[KnowledgeParentTextChunk]
    child_chunks: list[KnowledgeTextChunk]
    framework_config: dict[str, int | str]


def _build_document_and_spans(
    *,
    document_id: str,
    segments: list[ParsedDocumentSegment],
) -> tuple[Document, list[_SourceSpan]]:
    """把项目已解析且可追溯的段拼为 LlamaIndex Document。

    旧实现 ``text_chunker_service`` 自己负责遍历段落、标题和句子；切换后，文本切分
    算法交给 Node Parser，但来源定位不能丢。因此这里额外保留每个段在拼接文本中的
    范围，后续根据 Node 的字符偏移恢复 ``source_references``。
    """

    text_parts: list[str] = []
    spans: list[_SourceSpan] = []
    current_offset = 0
    for segment in segments:
        content = segment.text.strip()
        if not content:
            continue
        if text_parts:
            # 分隔符不属于任何原始段，不能错误地记录为来源文本。
            text_parts.append("\n\n")
            current_offset += 2
        start = current_offset
        text_parts.append(content)
        current_offset += len(content)
        spans.append(
            _SourceSpan(
                start=start,
                end=current_offset,
                reference=ChunkSourceReference(
                    segment_index=segment.segment_index,
                    location=segment.location,
                ),
            )
        )
    if not text_parts:
        raise ValueError("文档没有可用于 LlamaIndex IngestionPipeline 的文本段")

    # 元数据不放入 Document：LlamaIndex 会把它计入 token 预算；企业来源数据仍由项目
    # MySQL 保存，并在 _source_references_for_node 中按 offset 恢复。
    return Document(text="".join(text_parts), id_=document_id), spans


def _source_references_for_node(node: BaseNode, spans: list[_SourceSpan]) -> list[ChunkSourceReference]:
    """把 LlamaIndex Node 的字符范围映射回项目稳定的原始来源锚点。"""

    start = node.start_char_idx
    end = node.end_char_idx
    if start is None or end is None:
        # SentenceSplitter/HierarchicalNodeParser 正常会保留偏移。这里不能静默返回空来源，
        # 否则后续模型可能生成“无法定位”的引用，宁可在候选版本构建阶段失败。
        raise ValueError("LlamaIndex Node 缺少字符偏移，无法建立企业来源引用")
    matched = [
        span.reference
        for span in spans
        if span.end > start and span.start < end
    ]
    if not matched:
        raise ValueError("LlamaIndex Node 未映射到任何原始文档段")
    return matched


def _as_text_nodes(nodes: Sequence[BaseNode]) -> list[TextNode]:
    """当前 Node Parser 产出 TextNode；显式校验防止未来接入图片/表格节点后误入文本索引。"""

    text_nodes = [node for node in nodes if isinstance(node, TextNode) and node.text.strip()]
    if not text_nodes:
        raise ValueError("LlamaIndex 未生成可向量化 TextNode")
    return text_nodes


def build_llamaindex_text_chunks(
    *,
    document_id: str,
    segments: list[ParsedDocumentSegment],
    chunk_size: int,
    chunk_overlap: int,
) -> LlamaIndexNodeBuildResult:
    """使用 ``IngestionPipeline + SentenceSplitter`` 生成普通检索块。

    这是旧 ``build_text_chunks`` 的框架化替代：旧代码手写“段落/标题/句子”循环；现在
    ``SentenceSplitter`` 是可替换的 Node Parser，``IngestionPipeline`` 是统一的离线
    编排入口。版本持久化、原文来源和上线决策仍由项目层完成。

    ``chunk_size`` 是 LlamaIndex 的 *token* 预算。现有 API 字段仍叫
    ``max_characters`` 仅为向后兼容；调用方会将该值作为近似 token 预算快照保存，新的
    版本记录会明确标记 ``unit=token``，避免把两种单位混为一谈。
    """

    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("LlamaIndex chunk_size/chunk_overlap 参数不合法")
    document, spans = _build_document_and_spans(document_id=document_id, segments=segments)
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_metadata=False,
        include_prev_next_rel=False,
    )
    # 即使当前只有一个 Transform，也用 Pipeline：后续加入 Markdown/HTML Parser、清洗、
    # 元数据抽取或 Embedding 时不需要重写业务服务的流程控制。
    nodes = _as_text_nodes(IngestionPipeline(transformations=[splitter]).run(documents=[document]))
    chunks = [
        KnowledgeTextChunk(
            document_id=document_id,
            chunk_index=index,
            content=node.text.strip(),
            char_count=len(node.text.strip()),
            source_references=_source_references_for_node(node, spans),
        )
        for index, node in enumerate(nodes)
    ]
    return LlamaIndexNodeBuildResult(
        parent_chunks=[],
        child_chunks=chunks,
        framework_config={
            "framework": "llamaindex",
            "pipeline": "IngestionPipeline",
            "node_parser": "SentenceSplitter",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "unit": "token",
        },
    )


def build_llamaindex_parent_child_chunks(
    *,
    document_id: str,
    segments: list[ParsedDocumentSegment],
    parent_chunk_size: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
) -> LlamaIndexNodeBuildResult:
    """使用 ``HierarchicalNodeParser`` 生成父节点和叶子节点。

    旧项目的父子实现是：手写父块、手写子块、通过 ``parent_index`` 关联。
    框架化后：

    ``Document → HierarchicalNodeParser → parent TextNode / child TextNode``

    LlamaIndex 在 Node relationship 中维护父子关系；本函数将关系投影到项目的
    ``parent_chunk_id`` 外键模型。之后检索命中 child，项目仍可审计地回填 parent。
    ``AutoMergingRetriever`` 需要框架 docstore 常驻保存所有父节点；本项目的父块本来
    就在 MySQL，因此先采用“框架建层级、项目按外键回填”的企业可审计方案。
    """

    if parent_chunk_size <= 0 or child_chunk_size <= 0:
        raise ValueError("LlamaIndex 父子 chunk token 预算必须大于 0")
    if child_chunk_overlap < 0 or child_chunk_overlap >= child_chunk_size:
        raise ValueError("LlamaIndex child_chunk_overlap 参数不合法")
    if child_chunk_size >= parent_chunk_size:
        raise ValueError("父节点 token 预算必须大于子节点 token 预算")

    document, spans = _build_document_and_spans(document_id=document_id, segments=segments)
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[parent_chunk_size, child_chunk_size],
        chunk_overlap=child_chunk_overlap,
        include_metadata=False,
        include_prev_next_rel=False,
    )
    all_nodes = _as_text_nodes(IngestionPipeline(transformations=[parser]).run(documents=[document]))
    # 两层 parser 的顶层 Node 没有 PARENT relationship；叶节点才承担 Embedding/召回。
    parent_nodes = [node for node in all_nodes if node.parent_node is None]
    child_nodes = _as_text_nodes(get_leaf_nodes(all_nodes))
    if not parent_nodes:
        raise ValueError("LlamaIndex 未生成父节点")

    parent_index_by_node_id = {node.node_id: index for index, node in enumerate(parent_nodes)}
    parent_chunks = [
        KnowledgeParentTextChunk(
            document_id=document_id,
            parent_index=index,
            content=node.text.strip(),
            char_count=len(node.text.strip()),
            source_references=_source_references_for_node(node, spans),
        )
        for index, node in enumerate(parent_nodes)
    ]
    child_chunks: list[KnowledgeTextChunk] = []
    for index, child_node in enumerate(child_nodes):
        parent_info = child_node.parent_node
        parent_index = parent_index_by_node_id.get(parent_info.node_id if parent_info else "")
        if parent_index is None:
            raise ValueError("LlamaIndex 子节点未找到直接父节点，不能建立数据库外键")
        content = child_node.text.strip()
        child_chunks.append(
            KnowledgeTextChunk(
                document_id=document_id,
                chunk_index=index,
                parent_index=parent_index,
                content=content,
                char_count=len(content),
                source_references=_source_references_for_node(child_node, spans),
            )
        )
    return LlamaIndexNodeBuildResult(
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
        framework_config={
            "framework": "llamaindex",
            "pipeline": "IngestionPipeline",
            "node_parser": "HierarchicalNodeParser",
            "parent_chunk_size": parent_chunk_size,
            "child_chunk_size": child_chunk_size,
            "child_chunk_overlap": child_chunk_overlap,
            "unit": "token",
        },
    )


class ProjectGovernedEmbedding(BaseEmbedding):
    """将项目统一 Embedding 调用适配为 LlamaIndex Transform。

    LlamaIndex 的好处是离线 Node 与在线 Query 都使用同一个 ``BaseEmbedding`` 抽象；
    但 API Key、模型名、超时和异常类型仍必须由项目 ``knowledge_embedding_service``
    统一治理，不能让框架插件再创建一套不可审计的客户端。
    """

    model_name: str = Field(default_factory=lambda: settings.dashscope_embedding_model)

    def _get_query_embedding(self, query: str) -> list[float]:
        _, vectors = generate_text_embeddings([query], batch_size=self.embed_batch_size)
        return vectors[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        _, vectors = generate_text_embeddings([text], batch_size=self.embed_batch_size)
        return vectors[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        # Pipeline 会按 embed_batch_size 调用该方法；项目服务再保留网络调用与返回完整性校验。
        _, vectors = generate_text_embeddings(texts, batch_size=self.embed_batch_size)
        return vectors

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


class ProjectVersionedMilvusVectorStore(BasePydanticVectorStore):
    """LlamaIndex VectorStore 协议到项目 Milvus 契约的写入适配器。

    它故意只开放 ``add``：框架负责把带 embedding 的 Node 交给 VectorStore；项目仍负责
    Collection schema、版本过滤、删除范围和 active 发布。若直接使用框架默认 Collection，
    会绕过 ``version_id``、数量校验和 MySQL 原文回填，反而削弱企业治理。
    """

    document_id: str
    version_id: str
    stores_text: bool = False
    _last_written_node_ids: list[str] = PrivateAttr(default_factory=list)

    @property
    def client(self) -> None:
        """底层 Milvus 客户端只由项目服务创建，避免框架生命周期绕过连接治理。"""

        return None

    @property
    def last_written_node_ids(self) -> list[str]:
        return list(self._last_written_node_ids)

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        del kwargs
        records: list[dict[str, Any]] = []
        for node in nodes:
            if node.embedding is None:
                raise ValueError("LlamaIndex Node 缺少 embedding，拒绝写入 Milvus")
            node_document_id = str(node.metadata.get("document_id", self.document_id))
            node_version_id = str(node.metadata.get("version_id", self.version_id))
            if node_document_id != self.document_id or node_version_id != self.version_id:
                raise ValueError("VectorStore Node 跨越了当前企业文档版本边界")
            records.append(
                {
                    # 这里使用项目 chunk_id，保证 MySQL、Milvus、引用表三处使用同一个业务键。
                    "chunk_id": node.node_id,
                    "document_id": self.document_id,
                    "version_id": self.version_id,
                    "embedding": list(node.embedding),
                }
            )
        upsert_chunk_vectors(records=records)
        self._last_written_node_ids = [node.node_id for node in nodes]
        return self.last_written_node_ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        del ref_doc_id, delete_kwargs
        # 删除必须由候选版本重建服务按 version_id 显式执行，不能允许框架按 Document ID
        # 模糊删除而误伤 active 向量。
        raise NotImplementedError("Milvus 删除由项目版本治理服务统一处理")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        del query, kwargs
        # 在线读取使用 ExistingKnowledgeMilvusRetriever：它会验证 active_version、回填 MySQL
        # 原文并记录 Trace。不能让 VectorStoreIndex 绕过这些业务规则直接查询。
        raise NotImplementedError("线上检索必须使用项目受治理的 LlamaIndex Retriever")


def index_chunks_with_llamaindex(
    *,
    document_id: str,
    version_id: str,
    chunks: Sequence[Any],
    embedding_batch_size: int,
) -> tuple[str, list[str]]:
    """通过 ``IngestionPipeline`` 完成 Node → Embedding → VectorStore.add。

    替代旧代码中的手写“文本列表 → generate_text_embeddings → zip(records) → upsert”循环。
    仍然由调用方校验向量数、更新版本状态和决定是否发布；这正是框架编排与企业治理的
    分界线。
    """

    nodes = [
        TextNode(
            id_=chunk.chunk_id,
            text=chunk.embedding_text or chunk.content,
            metadata={"document_id": document_id, "version_id": version_id},
            # version/document 是跨库治理元数据，不能混进语义向量文本；否则框架默认的
            # get_content 会改变旧链路实际送给 Embedding 模型的文本，导致无法公平回归。
            excluded_embed_metadata_keys=["document_id", "version_id"],
        )
        for chunk in chunks
    ]
    embedding = ProjectGovernedEmbedding(
        model_name=settings.dashscope_embedding_model,
        embed_batch_size=embedding_batch_size,
    )
    vector_store = ProjectVersionedMilvusVectorStore(
        document_id=document_id,
        version_id=version_id,
    )
    # Pipeline 的 vector_store 参数会在所有 Transform 完成后调用 vector_store.add(nodes)。
    # 也就是说这里是真实使用了 LlamaIndex 的 Ingestion 编排，而非仅把它当数据类。
    IngestionPipeline(
        transformations=[embedding],
        vector_store=vector_store,
    ).run(nodes=nodes)
    return embedding.model_name, vector_store.last_written_node_ids
