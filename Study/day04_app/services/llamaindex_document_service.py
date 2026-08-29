"""Day31 LlamaIndex 文档切块教学适配层。

该服务只负责把已经落库的原始法规段转换为 LlamaIndex Document/Node。
它是可观察的预览链路，不直接写 MySQL 或 Milvus，避免在学习框架时绕过现有的
文档版本发布流程。正式入库仍需经过项目自己的版本、向量和评测门禁。
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from day04_app.schemas.knowledge_schema import ParsedDocumentSegment


@dataclass(frozen=True)
class LlamaIndexDocumentBuildResult:
    """LlamaIndex 切块结果及其可观测参数快照。"""

    nodes: list[TextNode]
    source_segment_count: int
    chunk_size: int
    chunk_overlap: int


def build_llamaindex_nodes_from_segments(
    *,
    document_id: str,
    segments: list[ParsedDocumentSegment],
    chunk_size: int,
    chunk_overlap: int,
) -> LlamaIndexDocumentBuildResult:
    """用 LlamaIndex SentenceSplitter 将项目原始段转换为 TextNode。

    每个原始段保留自己的 ``segment_index`` 和 ``location`` 元数据，后续回答引用可以
    追溯到法规页码/段落。与项目自研切块不同，LlamaIndex 这里使用 token-aware 的
    SentenceSplitter；两者可以在同一个样本上对照评测，而不是静默替换线上策略。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")

    documents = [
        Document(
            text=segment.text,
            doc_id=f"{document_id}:segment:{segment.segment_index}",
            metadata={
                "document_id": document_id,
                "segment_index": str(segment.segment_index),
                "source_location": segment.location,
            },
        )
        for segment in segments
        if segment.text.strip()
    ]
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_metadata=True,
        include_prev_next_rel=False,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return LlamaIndexDocumentBuildResult(
        nodes=[node for node in nodes if isinstance(node, TextNode)],
        source_segment_count=len(documents),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
