"""Day31：验证正式 LlamaIndex 离线/在线适配的关键契约，不访问真实 MySQL、Milvus 或模型。

这个脚本不是为了证明框架“能跑”而写的玩具测试，而是防止改造后发生三类企业风险：

1. Node Parser 切块后丢失 Word/PDF 原始段落来源；
2. 父子 Node relationship 无法投影为项目的 ``parent_chunk_id``；
3. LlamaIndex Pipeline 把治理 metadata 混进 Embedding 文本，导致与旧版本无法公平回归。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.schemas.knowledge_schema import ParsedDocumentSegment
from day04_app.services import llamaindex_ingestion_service as ingestion


class _Chunk:
    """最小化模拟已持久化 chunk，避免测试触碰真实数据库。"""

    def __init__(self, chunk_id: str, content: str) -> None:
        self.chunk_id = chunk_id
        self.content = content
        self.embedding_text = None


def main() -> None:
    segments = [
        ParsedDocumentSegment(
            segment_index=10,
            location="Paragraph:10",
            text="JVM 负责执行 Java 字节码，并提供运行时内存管理。" * 16,
        ),
        ParsedDocumentSegment(
            segment_index=11,
            location="Paragraph:11",
            text="垃圾回收器会回收不可达对象，避免无界内存占用。" * 16,
        ),
    ]

    normal = ingestion.build_llamaindex_text_chunks(
        document_id="document-demo",
        segments=segments,
        chunk_size=96,
        chunk_overlap=12,
    )
    assert normal.framework_config["pipeline"] == "IngestionPipeline"
    assert normal.framework_config["node_parser"] == "SentenceSplitter"
    assert normal.child_chunks
    assert {
        reference.segment_index
        for chunk in normal.child_chunks
        for reference in chunk.source_references
    } == {10, 11}

    hierarchical = ingestion.build_llamaindex_parent_child_chunks(
        document_id="document-demo",
        segments=segments,
        parent_chunk_size=192,
        child_chunk_size=72,
        child_chunk_overlap=12,
    )
    assert hierarchical.framework_config["node_parser"] == "HierarchicalNodeParser"
    assert hierarchical.parent_chunks and hierarchical.child_chunks
    assert all(
        child.parent_index is not None
        and child.parent_index < len(hierarchical.parent_chunks)
        for child in hierarchical.child_chunks
    )

    captured_embedding_inputs: list[str] = []
    captured_records: list[dict] = []

    def fake_embeddings(texts: list[str], batch_size: int = 10):
        captured_embedding_inputs.extend(texts)
        return "test-embedding", [[0.1, 0.2, 0.3] for _ in texts]

    def fake_upsert(*, records: list[dict]) -> None:
        captured_records.extend(records)

    # 把真实网络调用替换为内存桩，仅验证 LlamaIndex Transform/VectorStore 的边界。
    original_embeddings = ingestion.generate_text_embeddings
    original_upsert = ingestion.upsert_chunk_vectors
    ingestion.generate_text_embeddings = fake_embeddings
    ingestion.upsert_chunk_vectors = fake_upsert
    try:
        model, written_ids = ingestion.index_chunks_with_llamaindex(
            document_id="document-demo",
            version_id="version-v2",
            chunks=[_Chunk("chunk-1", "只应进入语义向量的真实文本")],
            embedding_batch_size=2,
        )
    finally:
        ingestion.generate_text_embeddings = original_embeddings
        ingestion.upsert_chunk_vectors = original_upsert

    # 对外版本快照使用服务配置的模型名；底层 fake 只模拟向量数值，不改变运行时配置。
    assert model == ingestion.settings.dashscope_embedding_model
    assert written_ids == ["chunk-1"]
    assert captured_embedding_inputs == ["只应进入语义向量的真实文本"]
    assert captured_records == [
        {
            "chunk_id": "chunk-1",
            "document_id": "document-demo",
            "version_id": "version-v2",
            "embedding": [0.1, 0.2, 0.3],
        }
    ]
    print("DAY31_LLAMA_PRODUCTION_PIPELINE_SMOKE_OK")
    print("ingestion=SentenceSplitter+HierarchicalNodeParser embedding=BaseEmbedding vector_store=project_milvus")


if __name__ == "__main__":
    main()
