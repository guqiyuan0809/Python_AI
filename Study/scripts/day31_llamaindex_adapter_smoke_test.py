"""Day31：不访问模型、Milvus 或数据库的 LlamaIndex 节点转换冒烟测试。"""

from pathlib import Path
import sys

# 允许从 Study 根目录直接执行 ``python scripts\\...``，与课程现有脚本保持一致。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.schemas.knowledge_schema import ChunkSourceReference, MilvusChunkSearchItem
from day04_app.services.llamaindex_law_retrieval_service import to_llamaindex_node


def main() -> None:
    item = MilvusChunkSearchItem(
        document_id="law-document-demo",
        version_id="law-version-v1",
        chunk_id="law-chunk-001",
        chunk_index=3,
        parent_chunk_id=None,
        score=0.91,
        vector_score=0.88,
        rerank_score=0.91,
        content="动火作业应当设置监护人，监护人不得擅自离岗。",
        source_references=[
            ChunkSourceReference(segment_index=12, location="第 8 页 / 第 3 条")
        ],
    )

    node_with_score = to_llamaindex_node(item)
    assert node_with_score.node.node_id == item.chunk_id
    assert node_with_score.node.get_content() == item.content
    assert node_with_score.node.metadata["version_id"] == item.version_id
    assert node_with_score.node.metadata["source_locations"] == ["第 8 页 / 第 3 条"]
    assert node_with_score.score == item.score
    print("DAY31_LLAMAINDEX_ADAPTER_SMOKE_OK")
    print("node_id=law-chunk-001 score=0.91 source=第 8 页 / 第 3 条")


if __name__ == "__main__":
    main()
