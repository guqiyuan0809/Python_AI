"""Day31：验证 LlamaIndex SentenceSplitter 的文档节点和来源元数据。"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.schemas.knowledge_schema import ParsedDocumentSegment
from day04_app.services.llamaindex_document_service import build_llamaindex_nodes_from_segments


def main() -> None:
    result = build_llamaindex_nodes_from_segments(
        document_id="law-document-demo",
        segments=[
            ParsedDocumentSegment(
                segment_index=12,
                text=(
                    "动火作业应当设置监护人。监护人不得擅自离岗。"
                    "作业前应当清理周边可燃物并确认消防器材有效。"
                ),
                location="第 8 页 / 第 3 条",
            ),
            ParsedDocumentSegment(
                segment_index=13,
                text="特殊作业票证应当由授权人员审批。",
                location="第 8 页 / 第 4 条",
            ),
        ],
        chunk_size=128,
        chunk_overlap=16,
    )
    assert result.source_segment_count == 2
    assert result.nodes
    assert all(node.metadata["document_id"] == "law-document-demo" for node in result.nodes)
    assert {node.metadata["segment_index"] for node in result.nodes} == {"12", "13"}
    assert all(node.get_content().strip() for node in result.nodes)
    print("DAY31_LLAMAINDEX_CHUNK_SMOKE_OK")
    print(f"source_segments={result.source_segment_count} nodes={len(result.nodes)} persisted=false")


if __name__ == "__main__":
    main()
