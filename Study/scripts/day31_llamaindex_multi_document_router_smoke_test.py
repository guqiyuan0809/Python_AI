"""Day31：多文档知识域路由冒烟测试，不访问 MySQL、Milvus 或真实模型。

验证的不是向量质量，而是框架编排边界：

1. ``RouterRetriever.retrieve`` 真正调用 Selector 并选择一个已授权领域；
2. 领域内允许有多篇文档，返回节点仍保留各自 document/version 元数据；
3. 无关键词时只回退到显式指定的综合领域，绝不扩大为未授权文档；
4. 既有 NodePostprocessor 仍为跨文档证据统一生成 [S1]/[S2] 引用。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.core.retrievers import BaseRetriever, RouterRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools import RetrieverTool
from llama_index.core.llms import MockLLM

from day04_app.services.llamaindex_multi_document_rag_service import (
    KeywordKnowledgeDomainSelector,
    KnowledgeDomain,
    MultiDocumentRouter,
    route_and_retrieve_multi_document_context,
)
from day04_app.services.llamaindex_rag_query_service import GovernedRagNodePostprocessor


class FakeDomainRetriever(BaseRetriever):
    """模拟每个领域内部已经完成一次全局排序的项目 Milvus Retriever。"""

    def __init__(self, nodes: list[NodeWithScore]) -> None:
        super().__init__()
        self.nodes = nodes
        self.call_count = 0
        self.last_result = None

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        del query_bundle
        self.call_count += 1
        return list(self.nodes)


def _node(
    *,
    document_id: str,
    version_id: str,
    chunk_id: str,
    score: float,
    content: str,
) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text=content,
            id_=chunk_id,
            metadata={
                "document_id": document_id,
                "version_id": version_id,
                "chunk_id": chunk_id,
                "chunk_index": 0,
                "parent_chunk_id": None,
                "source_locations": [f"{document_id}:Paragraph:1"],
            },
        ),
        score=score,
    )


def _build_router() -> tuple[MultiDocumentRouter, FakeDomainRetriever, FakeDomainRetriever]:
    compliance_retriever = FakeDomainRetriever(
        [
            _node(
                document_id="law-fire",
                version_id="law-fire-v2",
                chunk_id="law-fire-chunk-1",
                score=0.95,
                content="动火作业必须设置监护人。",
            ),
            _node(
                document_id="rule-fire",
                version_id="rule-fire-v4",
                chunk_id="rule-fire-chunk-3",
                score=0.90,
                content="企业动火作业前应办理审批并落实现场监护。",
            ),
        ]
    )
    emergency_retriever = FakeDomainRetriever(
        [
            _node(
                document_id="plan-emergency",
                version_id="plan-emergency-v1",
                chunk_id="plan-emergency-chunk-2",
                score=0.88,
                content="发生火情后应立即报警并启动应急预案。",
            )
        ]
    )
    domains = (
        KnowledgeDomain(
            domain_id="safety_compliance",
            description="安全法规和企业制度",
            document_ids=("law-fire", "rule-fire"),
        ),
        KnowledgeDomain(
            domain_id="emergency_plan",
            description="应急预案",
            document_ids=("plan-emergency",),
        ),
    )
    selector = KeywordKnowledgeDomainSelector(
        domain_keywords={
            "safety_compliance": ("动火", "审批", "法规"),
            "emergency_plan": ("应急", "火情", "报警"),
        },
        default_index=0,
    )
    router_retriever = RouterRetriever(
        selector=selector,
        retriever_tools=[
            RetrieverTool.from_defaults(
                retriever=compliance_retriever,
                name="safety_compliance",
                description="安全法规和企业制度",
            ),
            RetrieverTool.from_defaults(
                retriever=emergency_retriever,
                name="emergency_plan",
                description="应急预案",
            ),
        ],
        # RouterRetriever 构造时会读取 Settings.llm；传入 MockLLM 可保证该纯路由
        # 测试不因项目未安装 OpenAI 适配器而触发任何外部依赖或模型调用。
        llm=MockLLM(),
    )
    # 类型标注在正式服务中是 MultiDocumentMilvusRetriever；此处只验证框架分派，无需 DB。
    router = MultiDocumentRouter(  # type: ignore[arg-type]
        router_retriever=router_retriever,
        selector=selector,
        domains=domains,
        child_retrievers=(compliance_retriever, emergency_retriever),
    )
    return router, compliance_retriever, emergency_retriever


def main() -> None:
    router, compliance_retriever, emergency_retriever = _build_router()
    route, nodes = route_and_retrieve_multi_document_context(
        router=router,
        question="动火作业审批和现场监护有什么要求？",
    )
    assert route.selected_domain_id == "safety_compliance"
    assert route.route_reason == "deterministic_domain_keyword_match"
    assert route.selected_document_ids == ["law-fire", "rule-fire"]
    assert compliance_retriever.call_count == 1
    assert emergency_retriever.call_count == 0
    assert {node.node.metadata["document_id"] for node in nodes} == {"law-fire", "rule-fire"}

    selected_nodes = GovernedRagNodePostprocessor(max_context_characters=500).postprocess_nodes(nodes)
    assert [node.node.metadata["citation"] for node in selected_nodes] == ["[S1]", "[S2]"]
    assert selected_nodes[0].node.metadata["version_id"] == "law-fire-v2"
    assert selected_nodes[1].node.metadata["version_id"] == "rule-fire-v4"

    default_router, default_compliance, default_emergency = _build_router()
    default_route, _ = route_and_retrieve_multi_document_context(
        router=default_router,
        question="请概括这些资料的共同要求。",
    )
    assert default_route.selected_domain_id == "safety_compliance"
    assert default_route.route_reason == "deterministic_default_domain"
    assert default_compliance.call_count == 1
    assert default_emergency.call_count == 0

    print("DAY31_LLAMA_MULTI_DOCUMENT_ROUTER_SMOKE_OK")
    print("route=safety_compliance documents=2 global_nodes=2 citations=[S1],[S2] default=authorized_domain")


if __name__ == "__main__":
    main()
