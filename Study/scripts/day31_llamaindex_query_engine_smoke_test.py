"""Day31：验证 LlamaIndex QueryEngine 的编排边界，不访问数据库、Milvus 或真实模型。"""

from pathlib import Path
import sys

from pydantic import PrivateAttr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse, CustomLLM, LLMMetadata, MessageRole
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from day04_app.services.llamaindex_rag_query_service import GovernedRagNodePostprocessor


class DemoRetriever(BaseRetriever):
    """模拟项目的 ExistingKnowledgeMilvusRetriever 已返回的标准框架节点。"""

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        del query_bundle
        return [
            NodeWithScore(
                node=TextNode(
                    text="JVM 是 Java 程序的运行环境。",
                    id_="chunk-jvm-001",
                    metadata={
                        "document_id": "document-demo",
                        "version_id": "version-v1",
                        "chunk_id": "chunk-jvm-001",
                        "chunk_index": 0,
                        "parent_chunk_id": None,
                        "source_locations": ["Paragraph:1"],
                    },
                ),
                score=0.9,
            )
        ]


class DemoLlamaIndexLLM(CustomLLM):
    """不调用网络，只记录 QueryEngine 最终交给模型的消息。"""

    runtime_model: str = "demo-model"
    _received_messages: list[ChatMessage] = PrivateAttr(default_factory=list)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=256,
            is_chat_model=True,
            model_name=self.runtime_model,
        )

    @llm_chat_callback()
    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        del kwargs
        self._received_messages = messages
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="JVM 是 Java 程序的运行环境。[S1]",
            )
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        del formatted, kwargs
        return CompletionResponse(text=self.chat([ChatMessage(content=prompt)]).message.content or "")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs):
        completion = self.complete(prompt, formatted=formatted, **kwargs)
        yield CompletionResponse(text=completion.text, delta=completion.text)


def main() -> None:
    postprocessor = GovernedRagNodePostprocessor(max_context_characters=500)
    llm = DemoLlamaIndexLLM()
    prompt = ChatPromptTemplate(
        message_templates=[
            ChatMessage(role=MessageRole.SYSTEM, content="只能依据资料回答，每句需要引用来源。"),
            ChatMessage(role=MessageRole.USER, content="资料：\n{context_str}\n问题：{query_str}"),
        ]
    )
    query_engine = RetrieverQueryEngine.from_args(
        retriever=DemoRetriever(),
        llm=llm,
        text_qa_template=prompt,
        response_mode="compact",
        node_postprocessors=[postprocessor],
    )

    response = query_engine.query("JVM 是什么？")
    assert "[S1]" in str(response)
    assert response.source_nodes[0].node.metadata["citation"] == "[S1]"
    assert "JVM 是 Java 程序的运行环境。" in (llm._received_messages[1].content or "")
    assert "[S1]" in (llm._received_messages[1].content or "")
    print("DAY31_LLAMA_QUERY_ENGINE_SMOKE_OK")
    print("retriever=custom_milvus_adapter postprocessor=governed_context query_engine=compact")


if __name__ == "__main__":
    main()
