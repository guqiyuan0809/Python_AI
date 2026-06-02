"""
Demo 5: Retrieval Chain（检索链）— 完整 RAG 系统
目标：把前 4 步串起来，实现完整的 LangChain RAG 流程

演示内容：
1. 完整 RAG 流程：加载 → 分块 → 向量化 → 存储 → 检索 → 生成
2. create_retrieval_chain 用法
3. 引用溯源（显示回答来自哪个文件）
4. 多轮问答

需要 API Key（调用 LLM 生成回答）

前置条件：
1. 已安装 Ollama 并下载 nomic-embed-text
2. 修改下面的 API_KEY 为你自己的
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import shutil
import os

# ============================================================
# 配置
# ============================================================
API_KEY = "sk-2e56e7e02258424eaf6afe699b054031"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

PERSIST_DIR = "./chroma_db"


# ============================================================
# 第 1 步：构建知识库（加载 → 分块 → 向量化 → 存储）
# ============================================================
def build_knowledge_base():
    """构建知识库（只需要运行一次）"""
    print("=" * 60)
    print("第 1 步：构建知识库")
    print("=" * 60)

    # 加载文档
    loader = DirectoryLoader(
        "./sample_docs",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"  加载了 {len(documents)} 个文档")

    # 分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(documents)
    print(f"  分成了 {len(chunks)} 个块")

    # 向量化 + 存储
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 清理旧数据
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"  知识库构建完成，已保存到 {PERSIST_DIR}/")

    return vectorstore


# ============================================================
# 第 2 步：创建 RAG Chain
# ============================================================
def create_rag_chain(vectorstore):
    """创建 RAG 检索链"""
    print("\n" + "=" * 60)
    print("第 2 步：创建 RAG Chain")
    print("=" * 60)

    # 1. 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. 初始化 LLM
    llm = ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    # 3. 定义 Prompt 模板
    # {context} 会被替换为检索到的文档内容
    # {input} 会被替换为用户问题
    prompt = ChatPromptTemplate.from_template("""你是一个知识库助手。请根据以下文档内容回答用户的问题。

要求：
1. 只基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明"文档中没有找到相关信息"
3. 回答要简洁、有条理

文档内容：
{context}

用户问题：{input}""")

    # 4. 创建 Chain
    # create_stuff_documents_chain：把文档塞入 Prompt 的 {context}
    document_chain = create_stuff_documents_chain(llm, prompt)

    # create_retrieval_chain：串联 retriever + document_chain
    # 完整流程：用户问题 → retriever 检索 → 文档塞入 prompt → LLM 回答
    chain = create_retrieval_chain(retriever, document_chain)

    print("  RAG Chain 创建完成")
    return chain


# ============================================================
# 第 3 步：问答（带引用溯源）
# ============================================================
def ask_with_sources(chain, question):
    """提问并显示引用来源"""
    print(f"\n{'─' * 60}")
    print(f"问题: {question}")
    print(f"{'─' * 60}")

    # 调用 chain，一次完成：检索 → 拼接 → 生成
    result = chain.invoke({"input": question})

    # result 包含两个关键字段：
    # result["answer"]：AI 的回答
    # result["context"]：检索到的文档列表（Document 对象）

    print(f"\n回答:\n{result['answer']}")

    # 引用溯源：显示回答基于哪些文档
    print(f"\n📎 引用来源:")
    seen_sources = set()  # 去重
    for doc in result["context"]:
        source = doc.metadata.get("source", "未知")
        filename = source.split("/")[-1].split("\\")[-1]
        if filename not in seen_sources:
            seen_sources.add(filename)
            print(f"  - {filename}")
            print(f"    \"{doc.page_content[:60]}...\"")

    return result


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":

    # 第 1 步：构建知识库
    vectorstore = build_knowledge_base()

    # 第 2 步：创建 RAG Chain
    chain = create_rag_chain(vectorstore)

    # 第 3 步：提问
    print("\n" + "=" * 60)
    print("第 3 步：问答测试")
    print("=" * 60)

    # 测试问题 1：应该能从 python_intro.txt 找到答案
    ask_with_sources(chain, "Python 有哪些优点？")

    # 测试问题 2：应该能从 fastapi_guide.txt 找到答案
    ask_with_sources(chain, "FastAPI 和 Django 有什么区别？")

    # 测试问题 3：应该能从 rag_overview.txt 找到答案
    ask_with_sources(chain, "RAG 解决了什么问题？有哪些典型应用场景？")

    # 测试问题 4：文档中没有的信息
    ask_with_sources(chain, "如何用 Java 开发 Android 应用？")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("完成！这就是完整的 LangChain RAG 系统")
    print("=" * 60)
    print("""
完整流程回顾：

1. Document Loaders  → 加载 3 个 txt 文件
2. Text Splitters    → 分成多个 300 字符的块
3. Embeddings        → Ollama 本地模型向量化（自动完成）
4. Vector Stores     → 存入 Chroma 数据库（自动完成）
5. Retrieval Chain   → 检索 + LLM 生成回答

引用溯源原理：
  metadata 从加载到输出一路传递
  Document Loaders 自动记录 source（文件名）
  → Text Splitters 保留 metadata
  → Vector Stores 存储 metadata
  → Retrieval Chain 返回 metadata
  → 展示给用户

接下来做 RAG 项目时，你需要：
- 把 sample_docs 换成你自己的文档
- 加上 Web 界面（Streamlit / Gradio / Vue）
- 加上更多文件格式支持（PDF、Word）
- 加上性能优化（混合检索、Rerank）
""")
