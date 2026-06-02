"""
Demo 4: Vector Stores（向量数据库）
目标：学会用 LangChain 操作 Chroma 向量数据库

演示内容：
1. from_documents — 从文档创建向量库（自动向量化 + 存储）
2. similarity_search — 相似度搜索
3. 持久化存储（persist_directory）
4. as_retriever — 转为检索器

不需要 API Key（使用 Ollama 本地模型）

前置条件：
1. 已安装 Ollama 并下载 nomic-embed-text
2. pip install langchain-chroma langchain-ollama langchain-text-splitters
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import shutil
import os

# ============================================================
# 准备工作：加载文档 + 分块
# ============================================================
print("=" * 60)
print("准备工作：加载文档 + 分块")
print("=" * 60)

# 第 1 步：加载文档（Demo 1 学过的）
loader = DirectoryLoader(
    "./sample_docs",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
print(f"加载了 {len(documents)} 个文档")

# 第 2 步：分块（Demo 2 学过的）
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(documents)
print(f"分成了 {len(chunks)} 个块")

# 第 3 步：初始化 Embedding 模型（Demo 3 学过的）
embeddings = OllamaEmbeddings(model="nomic-embed-text")
print("Embedding 模型已初始化")


# ============================================================
# 1. from_documents — 创建向量库
# ============================================================
print("\n" + "=" * 60)
print("1. from_documents — 创建向量库")
print("=" * 60)

# 这一行代码做了三件事：
# (1) 对每个 chunk 的 page_content 调用 embeddings 向量化
# (2) 把向量 + 原文 + metadata 存入 Chroma
# (3) 返回一个可搜索的 vectorstore 对象
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

print(f"向量数据库创建完成，包含 {len(chunks)} 个文档块")


# ============================================================
# 2. similarity_search — 相似度搜索
# ============================================================
print("\n" + "=" * 60)
print("2. similarity_search — 相似度搜索")
print("=" * 60)

# 搜索与问题最相关的 3 个文档块
# 内部自动对查询做向量化，不需要手动调用 embed_query
query = "FastAPI 和 Django 有什么区别？"
results = vectorstore.similarity_search(query, k=3)

print(f"查询: {query}")
print(f"找到 {len(results)} 个相关文档块\n")

for i, doc in enumerate(results):
    source = doc.metadata.get("source", "未知")
    filename = source.split("/")[-1].split("\\")[-1]
    print(f"--- 结果 {i+1} ---")
    print(f"来源: {filename}")
    print(f"内容: {doc.page_content[:120]}...")
    print()


# ============================================================
# 3. similarity_search_with_score — 带分数的搜索
# ============================================================
print("=" * 60)
print("3. similarity_search_with_score — 带分数的搜索")
print("=" * 60)

query = "RAG 解决了什么问题？"
results = vectorstore.similarity_search_with_score(query, k=3)

print(f"查询: {query}\n")

for doc, score in results:
    filename = doc.metadata.get("source", "未知").split("/")[-1].split("\\")[-1]
    # Chroma 返回的是距离（越小越相似），不是余弦相似度（越大越相似）
    print(f"  距离: {score:.4f} | 来源: {filename}")
    print(f"  内容: {doc.page_content[:80]}...")
    print()


# ============================================================
# 4. 持久化存储
# ============================================================
print("=" * 60)
print("4. 持久化存储")
print("=" * 60)

persist_dir = "./chroma_db"

# 清理旧数据（仅 Demo 用，实际项目不需要）
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

# 创建持久化的向量库
vectorstore_persistent = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir     # 数据保存到这个文件夹
)
print(f"数据已保存到 {persist_dir}/")

# 模拟重新启动程序：从磁盘加载
vectorstore_loaded = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings     # 注意：加载时参数名是 embedding_function
)

# 加载后直接搜索，数据还在
results = vectorstore_loaded.similarity_search("Python Web 框架", k=2)
print(f"\n从磁盘加载后搜索，找到 {len(results)} 个结果:")
for doc in results:
    print(f"  {doc.page_content[:80]}...")


# ============================================================
# 5. as_retriever — 转为检索器
# ============================================================
print("\n" + "=" * 60)
print("5. as_retriever — 转为检索器")
print("=" * 60)

# as_retriever() 把 vectorstore 转成检索器对象
# 后面接入 Retrieval Chain 时要用
retriever = vectorstore_loaded.as_retriever(
    search_type="similarity",        # 搜索类型
    search_kwargs={"k": 3}           # 返回 3 个结果
)

# retriever 的调用方式
docs = retriever.invoke("什么是 RAG？")

print(f"retriever 返回 {len(docs)} 个文档\n")
for doc in docs:
    print(f"  {doc.page_content[:80]}...")

print("\nretriever 就是 Retrieval Chain 的输入，下一个 Demo 会用到它")


# ============================================================
# 清理（Demo 用，实际项目不需要）
# ============================================================
# 注释掉这行可以保留数据，下次运行 Demo 5 时直接加载
# shutil.rmtree(persist_dir)


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
1. Chroma.from_documents(chunks, embeddings)：一行代码完成向量化 + 存储
2. similarity_search("问题", k=3)：搜索最相关的文档块
3. persist_directory：持久化到磁盘，下次启动不用重新向量化
4. as_retriever()：转为检索器，接入 Retrieval Chain

整个 RAG 流程到这里已经完成了知识库的构建：
  加载文档 → 分块 → 向量化 + 存入数据库 → 搜索

下一步：接入 LLM，让 AI 基于检索结果回答问题 → Demo 5
""")
