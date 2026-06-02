"""
Demo 2: Text Splitters（文本分块）
目标：学会用 LangChain 的 RecursiveCharacterTextSplitter 分块

演示内容：
1. split_text — 分割纯文本字符串
2. split_documents — 分割 Document 对象（保留 metadata）
3. chunk_size 和 chunk_overlap 参数对比

不需要 API Key
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON_INTRO_PATH = BASE_DIR / "sample_docs" / "python_intro.txt"

# ============================================================
# 1. split_text — 分割纯文本
# ============================================================
print("=" * 60)
print("1. split_text — 分割纯文本")
print("=" * 60)

text = """Python 是一门高级编程语言，广泛应用于 Web 开发、数据分析和人工智能等领域。

Python 的优点包括语法简洁、生态丰富、社区活跃。它被称为"胶水语言"，能与其他语言无缝集成。Python 拥有超过 40 万个第三方库，几乎可以做任何事情。

FastAPI 是一个基于 Python 的现代 Web 框架。它支持异步操作，性能接近 Node.js 和 Go。FastAPI 能自动生成 Swagger UI 文档，非常适合 AI 应用后端开发。

Django 是另一个流行的 Python Web 框架。它提供了完整的功能集，包括 ORM、认证系统、Admin 后台等。Django 适合构建大型 Web 应用。"""

# 创建分块器
# chunk_size=100：为了演示效果设小一点（实际项目用 300-800）
# chunk_overlap=20：相邻块重叠 20 字符
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    # separators 默认值就是 ["\n\n", "\n", " ", ""]
    # 中文可以加上句号等分隔符
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
)

chunks = splitter.split_text(text)

print(f"原文长度: {len(text)} 字符")
print(f"分成了 {len(chunks)} 个块\n")

for i, chunk in enumerate(chunks):
    print(f"--- 块 {i+1}（{len(chunk)} 字符）---")
    print(chunk)
    print()

# 观察：
# 1. 每个块大约 100 字符以内
# 2. 尽量在段落边界或句号处分割，不会把句子切断
# 3. 相邻块之间有重叠部分


# ============================================================
# 2. split_documents — 分割 Document 对象
# ============================================================
print("=" * 60)
print("2. split_documents — 分割 Document 对象（保留 metadata）")
print("=" * 60)

# 先加载文档
loader = TextLoader(str(PYTHON_INTRO_PATH), encoding="utf-8")
documents = loader.load()

# 用实际项目的参数分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # 每块最大 300 字符
    chunk_overlap=30,    # 重叠 30 字符
)

# split_documents 方法：分割 Document 对象列表
# 关键：分块后每个 chunk 仍然保留原始的 metadata
chunks = splitter.split_documents(documents)

print(f"原始文档数: {len(documents)}")
print(f"分块后数量: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"--- 块 {i+1}（{len(chunk.page_content)} 字符）---")
    print(f"metadata: {chunk.metadata}")        # metadata 被保留了！
    print(f"内容: {chunk.page_content[:80]}...")
    print()


# ============================================================
# 3. chunk_size 对比实验
# ============================================================
print("=" * 60)
print("3. chunk_size 对比实验")
print("=" * 60)

# 加载同一个文件，用不同 chunk_size 分块
loader = TextLoader(str(PYTHON_INTRO_PATH), encoding="utf-8")
documents = loader.load()

for chunk_size in [200, 500, 1000]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1),  # overlap = 10% of chunk_size
    )
    chunks = splitter.split_documents(documents)
    print(f"chunk_size={chunk_size}, overlap={int(chunk_size * 0.1)} → {len(chunks)} 个块")

# 观察：
# chunk_size 越小 → 块越多 → 检索更精准，但可能丢失上下文
# chunk_size 越大 → 块越少 → 上下文更完整，但检索精度下降


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
1. RecursiveCharacterTextSplitter 是最常用的分块器
2. split_text(text)：分割纯文本字符串
3. split_documents(docs)：分割 Document 对象，保留 metadata
4. 两个核心参数：chunk_size（块大小）、chunk_overlap（重叠大小）
5. 推荐起步配置：chunk_size=500, chunk_overlap=50
6. 实际项目中需要根据效果调整，没有"最优"配置

下一步：用 Embeddings 把文本块向量化 → Demo 3
""")
