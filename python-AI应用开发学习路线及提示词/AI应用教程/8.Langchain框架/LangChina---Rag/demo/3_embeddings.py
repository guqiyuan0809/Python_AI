"""
Demo 3: Embeddings（向量化）
目标：学会用 LangChain 的 Embedding 接口做文本向量化

演示内容：
1. OllamaEmbeddings 基本用法
2. embed_query 和 embed_documents 的区别
3. 手动计算相似度（验证 Embedding 效果）

不需要 API Key（使用 Ollama 本地模型）

前置条件：
1. 已安装 Ollama
2. 已下载 Embedding 模型：ollama pull nomic-embed-text
"""

import numpy as np
from langchain_ollama import OllamaEmbeddings

# ============================================================
# 配置
# ============================================================
# 使用 Ollama 本地 Embedding 模型（免费、无限调用）
# 如果没有下载，先运行：ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# ============================================================
# 1. embed_query — 向量化单个文本
# ============================================================
print("=" * 60)
print("1. embed_query — 向量化单个文本")
print("=" * 60)

# embed_query：把一个文本转成向量
# 用途：用户提问时，把问题向量化
vector = embeddings.embed_query("Python 是什么编程语言？")

print(f"向量维度: {len(vector)}")       # 768（nomic-embed-text 的维度）
print(f"向量前 5 个值: {vector[:5]}")    # 一堆浮点数
print(f"向量类型: {type(vector)}")       # list


# ============================================================
# 2. embed_documents — 批量向量化
# ============================================================
print("\n" + "=" * 60)
print("2. embed_documents — 批量向量化")
print("=" * 60)

# embed_documents：一次性向量化多个文本
# 用途：构建知识库时，把所有文档块批量向量化
texts = [
    "Python 是一门高级编程语言",
    "FastAPI 是一个 Web 框架",
    "机器学习是人工智能的一个分支",
    "今天天气很好",
]

vectors = embeddings.embed_documents(texts)

print(f"输入了 {len(texts)} 个文本")
print(f"得到了 {len(vectors)} 个向量")
print(f"每个向量的维度: {len(vectors[0])}")


# ============================================================
# 3. 手动计算相似度（验证 Embedding 效果）
# ============================================================
print("\n" + "=" * 60)
print("3. 相似度计算（验证效果）")
print("=" * 60)

# 这一步你在 RAG 阶段已经学过
# 这里用 LangChain 的 Embedding 接口再做一次，验证本地模型的效果

def cosine_similarity(vec1, vec2):
    """余弦相似度"""
    # 余弦相似度公式：cos(theta) = (A·B) / (||A|| * ||B||)
    # A·B（点积）衡量方向重合程度；||A||、||B|| 是向量长度，用于归一化
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 用户提问
query = "如何学习 Python Web 开发？"
query_vector = embeddings.embed_query(query)

print(f"查询: {query}\n")

# 计算查询与每个文档的余弦相似度：分数越高，语义越接近
for i, text in enumerate(texts):
    sim = cosine_similarity(query_vector, vectors[i])
    print(f"  相似度 {sim:.4f} → {text}")

# 观察：与 "Python" 和 "Web 框架" 相关的文本相似度更高
# 与 "天气" 无关的文本相似度最低


# ============================================================
# 4. 实际项目中你不需要手动调用 Embedding
# ============================================================
print("\n" + "=" * 60)
print("4. 实际项目中的用法（预告）")
print("=" * 60)

print("""
实际项目中，你几乎不需要手动调用 embed_query 或 embed_documents。

向量数据库会自动帮你调用：

  # 存入时自动向量化
  vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
                                                        ^^^^^^^^^^^^^^^^
                                                        传入 embeddings 对象即可

  # 检索时自动向量化
  results = vectorstore.similarity_search("问题", k=3)

所以 Embedding 这一层，你需要做的就是：
1. 选一个模型（学习用 Ollama 本地模型，生产用云端 API）
2. 初始化后传给 Vector Store
3. 后续自动调用，不用管
""")


# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("总结")
print("=" * 60)
print("""
1. embed_query("文本")：向量化单个查询
2. embed_documents(["文本1", "文本2"])：批量向量化多个文档
3. Ollama 本地模型免费无限调用，学习阶段首选
4. 实际项目中 Embedding 是自动完成的，不需要手动调用
5. Embedding 原理（余弦相似度、语义搜索）你在 RAG 阶段已经学透了

下一步：用 Vector Store 存储和搜索向量 → Demo 4
""")
