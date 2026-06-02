"""
Demo 3: 语义搜索（Ollama 本地版本 - 完全免费）
目标：理解 RAG 的核心 - 如何找到和问题最相关的文档


这个 demo 是前两个 demo 的组合应用，把 Demo 1（文本向量化）和 Demo 2（余弦相似度）
串起来，实现了一个完整的语义搜索，也就是 RAG 检索的核心流程。

使用 Ollama 本地模型，完全免费

前置条件：
1. 已安装 Ollama
2. 已下载 Embedding 模型：ollama pull nomic-embed-text
"""

import requests
import numpy as np


#文本向量化
def get_embedding(text, model="nomic-embed-text"):
    """调用 Ollama 本地 Embedding API"""
    url = "http://localhost:11434/api/embeddings"
    
    payload = {
        "model": model,
        "prompt": text
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"API 调用失败: {response.text}")


#计算余弦相似度
def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))




def semantic_search(query, documents, top_k=3):
    """
    语义搜索：找到和查询最相关的文档
    
    Args:
        query: 用户问题
        documents: 文档列表
        top_k: 返回最相关的 k 个文档
    
    Returns:
        List[Tuple]: (文档, 相似度) 列表
    """
    print(f"用户问题: {query}")
    print(f"知识库文档数量: {len(documents)}")
    print("\n开始检索...")
    
    # 1. 向量化所有文档
    print("  [1/4] 向量化文档...")
    doc_vectors = []
    for doc in documents:
        vector = get_embedding(doc)
        doc_vectors.append(vector)
    
    # 2. 向量化查询
    print("  [2/4] 向量化问题...")
    query_vector = get_embedding(query)
    
    # 3. 计算相似度
    print("  [3/4] 计算相似度...")
    similarities = []
    for i, doc_vec in enumerate(doc_vectors):
        sim = cosine_similarity(query_vector, doc_vec)
        similarities.append((i, sim))
    
    # 4. 排序并返回 top_k
    print("  [4/4] 排序并返回结果...")
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for i, sim in similarities[:top_k]:
        results.append((documents[i], sim))
    
    return results


# ============ 主程序 ============

print("=" * 60)
print("语义搜索 Demo - 这就是 RAG 的核心检索流程")
print("使用 Ollama 本地模型，完全免费")
print("=" * 60)

# 知识库文档
documents = [
    "Python 是一门高级编程语言，广泛用于 Web 开发、数据分析、人工智能等领域",
    "FastAPI 是一个现代化的 Python Web 框架，用于构建高性能的 API",
    "机器学习是人工智能的一个分支，让计算机从数据中学习规律",
    "深度学习使用神经网络模型，在图像识别、自然语言处理等领域表现出色",
    "Django 是一个成熟的 Python Web 框架，适合构建大型 Web 应用",
    "NumPy 是 Python 的科学计算库，提供高效的数组操作",
    "Pandas 是数据分析库，提供 DataFrame 等数据结构",
    "TensorFlow 是 Google 开发的深度学习框架"
]

print("\n知识库内容：")
for i, doc in enumerate(documents, 1):
    print(f"  {i}. {doc}")

# 测试不同的查询
queries = [
    "如何学习 Python Web 开发？",
    "什么是人工智能？",
    "Python 有哪些数据分析工具？"
]

for query in queries:
    print("\n" + "=" * 60)
    results = semantic_search(query, documents, top_k=3)
    
    print("\n最相关的文档：")
    for i, (doc, sim) in enumerate(results, 1):
        print(f"\n  Top {i} (相似度: {sim:.3f})")
        print(f"  {doc}")

print("\n" + "=" * 60)
print("✅ 完成！你已经理解了 RAG 的核心检索流程")
print("=" * 60)

print("""
核心流程总结：
1. 用户提问 → 向量化
2. 知识库文档 → 向量化（提前做好）
3. 计算问题向量和每个文档向量的相似度
4. 按相似度排序，返回最相关的文档
5. 把相关文档 + 问题一起发给 AI，生成答案

这就是 RAG 的完整检索流程！

使用 Ollama 的优势：
- 完全免费，无限调用
- 数据不出本地，隐私安全
- 无需 API Key
""")
