"""
Demo 4: Chroma 向量数据库基本操作
目标：掌握向量数据库的核心操作（add、query）
"""

import chromadb

print("=" * 60)
print("Chroma 向量数据库基本操作")
print("=" * 60)

# 1. 创建客户端（数据保存到磁盘）
print("\n[1/5] 创建持久化客户端...")

#PersistentClient 表示数据存到磁盘，程序关了数据还在。路径 ./chroma_db 就是数据存放的文件夹
client = chromadb.PersistentClient(path="./chroma_db")
print("✓ 客户端创建成功，数据将保存到 ./chroma_db 目录")

# 2. 创建或获取集合（相当于 MySQL 的"创建表"。）
print("\n[2/5] 创建集合...")
collection = client.get_or_create_collection(
    name="my_knowledge_base",
    metadata={"description": "我的知识库"}
)
print(f"✓ 集合创建成功: {collection.name}")

# 3. 添加文档
print("\n[3/5] 添加文档...")
documents = [
    "Python 是一门编程语言",
    "FastAPI 是一个 Web 框架",
    "机器学习是 AI 的分支",
    "深度学习使用神经网络",
    "Django 是 Python Web 框架"
]

# 添加文档（Chroma 会自动向量化,相当于 MySQL 的 INSERT INTO）
collection.add(
    documents=documents,
    ids=[f"doc{i}" for i in range(len(documents))],
    metadatas=[
        {"source": "book1", "page": 1, "category": "编程"},
        {"source": "book2", "page": 5, "category": "Web"},
        {"source": "book3", "page": 10, "category": "AI"},
        {"source": "book3", "page": 15, "category": "AI"},
        {"source": "book2", "page": 20, "category": "Web"}
    ]
)
print(f"✓ 成功添加 {len(documents)} 个文档")

# 4. 查询
print("\n[4/5] 查询相关文档...")
query = "如何学习 Web 开发？"
print(f"问题: {query}")

results = collection.query(
    query_texts=[query],
    n_results=3  # 返回最相关的 3 个
)

print("\n最相关的文档：")
for i in range(len(results['documents'][0])):
    doc = results['documents'][0][i]
    distance = results['distances'][0][i]
    metadata = results['metadatas'][0][i]
    
    print(f"\nTop {i+1}:")
    print(f"  文档: {doc}")
    print(f"  距离: {distance:.3f} (越小越相似)")
    print(f"  元数据: {metadata}")

# 5. 查看集合统计(相当于 SELECT COUNT(*) FROM table，看数据库里有多少条数据)
print("\n[5/5] 集合统计...")
count = collection.count()
print(f"✓ 集合中共有 {count} 个文档")

print("\n" + "=" * 60)
print("✅ 完成！你已经掌握了 Chroma 的基本操作")
print("=" * 60)

print("""
核心操作总结：
1. 创建客户端: chromadb.PersistentClient(path="./chroma_db")
2. 创建集合: client.get_or_create_collection(name="...")
3. 添加文档: collection.add(documents=..., ids=..., metadatas=...)
4. 查询: collection.query(query_texts=..., n_results=...)
5. 统计: collection.count()

注意：
- Chroma 会自动向量化文档，不需要手动调用 Embedding API
- 距离越小表示越相似（L2 距离）
- 元数据可以用来过滤和追溯来源
""")
