# Chroma 简介

## Chroma 是什么

Chroma 是一个轻量级的向量数据库，专门用来存储和搜索向量。

你可以把它理解为"AI 专用的数据库"：
- MySQL 存表格数据，用 SQL 查询
- Chroma 存向量数据，用相似度搜索

---

## 为什么选 Chroma

| 特点 | 说明 |
|------|------|
| 简单 | `pip install chromadb` 装完就能用，不需要额外装服务 |
| 内置 Embedding | 传文本进去自动向量化，不需要调外部 API |
| 持久化存储 | 数据存到磁盘，程序关了数据还在 |
| 免费开源 | 完全免费，学习和小项目都够用 |

---

## 安装

```bash
pip install chromadb
```

第一次运行时会自动下载内置的 Embedding 模型（`all-MiniLM-L6-v2`，约 80MB），之后直接使用。

---

## 核心操作

就三步：创建 → 存数据 → 查数据。

```python
import chromadb

# 1. 创建客户端（数据保存到磁盘）
client = chromadb.PersistentClient(path="./chroma_db")

# 2. 创建集合（类似 MySQL 的表）
collection = client.get_or_create_collection(name="my_docs")

# 3. 添加文档（Chroma 自动向量化，不需要手动调 Embedding API）
collection.add(
    documents=["Python 是一门编程语言", "FastAPI 是一个 Web 框架"],
    ids=["doc1", "doc2"],
    metadatas=[{"source": "book1"}, {"source": "book2"}]
)

# 4. 查询（传文本进去，Chroma 自动向量化再搜索）
results = collection.query(
    query_texts=["如何学习 Web 开发？"],
    n_results=2
)

print(results['documents'])   # 最相关的文档
print(results['distances'])   # 距离（越小越相似）
print(results['metadatas'])   # 元数据（来源信息）
```

---

## 和 MySQL 类比

| 操作 | MySQL | Chroma |
|------|-------|--------|
| 创建数据库 | `CREATE DATABASE` | `PersistentClient(path="./chroma_db")` |
| 创建表 | `CREATE TABLE` | `get_or_create_collection(name="...")` |
| 插入数据 | `INSERT INTO` | `collection.add(documents=..., ids=...)` |
| 查询数据 | `SELECT ... WHERE` | `collection.query(query_texts=..., n_results=...)` |
| 删除数据 | `DELETE FROM` | `collection.delete(ids=...)` |
| 统计数量 | `SELECT COUNT(*)` | `collection.count()` |

---

## 在 RAG 中的角色

```
用户提问
  ↓
Chroma 自动向量化问题，搜索最相似的文档
  ↓
把相关文档 + 问题一起发给 LLM
  ↓
LLM 基于文档生成回答
```

Chroma 负责的就是中间"找相关文档"这一步。

---

## 注意事项

- 内置的默认 Embedding 模型偏英文，中文效果一般，学习阶段够用
- 适合小型项目（百万级数据以内）
- 生产环境数据量大的话，考虑换 Milvus 或 Qdrant

---

## 官方文档

https://docs.trychroma.com
