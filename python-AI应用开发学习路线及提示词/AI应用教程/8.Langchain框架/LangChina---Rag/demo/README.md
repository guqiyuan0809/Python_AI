# LangChain RAG Demo

## 安装依赖

```bash
pip install langchain langchain-core langchain-text-splitters langchain-ollama langchain-openai langchain-chroma pypdf numpy
```

确保 Ollama 已安装并下载 Embedding 模型：
```bash
ollama pull nomic-embed-text
```

## 学习顺序

### Demo 1: Document Loaders（文档加载）
**文件**: `1_document_loaders.py`

用 LangChain 加载文本文件和整个文件夹，理解 Document 对象的结构（page_content + metadata）。

**不需要 API Key**

---

### Demo 2: Text Splitters（文本分块）
**文件**: `2_text_splitters.py`

用 RecursiveCharacterTextSplitter 对文档分块，对比不同 chunk_size 的效果，理解 split_documents 如何保留 metadata。

**不需要 API Key**

---

### Demo 3: Embeddings（向量化）
**文件**: `3_embeddings.py`

用 Ollama 本地模型做文本向量化，理解 embed_query 和 embed_documents 的区别，手动计算相似度验证效果。

**不需要 API Key**（使用 Ollama 本地模型）

---

### Demo 4: Vector Stores（向量数据库）
**文件**: `4_vector_stores.py`

用 Chroma 存储向量和搜索相似文档，学习 from_documents（自动向量化+存储）、similarity_search（搜索）、persist_directory（持久化）、as_retriever（转为检索器）。

**不需要 API Key**（使用 Ollama 本地模型）

---

### Demo 5: Retrieval Chain（完整 RAG 系统）
**文件**: `5_retrieval_chain.py`

把前 4 步串起来，实现完整的 LangChain RAG 流程。包含引用溯源功能——回答时显示信息来自哪个文件。

**需要 API Key**（调用 DeepSeek 生成回答）

---

## 示例文档

`sample_docs/` 文件夹包含 3 个示例文档，用于 Demo 测试：
- `python_intro.txt` — Python 编程语言简介
- `fastapi_guide.txt` — FastAPI 入门指南
- `rag_overview.txt` — RAG 技术概述

做自己的 RAG 项目时，把这个文件夹替换成你自己的文档。

---

## 对应理论文档

| Demo | 对应理论文档 |
|------|------------|
| Demo 1: Document Loaders | 1.Document Loaders 文档加载.md |
| Demo 2: Text Splitters | 2.Text Splitters 文本分块.md |
| Demo 3: Embeddings | 3.Embeddings 向量化.md |
| Demo 4: Vector Stores | 4.Vector Stores 向量数据库.md |
| Demo 5: Retrieval Chain | 5.Retrieval Chain 检索链.md |

---

## 核心要点

- LangChain 把 RAG 流程标准化为 5 步：加载 → 分块 → 向量化 → 存储 → 检索生成
- 所有步骤围绕 Document 对象（page_content + metadata）
- metadata 从加载到最终输出一路传递，这是引用溯源的基础
- Embedding 和 Vector Store 在实际使用中几乎合为一步（from_documents）
- 学完这 5 个 Demo，就可以直接去做 RAG 项目了
