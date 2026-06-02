# RAG 核心技能必练 Demo

这个文件夹包含了 RAG 学习中必须动手练习的 5 个核心示例。

## 学习顺序

按照文件编号顺序学习，每个 demo 大约 15-30 分钟：

### 1. Embedding 基础调用 (15 分钟)
**文件**: `1_embedding_basic.py`

**目标**: 理解 Embedding 的输入输出，文本 → 数字向量

**学到什么**:
- 如何调用 Ollama Embedding API
- 向量的维度和格式
- 单个文本和批量文本的向量化

**运行前准备**:
```bash
# 确保已下载 Embedding 模型
ollama pull nomic-embed-text

# 安装依赖
pip install requests
```

**完全免费，无需 API Key！**

---

### 2. 余弦相似度计算 (20 分钟)
**文件**: `2_cosine_similarity.py`

**目标**: 理解相似度计算是 RAG 检索的核心原理

**学到什么**:
- 余弦相似度的计算公式
- 为什么只看方向不看长度
- 相似度范围和含义

**运行前准备**:
```bash
pip install numpy
```

**无需 API Key**，可以直接运行！

---

### 3. 语义搜索 (30 分钟)
**文件**: `3_semantic_search.py`

**目标**: 理解 RAG 的核心 - 如何找到和问题最相关的文档

**学到什么**:
- RAG 检索的完整流程
- 如何从多个文档中找到最相关的
- 向量化 → 计算相似度 → 排序的完整过程

**运行前准备**:
```bash
pip install requests numpy
```

**这是最重要的 demo**，理解这个就理解了 RAG 的核心！

**完全免费，使用 Ollama 本地模型！**

---

### 4. Chroma 向量数据库 (20 分钟)
**文件**: `4_chroma_basic.py`

**目标**: 掌握向量数据库的核心操作

**学到什么**:
- 如何创建向量数据库
- 如何添加文档（自动向量化）
- 如何查询相关文档
- 元数据的作用

**运行前准备**:
```bash
pip install chromadb
```

**无需 API Key**，Chroma 内置了默认的 Embedding 模型！

---

### 5. 文本分块 (30 分钟)
**文件**: `5_text_chunking.py`

**目标**: 理解重叠分块的原理

**学到什么**:
- 为什么要分块
- 重叠分块的原理和优势
- chunk_size 和 overlap 参数的选择
- 实际文档的分块效果

**无需任何依赖**，可以直接运行！

---

## 快速开始

### 1. 安装依赖

```bash
# 安装所有需要的库
pip install requests numpy chromadb
```

### 2. 下载 Ollama Embedding 模型

```bash
# 下载 Embedding 模型（约 274MB）
ollama pull nomic-embed-text

# 验证安装
ollama list
```

### 3. 按顺序运行

```bash
python 1_embedding_basic.py
python 2_cosine_similarity.py
python 3_semantic_search.py
python 4_chroma_basic.py
python 5_text_chunking.py
```

**完全免费，无需任何 API Key！**

---

## 学习建议

### 时间分配
- 总共 2 小时
- 每个 demo 15-30 分钟
- 不要只看代码，一定要运行！

### 学习重点

1. **Demo 1-2**: 理解 Embedding 和相似度的基本概念
2. **Demo 3**: 最重要！理解 RAG 检索的完整流程
3. **Demo 4**: 掌握向量数据库的基本操作
4. **Demo 5**: 理解文本分块的原理

### 学习方法

1. **先运行，再理解**
   - 不要只看代码，一定要运行看结果
   - 观察输出，理解每一步在做什么

2. **修改参数，观察变化**
   - 改变 chunk_size、overlap 等参数
   - 改变查询问题，看检索结果的变化
   - 添加自己的文档，测试效果

3. **理解原理，不要死记代码**
   - 理解为什么要这样做
   - 理解每个参数的作用
   - 实际项目中会用框架，不需要手写

---

## 常见问题

### Q1: 需要 API Key 吗？

**不需要！** 所有 demo 都使用 Ollama 本地模型，完全免费。

### Q2: 运行报错怎么办？

**常见错误**:
1. `ModuleNotFoundError`: 没有安装依赖，运行 `pip install requests numpy chromadb`
2. `Connection refused`: Ollama 没有运行，运行 `ollama serve`
3. `Model not found`: 没有下载模型，运行 `ollama pull nomic-embed-text`

### Q3: 需要全部练习吗？

**必练**:
- Demo 3: 语义搜索（最重要）
- Demo 4: Chroma 基本操作
- Demo 5: 文本分块

**可选**:
- Demo 1: 如果你已经理解 Embedding，可以跳过
- Demo 2: 如果你已经理解相似度，可以跳过

---

## 下一步

完成这 5 个 demo 后，你就掌握了 RAG 的核心技能！

接下来可以：
1. 阅读 `4.RAG/5.RAG完整流程实战.md`
2. 用 LangChain 构建完整的 RAG 系统
3. 用自己的文档构建知识库

---

## 反馈

如果有任何问题或建议，欢迎反馈！

祝学习顺利！🚀
