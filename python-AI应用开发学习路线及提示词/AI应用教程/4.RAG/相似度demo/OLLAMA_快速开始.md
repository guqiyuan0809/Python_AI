# 使用 Ollama 本地模型学习 RAG（完全免费）

你已经有 Ollama 和千问模型了，太好了！但是学习 RAG 还需要一个 Embedding 模型。

## 为什么需要 Embedding 模型？

- **千问 2.5:3b** - 聊天模型，用于生成回答
- **Embedding 模型** - 向量化模型，用于文本相似度计算（RAG 检索的核心）

两者作用不同，都需要！

---

## 第一步：下载 Embedding 模型

在你的终端运行：

```bash
# 推荐：nomic-embed-text（英文为主，约 274MB，速度快）
ollama pull nomic-embed-text

# 或者：mxbai-embed-large（多语言支持更好，约 669MB）
ollama pull mxbai-embed-large
```

下载完成后，验证一下：

```bash
ollama list
```

你应该能看到：
```
NAME                    ID              SIZE
qwen2.5:3b             xxx             2.0 GB
nomic-embed-text       xxx             274 MB
```

---

## 第二步：测试 Embedding 模型

运行测试命令：

```bash
# 测试 Embedding 模型
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "你好，世界"
}'
```

如果返回一大串数字（向量），说明成功了！

---

## 第三步：运行 Demo

现在你可以运行 Ollama 版本的 demo 了：

```bash
cd 4.RAG/相似度demo

# Demo 1: Embedding 基础（Ollama 版本）
python 1_embedding_basic_ollama.py

# Demo 2: 余弦相似度（不需要模型）
python 2_cosine_similarity.py

# Demo 3: 语义搜索（Ollama 版本）
python 3_semantic_search_ollama.py

# Demo 4: Chroma 向量数据库（不需要额外模型）
python 4_chroma_basic.py

# Demo 5: 文本分块（不需要模型）
python 5_text_chunking.py
```

---

## 完整的 RAG 流程（使用 Ollama）

### 1. Embedding 模型：文本向量化
```bash
ollama pull nomic-embed-text
```

用于：
- 把文档转成向量
- 把用户问题转成向量
- 计算相似度，找到相关文档

### 2. 聊天模型：生成回答
```bash
ollama pull qwen2.5:3b  # 你已经有了
```

用于：
- 基于检索到的文档生成回答

### 完整流程示意

```
用户问题："如何学习 Python？"
    ↓
[Embedding 模型] 向量化问题
    ↓
在向量数据库中搜索相似文档
    ↓
找到相关文档："Python 是一门编程语言..."
    ↓
[聊天模型] 基于文档生成回答
    ↓
返回："学习 Python 可以从基础语法开始..."
```

---

## 推荐的 Embedding 模型对比

| 模型 | 大小 | 语言支持 | 向量维度 | 推荐场景 |
|------|------|---------|---------|---------|
| nomic-embed-text | 274MB | 英文为主 | 768 | 学习、英文文档 |
| mxbai-embed-large | 669MB | 多语言 | 1024 | 中文文档、生产环境 |
| all-minilm | 46MB | 英文 | 384 | 快速测试 |

**学习阶段推荐**：`nomic-embed-text`（小巧快速）

---

## 常见问题

### Q1: 为什么不能用千问模型做 Embedding？

千问是聊天模型，专门训练用来生成文本的。Embedding 模型是专门训练用来把文本转成向量的，两者的训练目标和输出格式完全不同。

### Q2: Ollama Embedding 模型和 OpenAI 的有什么区别？

| 特性 | Ollama 本地 | OpenAI API |
|------|------------|-----------|
| 费用 | 完全免费 | 付费（但很便宜）|
| 速度 | 取决于你的电脑 | 通常更快 |
| 效果 | 略差一点 | 效果更好 |
| 隐私 | 数据不出本地 | 数据发送到云端 |

**学习阶段推荐用 Ollama**，免费无限调用！

### Q3: 下载模型很慢怎么办？

Ollama 默认从国外下载，可能比较慢。可以：
1. 使用代理
2. 或者耐心等待（只需要下载一次）

### Q4: 运行 demo 报错怎么办？

确保：
1. Ollama 正在运行（后台服务）
2. 已经下载了 Embedding 模型
3. 安装了依赖：`pip install requests numpy`

---

## 下一步

完成这些 demo 后，你就可以：
1. 用 Ollama 构建完整的 RAG 系统
2. Embedding 模型用于检索
3. 千问模型用于生成回答

完全本地化，完全免费！🚀

---

## 快速命令总结

```bash
# 1. 下载 Embedding 模型
ollama pull nomic-embed-text

# 2. 验证模型
ollama list

# 3. 测试 Embedding
curl http://localhost:11434/api/embeddings -d '{"model": "nomic-embed-text", "prompt": "测试"}'

# 4. 运行 demo
cd 4.RAG/相似度demo
python 1_embedding_basic_ollama.py
python 2_cosine_similarity.py
python 3_semantic_search_ollama.py
python 4_chroma_basic.py
python 5_text_chunking.py
```

开始学习吧！
