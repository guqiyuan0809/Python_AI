"""
Demo 5: 文本分块（Text Chunking）
目标：理解重叠分块的原理，这是 RAG Pipeline 的关键步骤

提示:
不需要精确调，先用默认值 chunk_size=500, overlap=50 跑起来，看检索效果。如果发现经常有答案被切断找不到，就把 overlap 调大一点。

overlap 太小：信息容易被切断，检索不到。 overlap 太大：重复内容太多，浪费存储，而且块与块之间内容太相似，检索结果会有很多重复。

实际项目中大部分人就用 500/50 或 1000/100，不会花太多时间在这上面调参。




"""


def chunk_by_length(text, chunk_size=500, overlap=50):
    """
    按固定长度分块（带重叠）
    
    Args:
        text: 原始文本
        chunk_size: 每块的字符数
        overlap: 重叠字符数
    
    Returns:
        分块后的文本列表
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 重叠：下一块从 (start + chunk_size - overlap) 开始
        start += chunk_size - overlap
    
    return chunks


# ============ Demo 1: 基础分块 ============

print("=" * 60)
print("Demo 1: 基础分块（无重叠）")
print("=" * 60)

text = "这是第一句。这是第二句。这是第三句。这是第四句。这是第五句。"
print(f"原始文本: {text}")
print(f"文本长度: {len(text)} 字符")

# 无重叠分块
chunks_no_overlap = chunk_by_length(text, chunk_size=15, overlap=0)

print(f"\n分块结果（chunk_size=15, overlap=0）:")
for i, chunk in enumerate(chunks_no_overlap, 1):
    print(f"  块 {i}: {chunk}")

print("\n问题：如果关键信息在块的边界，可能被切断！")


# ============ Demo 2: 重叠分块 ============

print("\n" + "=" * 60)
print("Demo 2: 重叠分块（推荐）")
print("=" * 60)

# 重叠分块
chunks_with_overlap = chunk_by_length(text, chunk_size=15, overlap=5)

print(f"分块结果（chunk_size=15, overlap=5）:")
for i, chunk in enumerate(chunks_with_overlap, 1):
    print(f"  块 {i}: {chunk}")

print("\n优势：相邻块有重叠，避免关键信息被切断")


# ============ Demo 3: 实际文档分块 ============

print("\n" + "=" * 60)
print("Demo 3: 实际文档分块")
print("=" * 60)

# 模拟一篇长文档
long_text = """
Python 是一门高级编程语言，由 Guido van Rossum 于 1991 年创建。
Python 的设计哲学强调代码的可读性和简洁的语法。

Python 广泛应用于多个领域：
1. Web 开发：Django、Flask、FastAPI 等框架
2. 数据分析：Pandas、NumPy、Matplotlib 等库
3. 人工智能：TensorFlow、PyTorch、scikit-learn 等
4. 自动化脚本：系统管理、测试自动化等

Python 的优势包括：
- 简单易学，适合初学者
- 丰富的第三方库
- 强大的社区支持
- 跨平台兼容性好

学习 Python 的建议路径：
1. 掌握基础语法
2. 学习常用库
3. 做实际项目
4. 参与开源社区
""".strip()

print(f"原始文档长度: {len(long_text)} 字符")

# 推荐配置：chunk_size=200, overlap=20
chunks = chunk_by_length(long_text, chunk_size=200, overlap=20)

print(f"\n分块结果（chunk_size=200, overlap=20）:")
print(f"总共分成 {len(chunks)} 个块\n")

for i, chunk in enumerate(chunks, 1):
    print(f"块 {i} ({len(chunk)} 字符):")
    print(f"{chunk}")
    print("-" * 60)


# ============ Demo 4: 参数选择指南 ============

print("\n" + "=" * 60)
print("Demo 4: 参数选择指南")
print("=" * 60)

print("""
推荐配置：
- 短文本（FAQ）: chunk_size=200-300, overlap=20-30
- 中等文档（文章）: chunk_size=500-800, overlap=50-100
- 长文档（书籍）: chunk_size=1000-1500, overlap=100-200

经验法则：
- overlap 通常是 chunk_size 的 10-20%
- 先用默认值（500/50），根据效果调整

为什么需要重叠？
1. 避免关键信息被切断
2. 保持上下文连贯性
3. 提高检索准确度

示例：
  原文: "...Python 是一门编程语言。FastAPI 是 Web 框架..."
  
  无重叠：
    块1: "...Python 是一门编程语言。"
    块2: "FastAPI 是 Web 框架..."
    问题：如果用户问"Python Web 框架"，可能检索不到
  
  有重叠：
    块1: "...Python 是一门编程语言。FastAPI..."
    块2: "...编程语言。FastAPI 是 Web 框架..."
    优势：两个块都包含相关信息，检索更准确
""")

print("=" * 60)
print("Demo 5: 分块策略总结")
print("=" * 60)

print("""
本 demo 用的是最基础的"固定长度分块"，实际中有多种策略：

1. 固定长度分块（本 demo）
   - 按字符数硬切，简单粗暴
   - 缺点：可能把一句话切断

2. 递归分块（LangChain 推荐，实际项目最常用）
   - 优先按段落切 → 段落太长按句子切 → 最后才按字符切
   - 尽量保持语义完整
   - 代码示例：
     from langchain.text_splitter import RecursiveCharacterTextSplitter
     splitter = RecursiveCharacterTextSplitter(
         chunk_size=500,
         chunk_overlap=50,
         separators=["\\n\\n", "\\n", "。", ".", " ", ""]
     )
     chunks = splitter.split_text(text)

3. 按段落分块 — 以 \\n\\n 为分隔符切
4. 按句子分块 — 以句号为分隔符切
5. 语义分块 — 用 Embedding 模型判断哪里该切，效果最好但最慢

本 demo 的函数帮你理解分块的核心原理（滑动窗口 + 重叠），
实际做项目时直接用 LangChain 的 RecursiveCharacterTextSplitter 就行。
""")

print("=" * 60)
print("✅ 完成！你已经理解了文本分块的原理")
print("=" * 60)
