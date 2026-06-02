"""
Demo 1: Document Loaders（文档加载）
目标：学会用 LangChain 加载不同格式的文档

演示内容：
1. TextLoader 加载单个文本文件
2. DirectoryLoader 批量加载文件夹
3. Document 对象的结构（page_content + metadata）

不需要 API Key
"""

from langchain_community.document_loaders import TextLoader, DirectoryLoader

# ============================================================
# 1. TextLoader — 加载单个文本文件
# ============================================================
print("=" * 60)
print("1. TextLoader — 加载单个文本文件")
print("=" * 60)

# 加载一个 txt 文件
# encoding="utf-8" 很重要，中文文件不加会乱码
loader = TextLoader("./sample_docs/python_intro.txt", encoding="utf-8")
documents = loader.load()

# load() 返回一个列表，每个文件对应一个 Document 对象
print(f"加载了 {len(documents)} 个文档")
print(f"\n文档类型: {type(documents[0])}")

# Document 对象只有两个字段：
# page_content：文本内容（字符串）
# metadata：元数据（字典），记录来源信息
print(f"\n--- page_content（前 200 字）---")
print(documents[0].page_content[:200])

print(f"\n--- metadata ---")
print(documents[0].metadata)
# 输出: {'source': './sample_docs/python_intro.txt'}
# metadata 自动记录了文件路径，后面做引用溯源时会用到

print(f"\n文档总长度: {len(documents[0].page_content)} 字符")


# ============================================================
# 2. DirectoryLoader — 批量加载文件夹
# ============================================================
print("\n" + "=" * 60)
print("2. DirectoryLoader — 批量加载文件夹")
print("=" * 60)

# 加载 sample_docs 文件夹下所有 .txt 文件
loader = DirectoryLoader(
    "./sample_docs",                    # 目标文件夹
    glob="**/*.txt",                    # 匹配模式：所有 .txt 文件
    loader_cls=TextLoader,              # 每个文件用 TextLoader 加载
    loader_kwargs={"encoding": "utf-8"} # 传给 TextLoader 的参数
)

documents = loader.load()

print(f"共加载了 {len(documents)} 个文档\n")

# 遍历每个文档，查看来源和长度
for i, doc in enumerate(documents):
    source = doc.metadata["source"]
    length = len(doc.page_content)
    # 取文件名（去掉路径）
    filename = source.split("/")[-1].split("\\")[-1]
    print(f"  文档 {i+1}: {filename}（{length} 字符）")


# ============================================================
# 3. Document 对象详解
# ============================================================
print("\n" + "=" * 60)
print("3. Document 对象详解")
print("=" * 60)

# 你也可以手动创建 Document 对象
from langchain_core.documents import Document

custom_doc = Document(
    page_content="这是我手动创建的文档内容",
    metadata={
        "source": "custom",
        "author": "me",
        "topic": "test"
    }
)

print(f"page_content: {custom_doc.page_content}")
print(f"metadata: {custom_doc.metadata}")

# 为什么要知道这个？
# 因为后面做 RAG 时，你可能需要手动给文档加 metadata
# 比如从数据库读出的数据，需要自己构造 Document 对象


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
1. TextLoader：加载单个文本文件，注意加 encoding="utf-8"
2. DirectoryLoader：批量加载文件夹，用 glob 指定匹配模式
3. 所有加载器返回 Document 对象列表
4. Document = page_content（文本）+ metadata（元数据）
5. metadata 自动记录来源信息，是引用溯源的基础

下一步：用 Text Splitters 把文档分块 → Demo 2
""")
