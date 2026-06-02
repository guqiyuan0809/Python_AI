"""
Demo 1: Embedding 基础调用（Ollama 本地版本 - 完全免费）
目标：理解 Embedding 的输入输出，文本 → 数字向量

使用 Ollama 本地 Embedding 模型，完全免费，无需 API Key

前置条件：
1. 已安装 Ollama
2. 已下载 Embedding 模型：ollama pull nomic-embed-text
"""

import requests
import json

def get_embedding(text, model="nomic-embed-text"):
    """
    调用 Ollama 本地 Embedding API
    
    Args:
        text: 要向量化的文本
        model: Embedding 模型名称
    
    Returns:
        向量列表
    """
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


# 单个文本向量化
print("=" * 50)
print("Demo 1: 单个文本向量化")
print("=" * 50)

text = "你好，世界"
vector = get_embedding(text)

print(f"原始文本: {text}")
print(f"向量维度: {len(vector)}")
print(f"向量前 10 个值: {vector[:10]}")
print(f"向量类型: {type(vector)}")

# 批量文本向量化
print("\n" + "=" * 50)
print("Demo 2: 批量文本向量化")
print("=" * 50)

texts = [
    "Python 是一门编程语言",
    "FastAPI 是一个 Web 框架",
    "机器学习很有趣"
]

vectors = []
for text in texts:
    vector = get_embedding(text)
    vectors.append(vector)

print(f"输入文本数量: {len(texts)}")
print(f"生成向量数量: {len(vectors)}")
print(f"每个向量维度: {len(vectors[0])}")

for i, text in enumerate(texts):
    print(f"\n文本 {i+1}: {text}")
    print(f"向量前 5 个值: {vectors[i][:5]}")

print("\n" + "=" * 50)
print("✅ 完成！你已经理解了 Embedding 的基本用法")
print("=" * 50)

print("""
Ollama Embedding 模型：
- nomic-embed-text: 英文为主，768 维，约 274MB
- mxbai-embed-large: 多语言支持，1024 维，约 669MB

优势：
- 完全免费
- 无限调用
- 数据不出本地
- 无需 API Key

使用方法：
1. ollama pull nomic-embed-text
2. 运行本脚本
""")






# 1. 停止 Ollama
#taskkill /F /IM ollama.exe 2>$null
#Start-Sleep -Seconds 3

# 2. 删除整个 models 文件夹（会删除所有模型）
#Remove-Item -Path "C:\Users\可以\.ollama\models" -Recurse -Force

# 3. 重新下载

