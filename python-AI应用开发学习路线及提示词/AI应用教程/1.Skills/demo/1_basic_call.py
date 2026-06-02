"""
Demo 1: 基础调用（最简单的 AI API 调用）
目标：理解 AI API 调用的最小流程

核心流程：
1. 创建 client（指定 API Key 和地址）
2. 构造 messages（用户消息）
3. 调用 client.chat.completions.create()
4. 从 response 中取出回答

这是所有 AI 应用的基础，后面的 RAG、Agent 全都建立在这之上。
"""

import sys
from openai import OpenAI

# 修复 Windows 终端编码问题
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# 配置
# ============================================================
client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-chat"


# ============================================================
# 最简单的调用：一问一答
# ============================================================
def basic_chat(message: str) -> str:
    """
    最简单的 AI 调用，发一句话，拿一个回答。

    参数说明：
    - model: 模型名称
    - messages: 消息列表，每条消息有 role 和 content
    - max_tokens: 限制输出长度，防止回答太长浪费 Token
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": message}
        ],
        max_tokens=200
    )

    # response 的结构：
    # response.choices[0].message.content → 回答文本
    # response.usage.total_tokens → Token 用量
    return response.choices[0].message.content


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: 基础调用 — 一问一答")
    print("=" * 60)

    # 测试 1：简单问题
    print("\n[测试 1] 问：用一句话解释什么是 API")
    answer = basic_chat("用一句话解释什么是 API")
    print(f"答：{answer}")

    # 测试 2：翻译
    print("\n[测试 2] 问：翻译成英文：你好世界")
    answer = basic_chat("翻译成英文：你好世界")
    print(f"答：{answer}")

    # 测试 3：代码问题
    print("\n[测试 3] 问：Python 的 list 和 tuple 有什么区别？一句话说")
    answer = basic_chat("Python 的 list 和 tuple 有什么区别？一句话说")
    print(f"答：{answer}")

    print("\n" + "=" * 60)
    print("[完成] 基础调用的核心就三步：")
    print("=" * 60)
    print("""
1. 创建 client：OpenAI(api_key=..., base_url=...)
2. 调用 API：client.chat.completions.create(model=..., messages=[...])
3. 取出回答：response.choices[0].message.content

就这么简单，所有 AI 应用的底层都是这三步。
""")
