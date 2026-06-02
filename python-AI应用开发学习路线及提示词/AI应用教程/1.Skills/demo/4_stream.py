"""
Demo 4: 流式输出（Stream）
目标：理解流式输出的原理和用法

核心概念：
- 普通模式：AI 全部生成完，一次性返回（用户干等）
- 流式模式：AI 一边生成一边返回，逐字出现（打字机效果）

你平时用 ChatGPT、DeepSeek 网页版看到的"打字机效果"，就是流式输出。
做 RAG 和 Agent 时必须用，不然用户要等很久。
"""

import sys
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# 配置
# ============================================================
client = OpenAI(
    api_key="sk-2e56e7e02258424eaf6afe699b054031",
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-chat"


# ============================================================
# 演示 1：普通模式 vs 流式模式 对比
# ============================================================
def demo_normal_mode(message: str):
    """普通模式：等全部生成完才返回"""
    print("[普通模式] 等待中...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": message}],
        max_tokens=200
    )

    # 一次性拿到完整回答
    answer = response.choices[0].message.content
    print(f"[普通模式] AI：{answer}")
    print(f"[Token] 输入: {response.usage.prompt_tokens}, 输出: {response.usage.completion_tokens}")


def demo_stream_mode(message: str):
    """流式模式：边生成边返回，逐字出现"""
    print("[流式模式] AI：", end="", flush=True)

    # 只加了 stream=True 这一个参数
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": message}],
        stream=True,  # 开启流式
        max_tokens=200
    )

    # response 变成了迭代器，每次返回一小段文本
    full_response = ""
    for chunk in response:
        # 流式模式用 .delta.content（不是 .message.content）
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)  # 逐字打印，不换行
            full_response += content  # 拼接完整回答

    print()  # 最后换行
    print(f"[完整回答长度] {len(full_response)} 字符")


# ============================================================
# 演示 2：流式输出拼接完整回答（实际开发中常用）
# ============================================================
def stream_and_collect(message: str) -> str:
    """
    流式输出 + 收集完整回答
    实际开发中常用：一边返回给前端，一边拼接完整回答用于存数据库
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手，用中文回答。"},
            {"role": "user", "content": message}
        ],
        stream=True,
        max_tokens=300
    )

    full_response = ""
    print("AI：", end="", flush=True)

    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    print()
    return full_response


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    question = "用三句话介绍 Python 这门编程语言"

    # 对比两种模式
    print("=" * 60)
    print(f"问题：{question}")
    print("=" * 60)

    print("\n--- 普通模式（等全部生成完才显示）---")
    demo_normal_mode(question)

    print("\n--- 流式模式（逐字出现，打字机效果）---")
    demo_stream_mode(question)

    # 演示拼接完整回答
    print("\n" + "=" * 60)
    print("演示：流式输出 + 收集完整回答")
    print("=" * 60)
    full = stream_and_collect("FastAPI 的三个核心优势是什么？")
    print(f"\n[收集到的完整回答] {full[:50]}...")

    print("\n" + "=" * 60)
    print("[完成] 流式输出核心要点")
    print("=" * 60)
    print("""
1. 普通模式：response.choices[0].message.content（完整文本）
   流式模式：chunk.choices[0].delta.content（每次一小段）

2. 开启流式只需要加一个参数：stream=True

3. 流式模式返回的是迭代器，用 for 循环逐个读取 chunk

4. 实际开发中需要：
   - 一边 print/返回给前端（用户看到打字机效果）
   - 一边拼接完整回答（存数据库、做后续处理）

5. 后面做项目时，前端用 SSE（Server-Sent Events）接收流式数据
   Stream 是从 AI 拿数据的方式，SSE 是把数据推给前端的方式
""")
