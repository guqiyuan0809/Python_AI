"""
Day 02: 多轮对话与流式输出

学习目标：
1. 理解 AI 的“记忆”不是模型天然拥有，而是代码维护的 messages 历史
2. 理解 stream=True 如何实现流式输出
3. 为后续 FastAPI 封装聊天接口打基础

运行前准备：
1. 确保 Study/.env 已经存在，并且配置了 DASHSCOPE_API_KEY
2. 已安装依赖：pip install openai python-dotenv
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=False)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-plus")


def validate_config() -> None:
    if not API_KEY:
        raise ValueError(
            "没有读取到通义千问 API Key。\n"
            "请在 D:\\Pythoncode\\Study\\.env 中配置：\n"
            "DASHSCOPE_API_KEY=你的APIKey"
        )


def create_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def demo_no_memory() -> None:
    """
    不带历史时，模型不会记住上文。
    每次请求如果只发当前问题，模型看到的就是“全新对话”。
    """
    client = create_client()

    print("=" * 60)
    print("演示 1：不带历史记录")
    print("=" * 60)

    response1 = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "我叫小刘"}],
        temperature=0.3,
        max_tokens=100,
    )
    print("用户提问：我叫小刘")
    print(f"AI：{response1.choices[0].message.content}")

    response2 = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "我叫什么名字？"}],
        temperature=0.3,
        max_tokens=100,
    )
    print("\n用户继续提问：我叫什么名字？")
    print(f"AI：{response2.choices[0].message.content}")
    print("\n结论：AI 不会天然记住你刚刚说过的话。")


def demo_with_memory() -> None:
    """
    通过维护 messages 列表，把完整对话历史一起发给模型。
    这就是多轮对话“记忆”的本质。
    """
    client = create_client()
    messages = [
        {"role": "system", "content": "你是刘家豪的一个简洁的中文助手。"},
    ]

    print("\n" + "=" * 60)
    print("演示 2：带上历史记录")
    print("=" * 60)

    messages.append({"role": "user", "content": "我叫刘家豪，是你的老大"})
    response1 = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=100,
    )
    assistant_reply1 = response1.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply1})

    print("用户提问：我叫小刘")
    print(f"AI：{assistant_reply1}")

    messages.append({"role": "user", "content": "我叫什么名字？"})
    response2 = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=100,
        stream=True,
    )
    # assistant_reply2 = response2.choices[0].message.content
    # 流式输出
    assistant_reply2 = ""
    for chunk in response2:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            assistant_reply2 += delta
    messages.append({"role": "assistant", "content": assistant_reply2})
    print("\n你：我叫什么名字？")
    print(f"AI：{assistant_reply2}")
    #
    # print("\n当前 messages 历史：")
    # for item in messages:
    #     print(f"[{item['role']}] {item['content']}")

    print("\n结论：多轮对话的记忆来自 messages 历史，不是模型自己记住的。加了一个流式输出")


def stream_chat(user_message: str) -> str:
    """
    流式输出：模型边生成边返回。
    实际项目里通常一边返回给前端，一边拼接完整文本以便存库。
    """
    client = create_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个 Python 学习助手，回答简洁清晰。"},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=300,
        stream=True,
    )

    full_text = ""
    print("AI：", end="", flush=True)
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_text += delta
    print()
    return full_text


def demo_stream() -> None:
    print("\n" + "=" * 60)
    print("演示 3：流式输出")
    print("=" * 60)
    full_text = stream_chat("请用三句话解释为什么 FastAPI 适合做 AI 接口服务。")
    print(f"\n收集到的完整回答长度：{len(full_text)} 字符")
    print("结论：stream=True 后，返回值不再是一次性完整文本，而是可迭代的数据块。")


def main() -> None:
    validate_config()

    print("=" * 60)
    print("Day 02 - 多轮对话与流式输出")
    print("=" * 60)
    print(f"当前模型: {MODEL}")
    print(f"接口地址: {BASE_URL}")

    # demo_no_memory()
    demo_with_memory()
    # demo_stream()

    print("\n" + "=" * 60)
    print("今天先记住这四件事")
    print("=" * 60)
    print("1. AI 本身不记忆，每次调用默认都是独立请求")
    print("2. 多轮对话的记忆，本质是程序维护 messages 历史")
    print("3. 流式输出只要加 stream=True，但读取方式会从 message.content 变成 delta.content")
    print("4. 后面做 FastAPI 项目时，多轮对话和流式输出都会直接用上")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[运行失败]")
        print(exc)
