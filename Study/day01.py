"""
Day 01: 智谱 API 最小调用示例

学习目标：
1. 知道 Python 怎么调大模型 API
2. 理解 client / model / messages / response 这四个核心概念
3. 看懂 System Prompt 如何影响回答

运行前准备：
1. 安装依赖：pip install openai
2. 设置环境变量（推荐）：
   PowerShell:
   $env:ZAI_API_KEY="你的智谱APIKey"

说明：
- 这里用的是 OpenAI 兼容写法，后面切换到 DeepSeek / OpenAI / Ollama 时基本只改 key、base_url、model
- 不建议把真实 API Key 直接写进代码
"""

import os
import sys

from openai import OpenAI


# 修复 Windows 控制台中文输出
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


API_KEY = os.getenv("ZAI_API_KEY", "")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-5.1"


def validate_config() -> None:
    """在真正发请求前，先检查最基本的配置。"""
    if not API_KEY:
        raise ValueError(
            "没有读取到智谱 API Key。请先在 PowerShell 中执行：\n"
            '$env:ZAI_API_KEY="你的智谱APIKey"'
        )


def create_client() -> OpenAI:
    """创建一个 OpenAI 兼容客户端。"""
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def basic_chat(user_message: str) -> str:
    """
    最简单的一问一答。

    这里只传 user 消息，适合理解最小闭环。
    """
    client = create_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content


def chat_with_system(system_prompt: str, user_message: str) -> str:
    """
    带 System Prompt 的调用。

    system 就像“角色说明书”，会直接影响模型输出风格。
    """
    client = create_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content


def show_response_structure() -> None:
    """
    演示返回值里最重要的两个信息：
    1. 文本内容
    2. token 用量
    """
    client = create_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个简洁的 Python 助手"},
            {"role": "user", "content": "用一句话解释什么是 FastAPI"},
        ],
        temperature=0.3,
        max_tokens=100,
    )

    print("\n[返回结构演示]")
    print(f"回答内容: {response.choices[0].message.content}")
    print(f"prompt_tokens: {response.usage.prompt_tokens}")
    print(f"completion_tokens: {response.usage.completion_tokens}")
    print(f"total_tokens: {response.usage.total_tokens}")


def main() -> None:
    validate_config()

    print("=" * 60)
    print("Day 01 - 智谱 API 调用入门")
    print("=" * 60)

    print("\n[1] 最小调用")
    answer = basic_chat("请用一句话解释什么是 API")
    print(f"问题: 请用一句话解释什么是 API")
    print(f"回答: {answer}")

    print("\n[2] System Prompt 演示")
    answer = chat_with_system(
        system_prompt="你是一个资深 Python 后端工程师，回答简洁，直接给结论。",
        user_message="什么是 FastAPI？",
    )
    print("问题: 什么是 FastAPI？")
    print(f"回答: {answer}")

    show_response_structure()

    print("\n[今天先记住这三件事]")
    print("1. 调模型的最小闭环：创建 client -> 组织 messages -> 解析 response")
    print("2. system prompt 会直接影响输出风格和边界")
    print("3. 这套写法后面可以平滑迁移到 DeepSeek / OpenAI / Ollama")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[运行失败]")
        print(exc)
