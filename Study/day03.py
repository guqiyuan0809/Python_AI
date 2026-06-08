"""
Day 03: 错误处理、超时控制与 Token 统计

学习目标：
1. 理解为什么 AI 调用不能只写 happy path
2. 学会给模型调用加错误处理和超时控制
3. 学会统计 token 用量，为后续成本监控打基础

运行前准备：
1. 确保 Study/.env 已存在，并配置了 DASHSCOPE_API_KEY
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


def create_client(timeout: float = 30.0) -> OpenAI:
    """
    timeout 是企业项目里很重要的配置。
    AI 接口比普通 CRUD 慢得多，不配超时会让请求一直挂住。
    """
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=timeout)


def safe_chat(message: str) -> dict:
    """
    结构化返回，而不是直接 print。
    这是后面封装 FastAPI 接口的基础思路。
    """
    client = create_client(timeout=30.0)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是一个简洁的 Python 学习助手。"},
                {"role": "user", "content": message},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return {
            "message": True,
            "answer": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
    except Exception as exc:
        return {
            "message": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def demo_safe_chat() -> None:
    print("=" * 60)
    print("演示 1：安全调用")
    print("=" * 60)

    result = safe_chat("请用一句话解释什么是 RESTful API。")
    if result["message"]:
        print("用户提问:请用一句话解释什么是 RESTful API。")
        print(f"回答：{result['answer']}")
        print("Token 统计：")
        print(f"  prompt_tokens: {result['usage']['prompt_tokens']}")
        print(f"  completion_tokens: {result['usage']['completion_tokens']}")
        print(f"  total_tokens: {result['usage']['total_tokens']}")
    else:
        print(f"错误类型：{result['error_type']}")
        print(f"错误信息：{result['error']}")


def demo_bad_model() -> None:
    """
    故意触发一个可控错误，帮助你理解：
    线上接口出错时，要把异常变成可读的业务信息。
    """
    print("\n" + "=" * 60)
    print("演示 2：错误模型名")
    print("=" * 60)

    client = create_client(timeout=30.0)
    try:
        client.chat.completions.create(
            model="not-exist-model",
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50,
        )
    except Exception as exc:
        print(f"捕获到错误类型：{type(exc).__name__}")
        print(f"错误信息：{exc}")
        print("结论：模型名、权限、平台可用性都可能导致请求失败。")


def demo_timeout() -> None:
    """
    这里把 timeout 故意压得很短，演示超时问题。
    有时不一定稳定复现，但这是企业项目必须考虑的能力。
    """
    print("\n" + "=" * 60)
    print("演示 3：超时控制")
    print("=" * 60)

    timeout_client = create_client(timeout=0.001)
    try:
        timeout_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50,
        )
        print("这次没有触发超时，但线上仍然要配 timeout。")
    except Exception as exc:
        print(f"捕获到错误类型：{type(exc).__name__}")
        print(f"错误信息：{exc}")
        print("结论：AI 接口必须配置 timeout，避免线程或请求长时间卡住。目前还不知道合适的应该设置为多少秒，但是我用codex中转站最长达到过几十分钟")


def demo_token_compare() -> None:
    """
    让你看到不同问题的 token 消耗不同。
    后面做产品时，这就是成本感知的起点。
    """
    print("\n" + "=" * 60)
    print("演示 4：Token 对比")
    print("=" * 60)

    questions = [
        "你好",
        "请用一句话解释什么是 API。",
        "请写一个 Python 快速排序函数，并加上详细注释。",
    ]

    total = 0
    for question in questions:
        result = safe_chat(question)
        if result["message"]:
            usage = result["usage"]
            total += usage["total_tokens"]
            print(f"问题：{question}")
            print(f"  prompt_tokens: {usage['prompt_tokens']}")
            print(f"  completion_tokens: {usage['completion_tokens']}")
            print(f"  total_tokens: {usage['total_tokens']}")
            print(f"  回答预览：{result['answer'][:60]}")
            print()
        else:
            print(f"问题失败：{question}")
            print(f"  错误：{result['error']}")

    print(f"三次调用 total_tokens 合计：{total}")
    print("结论：问题越复杂、回答越长，token 消耗通常越高。")


def main() -> None:
    validate_config()

    print("=" * 60)
    print("Day 03 - 错误处理、超时控制与 Token 统计")
    print("=" * 60)
    print(f"当前模型: {MODEL}")
    print(f"接口地址: {BASE_URL}")

    # demo_safe_chat()
    # demo_bad_model()
    # demo_timeout()
    demo_token_compare()

    # print("\n" + "=" * 60)
    # print("今天先记住这四件事")
    # print("=" * 60)
    # print("1. AI 调用不能只写成功路径，必须做异常处理")
    # print("2. timeout 是线上接口稳定性的基础配置")
    # print("3. 结构化返回比直接 print 更适合后续封装成服务")
    # print("4. token 统计是成本监控和优化的基础")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[运行失败]")
        print(exc)
