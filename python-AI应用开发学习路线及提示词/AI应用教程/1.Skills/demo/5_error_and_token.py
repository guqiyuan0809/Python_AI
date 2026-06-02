"""
Demo 5: 错误处理 + Token 统计（生产环境必备）
目标：理解生产环境中 AI 调用需要的防护措施

核心内容：
1. 错误处理：API Key 过期、余额不足、网络超时、模型不存在
2. Token 统计：输入/输出/总计 Token 数量，用于成本监控
3. 换模型演示：只改 base_url 和 model 就能切换不同模型

任何上线的 AI 应用都必须有错误处理和 Token 监控。
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
# 演示 1：带错误处理的调用
# ============================================================
def safe_chat(message: str) -> dict:
    """
    生产环境应该这样写：有错误处理、有 Token 统计
    返回结构化结果，方便后续处理
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=200
        )

        return {
            "success": True,
            "answer": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# 演示 2：故意触发错误
# ============================================================
def demo_error_handling():
    """演示各种错误情况"""

    # 错误 1：错误的 API Key
    print("[错误演示 1] 使用错误的 API Key")
    bad_client = OpenAI(
        api_key="sk-wrong-key-12345",
        base_url="https://api.deepseek.com"
    )
    try:
        response = bad_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50
        )
    except Exception as e:
        print(f"  捕获错误: {type(e).__name__}: {str(e)[:100]}")
        print(f"  → 解决：检查 API Key 是否正确、是否过期")

    # 错误 2：错误的模型名
    print("\n[错误演示 2] 使用不存在的模型名")
    try:
        response = client.chat.completions.create(
            model="not-exist-model",
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50
        )
    except Exception as e:
        print(f"  捕获错误: {type(e).__name__}: {str(e)[:100]}")
        print(f"  → 解决：检查模型名是否正确")

    # 错误 3：超时（设置很短的超时时间）
    print("\n[错误演示 3] 超时处理")
    try:
        timeout_client = OpenAI(
            api_key="sk-2e56e7e02258424eaf6afe699b054031",
            base_url="https://api.deepseek.com",
            timeout=0.001  # 设置极短的超时
        )
        response = timeout_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=50
        )
    except Exception as e:
        print(f"  捕获错误: {type(e).__name__}: {str(e)[:100]}")
        print(f"  → 解决：增加超时时间，或加重试逻辑")


# ============================================================
# 演示 3：Token 统计与成本计算
# ============================================================
def demo_token_stats():
    """演示 Token 统计"""
    messages_list = [
        "你好",
        "用一句话解释什么是 API",
        "写一段 Python 快速排序代码，加详细注释",
    ]

    print("不同问题的 Token 消耗对比：\n")
    total_tokens_all = 0

    for msg in messages_list:
        result = safe_chat(msg)
        if result["success"]:
            usage = result["usage"]
            total_tokens_all += usage["total_tokens"]
            print(f"  问题：{msg}")
            print(f"  输入 Token: {usage['prompt_tokens']}")
            print(f"  输出 Token: {usage['completion_tokens']}")
            print(f"  总计 Token: {usage['total_tokens']}")
            print(f"  回答预览：{result['answer'][:50]}...")
            print()

    print(f"  三次调用总 Token: {total_tokens_all}")
    print(f"  DeepSeek 参考成本: 约 ¥{total_tokens_all * 0.000001:.4f}（每百万 Token ¥1）")


# ============================================================
# 演示 4：换模型只改两行
# ============================================================
def demo_switch_model():
    """演示切换模型有多简单"""
    print("""
切换模型只需要改 base_url 和 model：

# DeepSeek（当前使用）
client = OpenAI(api_key="sk-xxx", base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"

# OpenAI GPT
client = OpenAI(api_key="sk-xxx", base_url="https://api.openai.com/v1")
MODEL = "gpt-4o"

# OpenRouter（一个 Key 调多种模型）
client = OpenAI(api_key="sk-or-xxx", base_url="https://openrouter.ai/api/v1")
MODEL = "deepseek/deepseek-chat"

# 本地 Ollama（免费）
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL = "qwen2.5:3b"

# 调用代码完全一样，一字不改！
response = client.chat.completions.create(model=MODEL, messages=[...])
""")


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    # 演示 1：正常调用 + Token 统计
    print("=" * 60)
    print("演示 1：带错误处理的安全调用")
    print("=" * 60)
    result = safe_chat("什么是 RESTful API？一句话说")
    if result["success"]:
        print(f"回答：{result['answer']}")
        print(f"Token：{result['usage']}")
    else:
        print(f"出错：{result['error']}")

    # 演示 2：错误处理
    print("\n" + "=" * 60)
    print("演示 2：错误处理（故意触发错误）")
    print("=" * 60)
    demo_error_handling()

    # 演示 3：Token 统计
    print("\n" + "=" * 60)
    print("演示 3：Token 统计与成本计算")
    print("=" * 60)
    demo_token_stats()

    # 演示 4：换模型
    print("\n" + "=" * 60)
    print("演示 4：切换模型只改两行")
    print("=" * 60)
    demo_switch_model()

    print("=" * 60)
    print("[完成] 生产环境核心要点")
    print("=" * 60)
    print("""
1. 所有 AI 调用都要 try-except 包裹
   常见错误：Key 过期、余额不足、网络超时、模型不存在

2. 每次调用都要记录 Token 用量
   response.usage.prompt_tokens（输入）
   response.usage.completion_tokens（输出）
   输出 Token 通常比输入贵 2-3 倍

3. 用 max_tokens 限制输出长度，防止 AI 说太多浪费钱

4. 换模型只改 base_url 和 model，代码不用改
   → 代码里不要写死，用配置文件管理
""")
