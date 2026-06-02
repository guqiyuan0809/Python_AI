"""
Ollama 本地模型调用 Demo
覆盖：基础调用、流式输出、多轮对话
使用本地 Ollama + qwen2.5:3b 模型

重点：和云端 API 代码几乎一模一样，只改了 base_url 和 model
"""

from openai import OpenAI

# ============================================================
# 配置（本地 Ollama，不需要真实 API Key）
# ============================================================
client = OpenAI(
    api_key="ollama",  # Ollama 不校验 Key，随便填
    base_url="http://localhost:11434/v1"  # Ollama 本地 API 地址
)
MODEL = "qwen2.5:3b"


# ============================================================
# 1. 基础调用（和云端 API 完全一样）
# ============================================================
def basic_chat(message: str) -> str:
    """最简单的调用，发一句话，拿一个回答"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": message}],
        max_tokens=200
    )
    return response.choices[0].message.content


# ============================================================
# 2. 带 System Prompt 的调用
# ============================================================
def chat_with_role(message: str) -> str:
    """给 AI 设定角色"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一个资深 Python 后端工程师，擅长 FastAPI 和 Django。回答简洁，直接给代码。"
            },
            {"role": "user", "content": message}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content


# ============================================================
# 3. 流式输出（逐字返回）
# ============================================================
def stream_chat(message: str):
    """流式输出：本地模型也支持，体验和云端一样"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": message}],
        stream=True,
        max_tokens=500
    )

    print("AI: ", end="", flush=True)
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # 换行


# ============================================================
# 4. 多轮对话（带上下文记忆）
# ============================================================
def multi_turn_chat():
    """
    多轮对话：和云端 API 的写法完全一致。
    每次请求都带上历史消息，AI 就能理解上下文。
    """
    history = [
        {"role": "system", "content": "你是一个有帮助的 AI 助手，用中文简洁回答。"}
    ]

    print("=== Ollama 本地多轮对话（输入 'quit' 退出）===\n")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "quit":
            print("再见！")
            break

        history.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=500
        )

        assistant_message = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_message})

        print(f"\nAI: {assistant_message}\n")


# ============================================================
# 5. 对比演示：云端 vs 本地，只改两行
# ============================================================
def show_comparison():
    """展示云端和本地切换有多简单"""
    print("=" * 50)
    print("云端 vs 本地，代码对比")
    print("=" * 50)

    code = """
# ---- 云端 API（OpenRouter + DeepSeek）----
client = OpenAI(
    api_key="sk-or-v1-xxx",
    base_url="https://openrouter.ai/api/v1"
)
MODEL = "deepseek/deepseek-chat-v3-0324"

# ---- 本地 Ollama ----
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)
MODEL = "qwen2.5:3b"

# ---- 调用代码完全一样，一字不改 ----
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)
"""
    print(code)


# ============================================================
# 运行示例
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("1. 基础调用")
    print("=" * 50)
    print(basic_chat("用一句话介绍 Python"))

    print("\n" + "=" * 50)
    print("2. 带角色设定的调用")
    print("=" * 50)
    print(chat_with_role("FastAPI 怎么定义一个 GET 接口？给最简代码"))

    print("\n" + "=" * 50)
    print("3. 流式输出")
    print("=" * 50)
    stream_chat("用三句话介绍什么是 RESTful API")

    print("\n" + "=" * 50)
    print("4. 云端 vs 本地对比")
    print("=" * 50)
    show_comparison()

    # 取消注释可体验多轮对话
    # print("\n" + "=" * 50)
    # print("5. 多轮对话")
    # print("=" * 50)
    # multi_turn_chat()
