"""
Demo 2: System Prompt 角色设定
目标：理解 System Prompt 如何改变 AI 的行为和回答风格

核心概念：
- messages 里的 role="system" 就是 System Prompt
- 它是 AI 的"角色说明书"，用户看不到但一直生效
- 同一个问题，不同的 System Prompt 会得到完全不同的回答

这是开发 AI 应用时最常用的技巧：
  客服机器人 → System Prompt 限定只回答业务相关问题
  代码助手 → System Prompt 设定回答风格为简洁、直接给代码
  翻译工具 → System Prompt 设定只做翻译，不解释
"""

import sys
from openai import OpenAI

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
# 带 System Prompt 的调用
# ============================================================
def chat_with_system(system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    """
    带角色设定的 AI 调用

    参数说明：
    - system_prompt: 角色设定（AI 的行为规则）
    - temperature: 随机性，0=确定，1=随机。写代码用 0-0.3，创意用 0.7-1.0
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=temperature,
        max_tokens=300
    )
    return response.choices[0].message.content


# ============================================================
# 测试：同一个问题，不同角色的回答
# ============================================================
if __name__ == "__main__":
    question = "什么是 FastAPI？"

    # 测试 1：无 System Prompt（默认回答）
    print("=" * 60)
    print("测试 1：无角色设定（默认风格）")
    print("=" * 60)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=200
    )
    print(f"问：{question}")
    print(f"答：{response.choices[0].message.content}")

    # 测试 2：Python 工程师角色
    print("\n" + "=" * 60)
    print("测试 2：Python 后端工程师（简洁 + 直接给代码）")
    print("=" * 60)
    answer = chat_with_system(
        system_prompt="你是一个资深 Python 后端工程师，擅长 FastAPI 和 Django。回答简洁，直接给代码示例，不要废话。",
        user_message=question,
        temperature=0.3  # 写代码用低温
    )
    print(f"问：{question}")
    print(f"答：{answer}")

    # 测试 3：初学者导师角色
    print("\n" + "=" * 60)
    print("测试 3：编程导师（用类比解释，适合初学者）")
    print("=" * 60)
    answer = chat_with_system(
        system_prompt="你是一个编程导师，学生是完全没有编程经验的初学者。用生活中的类比解释技术概念，不要用术语。",
        user_message=question,
        temperature=0.7
    )
    print(f"问：{question}")
    print(f"答：{answer}")

    # 测试 4：英文翻译器角色
    print("\n" + "=" * 60)
    print("测试 4：翻译器（只翻译，不解释）")
    print("=" * 60)
    answer = chat_with_system(
        system_prompt="你是一个翻译器。用户输入中文，你只输出对应的英文翻译，不要解释，不要加任何其他内容。",
        user_message="快速排序是一种高效的排序算法",
        temperature=0  # 翻译用 0 温度，结果确定
    )
    print(f"问：快速排序是一种高效的排序算法")
    print(f"答：{answer}")

    print("\n" + "=" * 60)
    print("[完成] System Prompt 核心要点")
    print("=" * 60)
    print("""
1. System Prompt = AI 的角色说明书
2. 同一个问题 + 不同的 System Prompt = 完全不同的回答
3. temperature 影响回答的随机性：
   - 写代码/翻译：0 - 0.3（确定性高）
   - 日常对话：0.5 - 0.7
   - 创意写作：0.7 - 1.0
4. 实际开发中，System Prompt 写得好不好直接决定 AI 应用的质量
""")
