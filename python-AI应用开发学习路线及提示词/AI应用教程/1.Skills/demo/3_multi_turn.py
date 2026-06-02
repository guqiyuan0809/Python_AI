"""
Demo 3: 多轮对话（上下文记忆）
目标：理解 AI 的"记忆"是怎么实现的

核心原理：
- AI 本身不记忆！每次调用都是独立的
- "记忆"是你在代码里维护一个 messages 列表
- 每次请求都把之前的对话历史全部带上
- AI 看到完整历史，就能理解上下文

这是聊天应用、RAG 多轮问答、Agent 的基础。
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
# 演示 1：没有历史记录，AI 不记得上文
# ============================================================
def demo_no_memory():
    """演示：不带历史，AI 不记得你说过什么"""
    print("=" * 60)
    print("演示 1：不带历史记录（AI 不记得上文）")
    print("=" * 60)

    # 第一次对话
    response1 = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "我叫张三"}],
        max_tokens=100
    )
    print(f"你：我叫张三")
    print(f"AI：{response1.choices[0].message.content}")

    # 第二次对话（没带历史）
    response2 = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "我叫什么名字？"}],
        max_tokens=100
    )
    print(f"\n你：我叫什么名字？")
    print(f"AI：{response2.choices[0].message.content}")
    print("\n→ AI 不知道你叫什么，因为第二次请求没带历史记录")


# ============================================================
# 演示 2：带上历史记录，AI 就能"记住"
# ============================================================
def demo_with_memory():
    """演示：带上历史记录，AI 就有了"记忆" """
    print("\n" + "=" * 60)
    print("演示 2：带上历史记录（AI 有了记忆）")
    print("=" * 60)

    # 维护一个 messages 列表（这就是"记忆"）
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手，用简洁的中文回答。"}
    ]

    # 第一轮
    messages.append({"role": "user", "content": "我叫张三"})
    response1 = client.chat.completions.create(
        model=MODEL,
        messages=messages,  # 发送完整历史
        max_tokens=100
    )
    ai_reply1 = response1.choices[0].message.content
    messages.append({"role": "assistant", "content": ai_reply1})  # 把 AI 回答也加入历史

    print(f"你：我叫张三")
    print(f"AI：{ai_reply1}")

    # 第二轮（带上了第一轮的历史）
    messages.append({"role": "user", "content": "我叫什么名字？"})
    response2 = client.chat.completions.create(
        model=MODEL,
        messages=messages,  # 包含了第一轮的对话
        max_tokens=100
    )
    ai_reply2 = response2.choices[0].message.content

    print(f"\n你：我叫什么名字？")
    print(f"AI：{ai_reply2}")
    print("\n→ AI 记住了！因为 messages 里包含了之前的对话")

    # 打印当前 messages 列表，看看"记忆"长什么样
    print("\n当前 messages 列表（这就是 AI 的记忆）：")
    for msg in messages:
        print(f"  [{msg['role']}] {msg['content']}")


# ============================================================
# 演示 3：交互式多轮对话
# ============================================================
def demo_interactive():
    """交互式多轮对话，体验完整流程"""
    print("\n" + "=" * 60)
    print("演示 3：交互式多轮对话（输入 quit 退出）")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一个 Python 教学助手，用简洁的中文回答，适当给代码示例。"}
    ]

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        # 用户消息加入历史
        messages.append({"role": "user", "content": user_input})

        # 发送完整历史给 AI
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=300
        )

        ai_reply = response.choices[0].message.content

        # AI 回答也加入历史
        messages.append({"role": "assistant", "content": ai_reply})

        print(f"AI：{ai_reply}")
        print(f"  [Token] 输入: {response.usage.prompt_tokens}, 输出: {response.usage.completion_tokens}")
        print(f"  [历史] messages 长度: {len(messages)} 条")


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    # 先看对比演示
    demo_no_memory()
    demo_with_memory()

    # 再体验交互式对话
    print("\n" + "=" * 60)
    print("现在你可以体验交互式多轮对话")
    print("=" * 60)
    demo_interactive()

    print("\n" + "=" * 60)
    print("[完成] 多轮对话核心要点")
    print("=" * 60)
    print("""
1. AI 本身不记忆，每次调用都是独立的
2. "记忆"是你在代码里维护的 messages 列表
3. 每次请求把完整历史发给 AI，AI 就能理解上下文
4. messages 三种 role：
   - system：角色设定（可选）
   - user：用户说的话
   - assistant：AI 之前的回答
5. 对话越长，messages 越大，Token 越多，成本越高
   → 后面学 Agent 时会学记忆管理（滑动窗口、长期记忆）
""")
