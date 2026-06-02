"""
Demo 3: 记忆机制（三种记忆方式）
目标：理解 Agent 的记忆管理

三种记忆：
1. 短期记忆：完整保存所有对话（你已经会了）
2. 工作记忆：只保留最近 N 轮（滑动窗口）
3. 长期记忆：重要信息存到文件/数据库

不需要调用 API，纯逻辑演示。
"""

import json

# ============================================================
# 1. 短期记忆（你在 Skills 阶段已经实现过）
# ============================================================
print("=" * 60)
print("Demo 1: 短期记忆（完整对话历史）")
print("=" * 60)

short_term_memory = [
    {"role": "system", "content": "你是一个助手"},
]

# 模拟对话
conversations = [
    ("user", "我叫张三"),
    ("assistant", "你好张三！"),
    ("user", "我喜欢 Python"),
    ("assistant", "Python 是个好选择！"),
    ("user", "我住在北京"),
    ("assistant", "北京是个好城市！"),
]

for role, content in conversations:
    short_term_memory.append({"role": role, "content": content})

print(f"对话轮数: {len(conversations) // 2}")
print(f"messages 长度: {len(short_term_memory)}")
print(f"问题: 对话越来越长，Token 越来越多，成本越来越高")
print(f"所有消息:")
for msg in short_term_memory:
    print(f"  [{msg['role']}] {msg['content']}")


# ============================================================
# 2. 工作记忆（滑动窗口，只保留最近 N 轮）
#    实现核心就是:通过列表切片去除历史对话
# ============================================================
print("\n" + "=" * 60)
print("Demo 2: 工作记忆（滑动窗口）")
print("=" * 60)

# 滑动窗口（Sliding Window）：
# 就是一个"移动的框"，框的大小固定，数据往前走，框也跟着往前走，框外面的就丢掉。
#
# MAX_ROUNDS = 2 意味着框只能装 2 轮对话：
#
# 第 1 轮进来：[张三]
#   窗口：[张三]                    ← 框还没满
#
# 第 2 轮进来：[张三] [Python]
#   窗口：[张三] [Python]           ← 框刚好满
#
# 第 3 轮进来：[张三] [Python] [北京]
#   窗口：       [Python] [北京]    ← 框满了，张三被挤出去
#
# 第 4 轮进来：[张三] [Python] [北京] [框架]
#   窗口：                [北京] [框架]  ← Python 也被挤出去了
#
# 就像一个固定宽度的窗户在一排数据上从左往右滑，窗户里能看到的就保留，滑出去的就丢掉。
# 这个概念在算法、网络协议（TCP 滑动窗口）、数据流处理里都很常见，核心思想都一样：
# 固定大小的框，新的进来，老的出去。

MAX_ROUNDS = 2  # 只保留最近 2 轮对话

def manage_working_memory(messages, max_rounds=MAX_ROUNDS):
    """
    滑动窗口记忆管理
    保留 system prompt + 最近 N 轮对话
    """
    if len(messages) <= 1:  # 只有 system prompt
        return messages

    system_msg = messages[0]  # 保留 system prompt
    history = messages[1:]     # 对话历史

    # 每轮 = 1 条 user + 1 条 assistant = 2 条消息
    max_messages = max_rounds * 2

    if len(history) > max_messages:
        # Python 负数切片 [-4:] 就是"取最后 4 条"，天然实现了滑动窗口的效果
        history = history[-max_messages:]  # 只保留最后 N 轮

    return [system_msg] + history


# 模拟：对话越来越长
working_memory = [
    {"role": "system", "content": "你是一个助手"},
]

all_conversations = [
    ("user", "我叫张三"),
    ("assistant", "你好张三！"),
    ("user", "我喜欢 Python"),
    ("assistant", "Python 是个好选择！"),
    ("user", "我住在北京"),
    ("assistant", "北京是个好城市！"),
    ("user", "推荐一个框架"),
    ("assistant", "推荐 FastAPI！"),
]

for role, content in all_conversations:
    working_memory.append({"role": role, "content": content})
    # 每次添加后，裁剪记忆
    working_memory = manage_working_memory(working_memory)

print(f"总共对话了 {len(all_conversations) // 2} 轮")
print(f"但 messages 只保留了最近 {MAX_ROUNDS} 轮:")
for msg in working_memory:
    print(f"  [{msg['role']}] {msg['content']}")
print(f"\n注意: '我叫张三' 和 '我喜欢 Python' 已经被丢掉了")
print(f"优点: 控制 Token 成本")
print(f"缺点: AI 不记得用户叫张三了")


# ============================================================
# 3. 长期记忆（持久化存储）
# ============================================================
# 长期记忆不是存对话原文，而是存从对话中提取出来的关键信息。
# 比如用户说"我叫张三，我喜欢 Python"，只需要存 name: 张三、language: Python。
# 存关键信息比存整句话更高效、检索更快。
#
# 存储结构（JSON 文件）：
# {
#   "user_001": {          ← user_id：用户唯一标识，区分不同用户，实际项目中一般用数据库自增 ID 或 UUID
#     "name": "张三",       ← key: 信息类别名，value: 具体的值
#     "language": "Python",
#     "city": "北京"
#   }
# }

print("\n" + "=" * 60)
print("Demo 3: 长期记忆（持久化存储）")
print("=" * 60)

MEMORY_FILE = "user_memory.json"

def save_long_term_memory(user_id: str, key: str, value: str):
    """保存长期记忆到文件"""
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            memory = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        memory = {}

    if user_id not in memory:
        memory[user_id] = {}
    memory[user_id][key] = value

    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

    print(f"  已保存: {user_id}.{key} = {value}")


def load_long_term_memory(user_id: str) -> dict:
    """读取长期记忆"""
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        return memory.get(user_id, {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def build_system_prompt_with_memory(user_id: str) -> str:
    """
    把长期记忆加入 system prompt

    为什么要这样做？
    AI 每次对话都是"失忆"的，它不会自动记得之前的任何信息。
    把记忆塞进 system prompt，AI 回答前会先读到这些信息，就好像它"记得"用户是谁。

    效果对比：
    没有长期记忆：
      用户：推荐一个框架
      AI：你想用什么语言？（它不知道你用 Python）

    有长期记忆（system prompt 里写了 language: Python）：
      用户：推荐一个框架
      AI：推荐 FastAPI，很适合 Python Web 开发（它"知道"你用 Python）

    本质：AI 没有真正的记忆，通过 system prompt 把信息"喂"给它，模拟出记忆的效果。
    """
    memory = load_long_term_memory(user_id)
    if memory:
        memory_str = "\n".join([f"- {k}: {v}" for k, v in memory.items()])
        return f"""你是一个有帮助的助手。

你记得这个用户的以下信息：
{memory_str}

请根据这些信息个性化回答。"""
    else:
        return "你是一个有帮助的助手。"


# 模拟：保存用户信息
print("保存用户信息到文件:")
save_long_term_memory("user_001", "name", "张三")
save_long_term_memory("user_001", "language", "Python")
save_long_term_memory("user_001", "city", "北京")

# 模拟：下次对话时读取
print(f"\n下次对话时，读取用户记忆:")
memory = load_long_term_memory("user_001")
print(f"  用户记忆: {memory}")

system_prompt = build_system_prompt_with_memory("user_001")
print(f"\n生成的 system prompt:")
print(f"  {system_prompt}")

print(f"\n这样即使程序重启，AI 也能记住用户信息")

# 清理测试文件
import os
if os.path.exists(MEMORY_FILE):
    os.remove(MEMORY_FILE)


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("✅ 完成！三种记忆方式总结")
print("=" * 60)
print("""
1. 短期记忆（messages 列表）
   - 保存完整对话历史
   - 简单但 Token 成本高
   - 适合：短对话

2. 工作记忆（滑动窗口）
   - 只保留最近 N 轮
   - 控制成本但会丢失早期信息
   - 适合：长对话、成本敏感

3. 长期记忆（持久化存储）
   - 重要信息存到文件/数据库
   - 程序重启也不丢失
   - 适合：记住用户偏好、跨会话记忆

实际项目中通常组合使用：
  短期记忆（当前对话）+ 长期记忆（用户偏好）
""")
