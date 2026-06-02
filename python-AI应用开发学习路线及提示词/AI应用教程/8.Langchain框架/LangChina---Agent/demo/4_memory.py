"""
Demo 4: Memory 对话记忆
目标：让 Agent 记住对话历史，能做多轮问答

演示内容：
1. 不带记忆的 Agent（失忆）
2. 用 RunnableWithMessageHistory 加上记忆
3. 多个 session 隔离不同用户
4. 查看历史消息
5. 持久化历史到文件

需要 API Key
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory

# ============================================================
# 配置
# ============================================================
API_KEY = "sk-2e56e7e02258424eaf6afe699b054031"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"


# ============================================================
# 准备 Tools 和 LLM
# ============================================================
@tool
def get_weather(city: str) -> str:
    """查询某个城市的天气。

    Args:
        city: 城市名，例如"北京"
    """
    return f"{city} 25 度，晴"


tools = [get_weather]

llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0,
)


# ============================================================
# 演示 1: 不带记忆 — Agent 会失忆
# ============================================================
print("=" * 60)
print("演示 1: 不带记忆的 Agent")
print("=" * 60)

prompt_no_memory = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent_no_memory = create_tool_calling_agent(llm, tools, prompt_no_memory)
executor_no_memory = AgentExecutor(agent=agent_no_memory, tools=tools, verbose=False)

# 第 1 轮告诉它名字
result1 = executor_no_memory.invoke({"input": "你好，我叫小明"})
print(f"第 1 轮回答: {result1['output']}")

# 第 2 轮问名字 —— 它不记得！
result2 = executor_no_memory.invoke({"input": "我叫什么名字？"})
print(f"第 2 轮回答: {result2['output']}")
print("\n👆 没有记忆，Agent 完全不知道用户是谁")


# ============================================================
# 演示 2: 加上记忆 (RunnableWithMessageHistory)
# ============================================================
print("\n\n" + "=" * 60)
print("演示 2: 带记忆的 Agent")
print("=" * 60)

# Step 1: Prompt 加上 chat_history 占位符（位置在 human 之前）
prompt_with_memory = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手，能记住和用户的对话历史。"),
    MessagesPlaceholder(variable_name="chat_history"),    # ⚠️ 关键
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt_with_memory)
executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# Step 2: 准备一个内存存储 (实际项目用 Redis)
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """根据 session_id 取得该用户的历史消息容器"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Step 3: 用 RunnableWithMessageHistory 包装 executor
agent_with_history = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key="input",            # 输入字段
    history_messages_key="chat_history",   # 历史字段（要和 prompt 里 MessagesPlaceholder 的 variable_name 一致）
)

# Step 4: 调用时多传一个 session_id
config = {"configurable": {"session_id": "user_xiaoming"}}

result1 = agent_with_history.invoke({"input": "你好，我叫小明"}, config=config)
print(f"第 1 轮: {result1['output']}")

result2 = agent_with_history.invoke({"input": "我刚才说我叫什么名字？"}, config=config)
print(f"第 2 轮: {result2['output']}")

result3 = agent_with_history.invoke({"input": "北京天气怎么样？"}, config=config)
print(f"第 3 轮: {result3['output']}")

result4 = agent_with_history.invoke({"input": "我之前问的城市叫什么？"}, config=config)
print(f"第 4 轮: {result4['output']}")
# Agent 应该记得"北京"


# ============================================================
# 演示 3: 多 session 隔离
# ============================================================
print("\n\n" + "=" * 60)
print("演示 3: 多 session 隔离 (不同用户独立记忆)")
print("=" * 60)

config_a = {"configurable": {"session_id": "alice"}}
config_b = {"configurable": {"session_id": "bob"}}

# Alice 告诉 Agent 她叫 Alice
agent_with_history.invoke({"input": "我叫 Alice"}, config=config_a)

# Bob 告诉 Agent 他叫 Bob
agent_with_history.invoke({"input": "我叫 Bob"}, config=config_b)

# Alice 问名字
result_a = agent_with_history.invoke({"input": "我叫什么名字？"}, config=config_a)
print(f"Alice 看到: {result_a['output']}")

# Bob 问名字
result_b = agent_with_history.invoke({"input": "我叫什么名字？"}, config=config_b)
print(f"Bob 看到:   {result_b['output']}")

print("\n👆 两个 session 彼此独立，互不干扰")


# ============================================================
# 演示 4: 查看历史消息
# ============================================================
print("\n\n" + "=" * 60)
print("演示 4: 查看 session 中的历史消息")
print("=" * 60)

history = get_session_history("user_xiaoming")
print(f"小明的对话历史共 {len(history.messages)} 条消息：\n")
for i, msg in enumerate(history.messages, 1):
    role = "👤 用户" if msg.type == "human" else "🤖 助手"
    print(f"{i}. {role}: {msg.content[:60]}...")


# ============================================================
# 演示 5: 持久化到文件 (可选)
# ============================================================
print("\n\n" + "=" * 60)
print("演示 5: 持久化历史到文件")
print("=" * 60)

import os
os.makedirs("./chat_logs", exist_ok=True)


def get_session_history_file(session_id: str):
    """文件版本的历史存储 — 程序重启不丢数据"""
    return FileChatMessageHistory(f"./chat_logs/{session_id}.json")


agent_with_file_history = RunnableWithMessageHistory(
    executor,
    get_session_history_file,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 第 1 次跑：记下信息
config_file = {"configurable": {"session_id": "persistent_user"}}
agent_with_file_history.invoke({"input": "我喜欢吃火锅"}, config=config_file)
print("已保存历史到 ./chat_logs/persistent_user.json")
print("即使程序重启，下次用 session_id='persistent_user' 调用，AI 还是会记得你喜欢火锅")


# ============================================================
# 演示 6: 清空某个 session
# ============================================================
print("\n\n" + "=" * 60)
print("演示 6: 清空 session 历史")
print("=" * 60)

print(f"清空前 alice 的消息数: {len(get_session_history('alice').messages)}")
get_session_history("alice").clear()
print(f"清空后 alice 的消息数: {len(get_session_history('alice').messages)}")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
让 Agent 记住对话历史的关键步骤：

1. ✅ Prompt 加 MessagesPlaceholder("chat_history")
2. ✅ 写 get_session_history(session_id) 函数
3. ✅ 用 RunnableWithMessageHistory 包装 executor
4. ✅ 调用时传 config={"configurable": {"session_id": "..."}}

工程要点：
- session_id 是不同用户/对话的隔离边界
- 内存存储仅适合开发，生产用 Redis 或 SQL
- 长对话要用 trim_messages 截断防 token 超限
- Memory 只记录用户和最终回答的对话，不记录中间工具调用

旧版的 ConversationBufferMemory 等已经过时，都用 RunnableWithMessageHistory。

下一步：把 Tools + RAG + Memory 综合在一起，做完整 Agent → Demo 5
""")
