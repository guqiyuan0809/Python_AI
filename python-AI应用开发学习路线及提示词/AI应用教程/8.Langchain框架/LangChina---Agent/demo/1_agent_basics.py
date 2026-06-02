"""
Demo 1: Agent 基础概念
目标：跑通你的第一个 Agent，直观感受 LLM 自己决策调用工具的过程

演示内容：
1. 最小 Agent：1 个 LLM + 2 个工具 + 1 个 Prompt
2. 看 Agent 如何根据问题自动选择工具
3. 多步推理（先调 A 再调 B）
4. 不需要工具时，直接回答

需要 API Key（调用 DeepSeek 的 Function Calling）

前置条件：
- pip install langchain langchain-openai langchain-core
- 把下面的 API_KEY 改成你自己的
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# ============================================================
# 配置
# ============================================================
API_KEY = "sk-2e56e7e02258424eaf6afe699b054031"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
DEBUG_AGENT = False  # 改成 True 可以查看 LangGraph 的原始调试日志


# ============================================================
# 1. 定义工具（Tools）
# ============================================================
# 用 @tool 装饰器，把普通函数变成 Agent 能调用的工具
# 关键：docstring 决定 LLM 看到什么——它只看 description，不看代码

@tool
def get_weather(city: str) -> str:
    """查询某个城市的实时天气，包括温度和天气状况。

    Args:
        city: 城市名称，例如"北京"、"上海"
    """
    # 假数据，实际项目里会调真实 API
    fake_data = {
        "北京": "25 度，晴朗",
        "上海": "22 度，多云",
        "广州": "30 度，雷阵雨",
    }
    return fake_data.get(city, f"{city} 天气查询暂不支持")


@tool
def calculator(expression: str) -> str:
    """执行数学计算，支持加减乘除。

    Args:
        expression: 数学表达式，例如 "2 + 3 * 4"、"100 - 25"
    """
    try:
        # 实际项目要做表达式安全检查，不能直接 eval 用户输入
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"


# ============================================================
# 2. 把 Tools 装成列表
# ============================================================
tools = [get_weather, calculator]

# 看一下自动生成的 Tool 定义
print("=" * 60)
print("Tools 信息（LLM 看到的就是这些）")
print("=" * 60)
for t in tools:
    print(f"\n名字: {t.name}")
    print(f"描述: {t.description.strip()}")
    print(f"参数: {t.args}")


# ============================================================
# 3. 初始化 LLM
# ============================================================
# Agent 推荐 temperature=0，保证决策稳定
llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0,
)


# ============================================================
# 4. 创建 Agent（LangChain 1.x 写法）
# ============================================================
# create_agent 会直接返回一个可运行的 Agent 图。
# 在 1.x 中，不再需要手动创建 AgentExecutor，也不用手写 agent_scratchpad。
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个智能助手。根据用户问题，需要时调用工具获取信息。",
    debug=DEBUG_AGENT,
)


# ============================================================
# 6. 测试 Agent
# ============================================================
def run_agent(question):
    print("\n" + "=" * 60)
    print(f"❓ 用户提问: {question}")
    print("=" * 60)

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    for message in result["messages"]:
        for tool_call in getattr(message, "tool_calls", []) or []:
            print(f"🔧 决定调用工具: {tool_call['name']}({tool_call['args']})")

        if getattr(message, "type", "") == "tool":
            tool_name = getattr(message, "name", "unknown_tool")
            print(f"📥 工具返回 {tool_name}: {message.content}")

    final_message = result["messages"][-1]
    print(f"\n✅ 最终回答: {final_message.content}")
    return result


if __name__ == "__main__":

    # 测试 1：单工具调用 —— LLM 该选 get_weather
    run_agent("北京今天天气怎么样？")

    # 测试 2：单工具调用 —— LLM 该选 calculator
    run_agent("帮我算一下 23 * 45")

    # 测试 3：多工具调用 ⭐ 重点 —— LLM 要先查两个城市天气，再算温差
    run_agent("北京比上海热多少度？")

    # 测试 4：不需要工具 —— LLM 应该直接回答，不调任何工具
    run_agent("你好，自我介绍一下")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
你已经看到了 Agent 的核心特性：

1. 🤖 自主决策：LLM 自己看问题选工具，不是你写 if-else
2. 🔄 循环执行：create_agent 自动管理"决策→执行→回填→再决策"循环
3. 🧩 多步推理：复杂问题能拆成多个工具调用（如查温差先查两个温度）
4. 🚫 智能跳过：不需要工具时不会强行调用

每次工具调用，程序会把关键过程打印出来：
  - "决定调用工具"  ← LLM 决定要调哪个工具
  - "工具返回"      ← Agent 拿到工具执行结果
  - "最终回答"      ← LLM 看完工具结果后组织回答

下一步：学会写更复杂的 Tool（参数校验、错误处理）→ Demo 2
""")
