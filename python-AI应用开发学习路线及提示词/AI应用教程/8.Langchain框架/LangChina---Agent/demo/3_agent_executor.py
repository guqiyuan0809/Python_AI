"""
Demo 3: Agent 与 AgentExecutor
目标：深入理解 AgentExecutor 的各种参数和调试技巧

演示内容：
1. AgentExecutor 的核心参数（max_iterations、verbose、return_intermediate_steps）
2. 拿到中间步骤（用于前端展示进度）
3. 流式输出（边想边说）
4. 自定义 System Prompt（控制 Agent 行为）

需要 API Key
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ============================================================
# 配置
# ============================================================
API_KEY = "sk-2e56e7e02258424eaf6afe699b054031"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"


# ============================================================
# 准备工具（复用 Demo 2 的 ）
# ============================================================
@tool
def get_weather(city: str) -> str:
    """查询某个城市的实时天气。

    Args:
        city: 城市名，例如"北京"
    """
    fake_data = {"北京": "25 度，晴", "上海": "22 度，多云", "广州": "30 度，雨"}
    return fake_data.get(city, f"{city} 暂无数据")


@tool
def calculator(expression: str) -> str:
    """执行数学计算，例如 "23 * 45"
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"


@tool
def get_user_info(user_id: str) -> str:
    """根据用户 ID 查询用户信息（姓名、会员等级）。

    Args:
        user_id: 用户 ID，例如 "U001"
    """
    fake_users = {
        "U001": "张三, VIP 会员",
        "U002": "李四, 普通用户",
    }
    return fake_users.get(user_id, f"未找到用户 {user_id}")


tools = [get_weather, calculator, get_user_info]

llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0,
)


# ============================================================
# 1. 自定义 System Prompt — 控制 Agent 的"性格"
# ============================================================
SYSTEM_PROMPT = """你是一个智能助手，名字叫小 A。

【你能做的事】
1. 查询城市天气（用 get_weather）
2. 数学计算（用 calculator）
3. 查询用户信息（用 get_user_info）

【行为准则】
- 优先使用工具获取真实信息，不要编造
- 如果用户没给关键信息（比如用户 ID），礼貌地问用户提供
- 用中文回答，语气友好简洁
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)


# ============================================================
# 2. 创建 AgentExecutor (重点演示各个参数)
# ============================================================
executor = AgentExecutor(
    agent=agent,
    tools=tools,

    # 参数 1: verbose 打印执行过程，新手必开
    verbose=True,

    # 参数 2: max_iterations 最多循环次数（防 LLM 死循环烧钱）
    max_iterations=10,

    # 参数 3: return_intermediate_steps 返回中间步骤（用于前端展示进度）
    return_intermediate_steps=True,

    # 参数 4: handle_parsing_errors 解析错误时让 LLM 重试
    handle_parsing_errors=True,
)


# ============================================================
# 演示 1: 看完整中间步骤
# ============================================================
def demo_intermediate_steps():
    print("=" * 60)
    print("演示 1: 拿到中间步骤")
    print("=" * 60)

    result = executor.invoke({"input": "北京天气怎么样？再告诉我用户 U001 是谁"})

    print(f"\n📝 最终回答: {result['output']}")
    print(f"\n🔧 中间步骤数量: {len(result['intermediate_steps'])}")

    # intermediate_steps 是一个列表，每个元素是 (AgentAction, observation)
    for i, (action, observation) in enumerate(result['intermediate_steps'], 1):
        print(f"\n--- 步骤 {i} ---")
        print(f"  调用工具: {action.tool}")
        print(f"  传入参数: {action.tool_input}")
        print(f"  工具返回: {observation}")
    # 实际项目里，前端可以用这些数据展示"AI 正在调用 X 工具..."的提示


# ============================================================
# 演示 2: 流式输出 (打字机效果)
# ============================================================
def demo_streaming():
    print("\n" + "=" * 60)
    print("演示 2: 流式输出")
    print("=" * 60)
    print("（实时事件流，每个事件都是一步进展）\n")

    # executor.stream 输出的每个事件都是 dict
    for event in executor.stream({"input": "上海多少度？"}):
        # 不同步骤会有不同的 key
        if "actions" in event:
            for action in event["actions"]:
                print(f"🔧 决定调用: {action.tool}({action.tool_input})")
        elif "steps" in event:
            for step in event["steps"]:
                print(f"📥 工具返回: {step.observation}")
        elif "output" in event:
            print(f"\n✅ 最终回答: {event['output']}")


# ============================================================
# 演示 3: max_iterations 防死循环
# ============================================================
def demo_max_iterations():
    print("\n" + "=" * 60)
    print("演示 3: max_iterations 起作用")
    print("=" * 60)

    # 用一个 max_iterations=2 的小 executor
    safe_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=2,        # 故意设很小
    )

    # 一个需要多步的复杂问题
    result = safe_executor.invoke({
        "input": "查北京、上海、广州的天气，然后算一下三个城市平均温度"
    })

    print(f"\n📝 输出: {result['output']}")
    print("注意：max_iterations=2 可能不够，Agent 会被强制停止")


# ============================================================
# 演示 4: 缺少信息时的引导回复
# ============================================================
def demo_missing_info():
    print("\n" + "=" * 60)
    print("演示 4: 用户信息不全时，Agent 会主动询问")
    print("=" * 60)

    # 用户没说用户 ID，Agent 不能调 get_user_info
    result = executor.invoke({"input": "查询我的账户信息"})
    print(f"\n📝 回答: {result['output']}")
    # 应该是："请提供您的用户 ID..."


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":

    # 演示 1: 中间步骤
    demo_intermediate_steps()

    # 演示 2: 流式输出
    demo_streaming()

    # 演示 3: 防死循环
    # demo_max_iterations()    # 可选，演示 max_iterations 触发

    # 演示 4: 引导用户提供信息
    demo_missing_info()

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
AgentExecutor 的工程要点：

1. ✅ verbose=True   开发调试必开，能看每步在做什么
2. ✅ max_iterations 设 10 兜底，防 LLM 抽风死循环
3. ✅ return_intermediate_steps=True  拿到中间步骤，前端可展示进度
4. ✅ handle_parsing_errors=True  解析失败自动重试
5. ✅ stream() 用于流式输出，前端 SSE 可以转发这些事件

System Prompt 三件套：
  1. 给身份（你是谁）
  2. 给能力清单（什么时候用什么工具）
  3. 给约束（不能做什么、不知道时怎么办）

下一步：让 Agent 记住对话历史，做多轮问答 → Demo 4
""")
