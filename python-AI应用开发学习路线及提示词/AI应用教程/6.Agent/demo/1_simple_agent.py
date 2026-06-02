"""
Demo 1: 最简 Agent（纯代码实现）
目标：用最少的代码实现一个完整的 Agent

这个 Demo 和 Agent Skills Demo 2 的区别：
- Demo 2 是 Function Calling 循环的演示
- 这里是一个结构更清晰的完整 Agent：有明确的组件划分

核心结构：
  Agent = LLM + Tools + Loop + System Prompt

需要 API Key
"""

import json
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
API_KEY = "sk-2e56e7e02258424eaf6afe699b054031"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============================================================
# 工具定义
# ============================================================
def get_weather(city: str) -> str:
    """查询城市天气（模拟）"""
    data = {
        "北京": "晴天，35°C，空气质量良好",
        "上海": "多云，22°C，有轻雾",
        "深圳": "阵雨，28°C，湿度较大",
    }
    return data.get(city, f"暂无 {city} 的天气数据")


def calculator(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        # 只允许数字和基本运算符
        allowed = set("0123456789+-*/.() ")   #白名单，允许的字符
        if not all(c in allowed for c in expression):
            return "不支持的表达式"
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


def search_knowledge(query: str) -> str:
    """搜索知识库（模拟 RAG 检索）"""
    knowledge = {
        "python": "Python 是一种高级编程语言，适合 Web 开发、数据分析、AI 等领域",
        "fastapi": "FastAPI 是一个高性能的 Python Web 框架，支持异步，自动生成 API 文档",
        "agent": "Agent 是能自主完成任务的 AI 程序，核心是 LLM + 工具 + 循环决策",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"知识库中没有找到关于 '{query}' 的信息"


# ============================================================
# 工具注册（tools 参数 + 函数映射）
# ============================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持加减乘除",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "从知识库中搜索技术相关信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    }
]

# 函数名 → 函数对象 的映射
available_functions = {
    "get_weather": get_weather,
    "calculator": calculator,
    "search_knowledge": search_knowledge,
}


# ============================================================
# Agent 核心：循环决策
# ============================================================

SYSTEM_PROMPT = """你是一个智能助手，可以使用工具来帮助用户完成任务。

你的工作方式：
1. 理解用户的问题
2. 判断是否需要使用工具
3. 如果需要，调用合适的工具
4. 根据工具返回的结果，生成最终回答

注意：
- 当你获得了足够的信息，直接给出最终回答
- 如果工具返回错误，尝试换一种方式
- 回答要基于工具返回的实际数据"""


def run_agent(user_message: str, max_steps: int = 10):
    """
    Agent 主循环
    
    这就是 Agent 的核心：
    1. 把用户消息发给 LLM
    2. LLM 决定调用工具 or 直接回答
    3. 如果调用工具，执行工具，把结果加入 messages
    4. 再次调用 LLM（带上工具结果）
    5. 重复直到 LLM 给出最终回答
    """
    print(f"\n{'=' * 60}")
    print(f"用户: {user_message}")
    print(f"{'=' * 60}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} ---")

        # 调用 LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message

        # 打印 Token 使用情况
        if response.usage:
            print(f"  Token: {response.usage.total_tokens}")

        # 判断：调用工具 or 最终回答
        if message.tool_calls:
            # AI 决定调用工具

            # 把 AI 的决策消息（包含 tool_calls）加入 messages 历史
            # 为什么要加？因为下一轮调 API 时，AI 需要看到完整的对话历史，
            # 包括"我之前决定调了什么工具"，这样它才能知道上下文
            messages.append(message)

            # 遍历 AI 要调用的工具列表
            # AI 一次可能要调多个工具（比如同时查北京和上海天气），tool_calls 里就有多个
            for tool_call in message.tool_calls:
                # 从 tool_call 里取出函数名，比如 "get_weather"
                func_name = tool_call.function.name
                # AI 返回的 arguments 是 JSON 字符串（如 '{"city": "北京"}'）
                # json.loads 把它解析成 Python 字典 {"city": "北京"}
                func_args = json.loads(tool_call.function.arguments)

                print(f"  Action: {func_name}({func_args})")

                # 执行工具
                func = available_functions.get(func_name)
                if func:
                    result = func(**func_args)
                else:
                    result = f"未知工具: {func_name}"

                print(f"  Observation: {result}")

                # 把工具结果加入 messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            # AI 给出最终回答
            print(f"  Answer: {message.content}")
            return message.content

    print("达到最大步数，强制结束")
    return None


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 测试 1：需要调用工具
    run_agent("北京今天天气怎么样？")

    # 测试 2：需要调用多个工具
    run_agent("北京和上海哪个更热？")

    # 测试 3：需要计算
    run_agent("帮我算一下 (100 + 200) * 3.5")

    # 测试 4：知识库检索
    run_agent("FastAPI 是什么？")

    # 测试 5：不需要工具
    run_agent("你好，你是谁？")

    print("\n" + "=" * 60)
    print("完成！这就是一个最简 Agent")
    print("=" * 60)
    print("""
Agent 的核心就这么简单：
1. System Prompt 定义 Agent 的身份和行为
2. Tools 定义 Agent 能用的工具
3. While 或者 for 循环让 Agent 自主决策
4. LLM 是大脑，决定每一步做什么

接下来的 Demo 会在这个基础上加入：
- 显式的思考过程（ReAct）
- 先规划再执行（Plan-and-Execute）
- 记忆管理
- 更多工具和复杂任务
""")
