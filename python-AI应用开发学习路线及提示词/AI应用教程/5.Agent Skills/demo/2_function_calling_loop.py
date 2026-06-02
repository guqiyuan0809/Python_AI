"""
Demo 2: Function Calling 循环调用（Agent 的雏形）
目标：理解 Agent 的核心循环 —— AI 可以多次调用工具，直到任务完成

上一个 demo 是单次调用，这个 demo 演示 AI 自主循环：
思考 → 调用工具 → 看结果 → 继续思考 → 再调用 → ... → 最终回答

这就是 Agent 的核心模式！
"""

import json
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
client = OpenAI(
    api_key="sk-2e56e7e02258424eaf6afe699b054031",
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-chat"


# ============================================================
# 工具函数
# ============================================================
def get_weather(city: str) -> str:
    """查询城市天气"""
    weather_data = {
        "北京": "晴天，35°C",
        "上海": "多云，22°C",
        "深圳": "阵雨，28°C",
    }
    return weather_data.get(city, f"暂无 {city} 的天气数据")


def get_population(city: str) -> str:
    """查询城市人口"""
    population_data = {
        "北京": "2170 万人",
        "上海": "2490 万人",
        "深圳": "1760 万人",
    }
    return population_data.get(city, f"暂无 {city} 的人口数据")


def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


# 工具描述
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气和温度",
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
            "name": "get_population",
            "description": "查询指定城市的人口数量",
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
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    }
]

# 函数名（字符串）→ 函数对象 的映射
# AI 返回的是字符串 "calculator"，不能直接调用
# 通过这个字典把字符串转成真正的函数：available_functions["calculator"] → calculator 函数
# 这样就能动态调用：available_functions[function_name](**args)，不需要写一堆 if-else


#本质
#就是一个注册表——你定义了哪些工具函数，就在这里注册一下，Agent 循环里就能根据 AI 的决策自动找到并执行对应的函数。
available_functions = {
    "get_weather": get_weather,
    "get_population": get_population,
    "calculator": calculator,
}


# ============================================================
# Agent 循环：AI 自主决定调用工具，直到任务完成
# ============================================================
def agent_loop(user_message: str, max_iterations: int = 5):
    """
    Agent 核心循环

    和 Demo 1 的区别：
    - Demo 1：AI 调用一次工具就结束
    - 这里：AI 可以调用多次工具，自己判断什么时候结束

    这就是 Agent 的核心！
    """
    print(f"\n{'=' * 60}")
    print(f"用户: {user_message}")
    print(f"{'=' * 60}")

    messages = [
        {"role": "system", "content": "你是一个有帮助的助手，可以查询天气、人口和做计算。请一步步完成用户的任务。"},
        {"role": "user", "content": user_message}
    ]

    for iteration in range(max_iterations):
        print(f"\n--- 第 {iteration + 1} 轮 ---")

        # 调用 AI
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        

        #从 API 返回结果中取出 AI 的回复消息对象
        #assistant_message.content     # AI 的文字回答（如果直接回答的话）
        #assistant_message.tool_calls  # AI 想调用的工具列表（如果需要调工具的话）

        assistant_message = response.choices[0].message

        # 检查是否需要调用工具
        if assistant_message.tool_calls:
            print(f"AI 决定调用工具：")
            messages.append(assistant_message)

            # 执行所有工具调用
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"  → {function_name}({function_args})")

                func = available_functions[function_name]
                result = func(**function_args)
                print(f"  ← 结果: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        else:
            # AI 不再调用工具，给出最终回答
            print(f"AI 完成任务，给出最终回答：")
            print(f"\nAI: {assistant_message.content}")
            return assistant_message.content

    print("达到最大循环次数，强制结束")
    return None


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 测试 1：需要调用多个工具的复杂问题
    agent_loop("帮我查一下北京和上海的天气，然后告诉我哪个城市更热")

    # 测试 2：需要先查数据再计算
    agent_loop("北京和上海的人口加起来是多少？")

    # 测试 3：简单问题，不需要工具
    agent_loop("什么是 Python？")

    print("\n" + "=" * 60)
    print("✅ 完成！你已经理解了 Agent 的核心循环")
    print("=" * 60)
    print("""
Agent 循环 vs 普通 Function Calling：

普通 Function Calling（Demo 1）：
  用户提问 → AI 调用 1 次工具 → 回答
  
Agent 循环（本 Demo）：
  用户提问 → AI 调用工具 → 看结果 → 需要更多信息？
           → 再调用工具 → 看结果 → 够了吗？
           → ... → 最终回答

关键区别：Agent 自己决定什么时候停下来。
这就是 Agent 的核心模式：循环 + 自主决策。
""")
