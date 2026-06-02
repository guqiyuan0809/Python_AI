"""
Demo 1: Function Calling 基础（纯代码实现）
目标：理解 AI 如何决定调用函数，以及完整的调用流程

使用 OpenRouter + DeepSeek 模型

核心流程：
1. 定义工具函数
2. 把工具描述告诉 AI
3. AI 判断是否需要调用
4. 你执行函数
5. 把结果给 AI
6. AI 生成最终回答
"""

import json
import sys
from openai import OpenAI

# 修复 Windows 终端编码问题
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
# 第 1 步：定义工具函数（你自己写的普通 Python 函数）
# ============================================================
def get_weather(city: str) -> str:
    """查询城市天气（模拟数据）"""
    weather_data = {
        "北京": "晴天，25°C，空气质量良好",
        "上海": "多云，22°C，有轻雾",
        "深圳": "阵雨，28°C，湿度较大",
        "广州": "晴天，30°C，紫外线强",
    }
    return weather_data.get(city, f"暂无 {city} 的天气数据")


def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


# ============================================================
# 第 2 步：定义工具描述（告诉 AI 有哪些函数可用）
# ============================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气信息，包括温度、天气状况等",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海、深圳"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持加减乘除等运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如：123 * 456、(10 + 20) * 3"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# 函数名到函数对象的映射（方便后面根据名字调用）
available_functions = {
    "get_weather": get_weather,
    "calculator": calculator,
}


# ============================================================
# 第 3-6 步：完整的 Function Calling 流程
# ============================================================
def chat_with_tools(user_message: str):
    """
    带工具调用的完整对话流程
    """
    print(f"\n{'=' * 60}")
    print(f"用户: {user_message}")
    print(f"{'=' * 60}")

    messages = [
        {"role": "system", "content": "你是一个有帮助的助手，可以查询天气和做数学计算。"},
        {"role": "user", "content": user_message}
    ]

    # 第 3 步：发送请求，带上 tools 参数
    print("\n[1] 发送请求给 AI（带上工具描述）...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
    )

    assistant_message = response.choices[0].message

    # 第 4 步：检查 AI 是否要调用函数
    if assistant_message.tool_calls:
        # AI 决定调用函数
        print(f"[2] AI 决定调用工具！")

        # 把 AI 的决策加入消息历史
        messages.append(assistant_message)

        # 遍历所有工具调用（AI 可能一次调用多个）
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"    工具: {function_name}")
            print(f"    参数: {function_args}")

            # 第 5 步：执行函数
            func = available_functions[function_name]
            result = func(**function_args)
            print(f"    结果: {result}")

            # 把函数执行结果加入消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # 第 6 步：把结果发给 AI，让它生成最终回答
        print(f"\n[3] 把工具结果发给 AI，生成最终回答...")
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        final_answer = final_response.choices[0].message.content
        print(f"\nAI: {final_answer}")

    else:
        # AI 不需要调用函数，直接回答
        print(f"[2] AI 不需要调用工具，直接回答")
        print(f"\nAI: {assistant_message.content}")


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 测试 1：需要调用天气工具
    chat_with_tools("北京今天天气怎么样？")

    # 测试 2：需要调用计算器工具
    chat_with_tools("帮我算一下 123 乘以 456 等于多少")

    # 测试 3：不需要调用任何工具
    chat_with_tools("Python 是什么编程语言？")

    print("\n" + "=" * 60)
    print("[完成] 你已经理解了 Function Calling 的完整流程")
    print("=" * 60)
    print("""
核心要点：
1. 你定义工具函数 + 工具描述（tools 参数）
2. AI 根据用户问题和工具描述，自动判断是否需要调用
3. AI 不执行函数，只返回"我想调哪个函数、传什么参数"
4. 你的代码执行函数，把结果发回给 AI
5. AI 基于函数结果生成最终回答

这就是 Agent 的核心能力！
""")
