"""
Demo 2: ReAct Agent（推理 + 行动）
目标：实现带有显式思考过程的 Agent

和 Demo 1 的区别：
- Demo 1：AI 内部思考，你看不到过程
- 这里：AI 每步都说出思考过程（Thought），方便理解和调试

ReAct 三要素：
  Thought（思考）→ Action（行动）→ Observation（观察）→ 循环

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
# 工具函数
# ============================================================
def get_weather(city: str) -> str:
    """查询城市天气"""
    data = {
        "北京": "晴天，35°C，空气质量良好",
        "上海": "多云，22°C，有轻雾",
        "深圳": "阵雨，28°C，湿度较大",
        "广州": "晴天，30°C，紫外线强",
        "成都": "阴天，18°C，有小雨",
    }
    return data.get(city, f"暂无 {city} 的天气数据")


def get_population(city: str) -> str:
    """查询城市人口"""
    data = {
        "北京": "常住人口 2170 万",
        "上海": "常住人口 2490 万",
        "深圳": "常住人口 1760 万",
        "广州": "常住人口 1880 万",
        "成都": "常住人口 2120 万",
    }
    return data.get(city, f"暂无 {city} 的人口数据")


def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "不支持的表达式"
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
            "description": "查询指定城市的常住人口数量",
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
    }
]

available_functions = {
    "get_weather": get_weather,
    "get_population": get_population,
    "calculator": calculator,
}


# ============================================================
# ReAct Agent 核心
# ============================================================

# ReAct 的关键：在 System Prompt 中要求 AI 说出思考过程
REACT_SYSTEM_PROMPT = """你是一个智能助手，使用 ReAct 模式工作。

每次回答前，你必须先说出你的思考过程。格式：

当你需要使用工具时：
先在 content 中说出你的思考（为什么要用这个工具），然后调用工具。

当你准备给出最终回答时：
先总结你的思考过程，然后给出回答。

重要规则：
- 每一步都要解释你在想什么
- 获得足够信息后，立即给出最终回答，不要多余的工具调用
- 如果工具返回错误，说明你的判断并尝试其他方法"""


def react_agent(user_message: str, max_steps: int = 10):
    """
    ReAct Agent

    和 Demo 1 的 run_agent 结构一样，区别在于：
    1. System Prompt 要求 AI 说出思考过程
    2. 打印时区分 Thought / Action / Observation
    3. 检测重复调用，防止无限循环
    """
    print(f"\n{'=' * 60}")
    print(f"用户: {user_message}")
    print(f"{'=' * 60}")

    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    # 用于检测重复调用
    recent_calls = []
    total_tokens = 0

    for step in range(1, max_steps + 1):
        print(f"\n━━━ Step {step} ━━━")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message

        # 统计 Token
        if response.usage:
            total_tokens += response.usage.total_tokens

        # Thought: AI 的思考过程（content 部分）
        if message.content:
            print(f"Thought: {message.content}")

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # Action: 调用工具
                call_signature = f"{func_name}({func_args})"
                print(f"Action:  {call_signature}")

                # 检测重复调用
                if call_signature in recent_calls[-3:]:
                    print(f"⚠️ 检测到重复调用，强制结束")
                    return "Agent 检测到重复操作，已停止。请重新描述你的需求。"
                recent_calls.append(call_signature)

                # 执行工具
                func = available_functions.get(func_name)
                if func:
                    result = func(**func_args)
                else:
                    result = f"未知工具: {func_name}"

                # Observation: 工具结果
                print(f"Observation: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            # 最终回答
            print(f"\nAnswer: {message.content}")
            print(f"\n[统计] 总步数: {step}, 总 Token: {total_tokens}")
            return message.content

    print(f"\n达到最大步数 ({max_steps})，强制结束")
    print(f"[统计] 总 Token: {total_tokens}")
    return None


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 测试 1：简单工具调用（观察 Thought 过程）
    react_agent("北京天气怎么样？")

    # 测试 2：多步推理（需要查多个数据再比较）
    react_agent("比较北京和上海，哪个城市更热？人口更多？")

    # 测试 3：需要计算的任务
    react_agent("北京和上海的人口加起来是多少万？")

    # 测试 4：不需要工具
    react_agent("什么是 ReAct 模式？")

    print("\n" + "=" * 60)
    print("完成！ReAct Agent 的核心要点")
    print("=" * 60)
    print("""
ReAct vs 普通 Agent：

普通 Agent（Demo 1）：
  AI 内部思考（你看不到）→ 调用工具 → 回答

ReAct Agent（本 Demo）：
  Thought（AI 说出思考）→ Action（调用工具）→ Observation（看结果）→ 循环

ReAct 的价值：
1. 可解释：你能看到 AI 每步在想什么
2. 可调试：出了问题，看 Thought 就知道 AI 哪里想错了
3. 更准确：让 AI "说出来" 能提高推理质量（思维链效应）
4. 能纠错：AI 看到错误结果后，会在 Thought 中调整策略
""")
