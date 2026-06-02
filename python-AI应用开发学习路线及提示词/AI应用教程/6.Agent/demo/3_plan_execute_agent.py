"""
Demo 3: Plan-and-Execute Agent（先规划再执行）
目标：理解另一种 Agent 架构 —— 先制定完整计划，再逐步执行

和 ReAct 的区别：
- ReAct：走一步看一步（边想边做）
- Plan-and-Execute：先想好所有步骤，再一步步做

适用场景：步骤明确的任务（数据分析、报告生成等）

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
        "北京": "晴天，35°C", "上海": "多云，22°C",
        "深圳": "阵雨，28°C", "广州": "晴天，30°C", "成都": "阴天，18°C",
    }
    return data.get(city, f"暂无 {city} 的天气数据")


def get_population(city: str) -> str:
    """查询城市人口"""
    data = {
        "北京": "2170万", "上海": "2490万",
        "深圳": "1760万", "广州": "1880万", "成都": "2120万",
    }
    return data.get(city, f"暂无 {city} 的人口数据")


def get_gdp(city: str) -> str:
    """查询城市 GDP"""
    data = {
        "北京": "4.16万亿", "上海": "4.72万亿",
        "深圳": "3.46万亿", "广州": "2.88万亿", "成都": "2.21万亿",
    }
    return data.get(city, f"暂无 {city} 的 GDP 数据")


def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

# ============================================================
# 工具函数
# ============================================================
tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "查询城市天气", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_population", "description": "查询城市人口", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_gdp", "description": "查询城市GDP", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
]



# ============================================================
# 工具注册表：函数名（字符串）→ 函数对象的映射
# AI 返回字符串 "get_weather"，通过此表找到真正的函数执行
# ============================================================
available_functions = {          
    "get_weather": get_weather,     
    "get_population": get_population,
    "get_gdp": get_gdp,
    "calculator": calculator,
}


# ============================================================
# Plan-and-Execute Agent
# ============================================================

def plan_phase(user_message: str) -> list:
    """
    第一阶段：规划
    让 AI 制定执行计划，返回步骤列表
    """
    print("\n📋 规划阶段")
    print("-" * 40)

    plan_prompt = f"""用户任务：{user_message}    # 总提示词 = 用户输入（动态）+ 开发者写的格式要求（固定）


请制定一个执行计划。你可以使用以下工具：
- get_weather(city): 查询城市天气
- get_population(city): 查询城市人口
- get_gdp(city): 查询城市GDP
- calculator(expression): 数学计算

请用 JSON 格式返回计划，格式如下：
{{
    "steps": [
        {{"step": 1, "description": "做什么", "tool": "工具名", "args": {{"参数名": "参数值"}}}},
        {{"step": 2, "description": "做什么", "tool": "工具名", "args": {{"参数名": "参数值"}}}},
        ...
        {{"step": N, "description": "汇总分析并给出最终回答", "tool": null, "args": null}}
    ]
}}

最后一步必须是汇总分析（tool 为 null），用来生成最终回答。
只返回 JSON，不要其他内容。"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个任务规划专家，擅长把复杂任务拆解成清晰的步骤。"},
            {"role": "user", "content": plan_prompt}
        ],
        stream=True  # 流式输出
    )

    # 流式接收并拼接完整文本（规划阶段需要完整 JSON 才能解析）
    print("  ", end="", flush=True)
    plan_text = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            plan_text += delta
    print()  # 换行

    # AI 输出不可控，有时会把 JSON 包在 ```json ... ``` 代码块里，需要先剥掉
    plan_text = plan_text.strip()
    if plan_text.startswith("```"):
        plan_text = plan_text.split("\n", 1)[1]   # 去掉第一行 ```json
        plan_text = plan_text.rsplit("```", 1)[0]  # 去掉最后的 ```

    try:
        plan = json.loads(plan_text)  # 字符串 → Python 字典
        steps = plan["steps"]         # 取出步骤列表
    except (json.JSONDecodeError, KeyError):
        # AI 返回格式不对时的兜底，保证程序不崩
        steps = [{"step": 1, "description": "直接回答", "tool": None, "args": None}]

    # 打印计划
    for s in steps:
        tool_info = f" → {s['tool']}({s.get('args', '')})" if s.get('tool') else ""
        print(f"  Step {s['step']}: {s['description']}{tool_info}")

    return steps


def execute_phase(steps: list, user_message: str) -> str:
    """
    第二阶段：执行
    按照计划逐步执行，收集结果
    """
    print("\n🚀 执行阶段")
    print("-" * 40)

    results = []

    for step in steps:
        step_num = step["step"]
        description = step["description"]
        tool_name = step.get("tool")
        tool_args = step.get("args") or {}

        print(f"\n  执行 Step {step_num}: {description}")

        if tool_name and tool_name in available_functions:
            # 调用工具
            func = available_functions[tool_name]
            result = func(**tool_args)
            print(f"  结果: {result}")
            results.append({"step": step_num, "description": description, "result": result})
        elif tool_name:
            print(f"  ⚠️ 未知工具: {tool_name}")
            results.append({"step": step_num, "description": description, "result": f"工具 {tool_name} 不存在"})
        else:
            # 最后一步：汇总（不需要工具）
            results.append({"step": step_num, "description": description, "result": "汇总步骤"})

    # 第三阶段：让 AI 基于所有结果生成最终回答
    print("\n📝 汇总阶段")
    print("-" * 40)

    results_text = "\n".join([
        f"Step {r['step']} ({r['description']}): {r['result']}"
        for r in results if r['result'] != "汇总步骤"
    ])

    summary_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一个分析专家，请根据收集到的数据给出清晰的分析和回答。"},
            {"role": "user", "content": f"用户问题：{user_message}\n\n收集到的数据：\n{results_text}\n\n请基于以上数据给出完整的回答。"}
        ],
        stream=True  # 流式输出，边生成边打印
    )

    # 流式打印最终回答
    answer = ""
    for chunk in summary_response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            answer += delta
    print()  # 换行
    return answer


def plan_execute_agent(user_message: str):
    """
    Plan-and-Execute Agent 主函数

    流程：
    1. Planning：让 AI 制定计划
    2. Execution：按计划逐步执行工具
    3. Summary：让 AI 汇总结果，生成最终回答
    """
    print(f"\n{'=' * 60}")
    print(f"用户: {user_message}")
    print(f"{'=' * 60}")

    # 第一阶段：规划
    steps = plan_phase(user_message)

    # 第二阶段：执行
    answer = execute_phase(steps, user_message)

    print(f"\n{'=' * 60}")

    return answer


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 测试 1：多城市综合比较（需要多步数据收集）
    plan_execute_agent("比较北京、上海、深圳三个城市的天气、人口和GDP，告诉我哪个城市综合实力最强")

    # 测试 2：简单任务
    plan_execute_agent("北京天气怎么样？")

    print("\n" + "=" * 60)
    print("完成！Plan-and-Execute 的核心要点")
    print("=" * 60)
    print("""
Plan-and-Execute vs ReAct：

ReAct（Demo 2）：
  想 → 做 → 看 → 想 → 做 → 看 → ... → 回答
  优点：灵活，能根据中间结果调整
  缺点：路径不可预测# 解析 JSON（处理可能的 markdown 代码块包裹）
    plan_text = plan_text.strip()
    if plan_text.startswith("```"):
        plan_text = plan_text.split("\n", 1)[1]  # 去掉第一行 ```json
        plan_text = plan_text.rsplit("```", 1)[0]  # 去掉最后的 ```

    try:
        plan = json.loads(plan_text)
        steps = plan["steps"]
    except (json.JSONDecodeError, KeyError):
        print(f"  计划解析失败，使用默认计划")
        steps = [{"step": 1, "description": "直接回答", "tool": None, "args": None}]

Plan-and-Execute（本 Demo）：
  先想好所有步骤 → 执行步骤1 → 执行步骤2 → ... → 汇总回答
  优点：步骤清晰，可预测
  缺点：不够灵活，某步失败可能影响后续

选择建议：
- 简单任务、需要灵活应对 → ReAct
- 步骤明确、流程固定 → Plan-and-Execute
- 实际项目中可以结合使用（自适应规划）
""")
