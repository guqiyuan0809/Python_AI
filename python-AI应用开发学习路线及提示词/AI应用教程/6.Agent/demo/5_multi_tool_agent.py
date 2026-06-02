"""
Demo 5: 多工具复杂任务 Agent
目标：综合运用前面所有知识，实现一个能处理复杂任务的 Agent

特点：
1. 多个工具（7 个），覆盖不同场景
2. ReAct 模式（带思考过程）
3. 完整的日志记录
4. 错误处理和重试
5. Token 统计

这是本章最完整的 Agent 实现，接近实际项目水平。

需要 API Key
"""

import json
import time
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
API_KEY = "sk-2e56e7e02258424eaf6afe699b054031"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============================================================
# 工具函数（7 个工具，模拟真实场景）
# ============================================================
def get_weather(city: str) -> str:
    """查询城市天气"""
    data = {
        "北京": "晴天，35°C，空气质量良好，紫外线强",
        "上海": "多云，22°C，有轻雾，空气质量中等",
        "深圳": "阵雨，28°C，湿度 85%，注意带伞",
        "广州": "晴天，30°C，紫外线强，注意防晒",
        "成都": "阴天，18°C，有小雨，适合室内活动",
    }
    return data.get(city, f"暂无 {city} 的天气数据")


def get_city_info(city: str) -> str:
    """查询城市综合信息"""
    data = {
        "北京": "首都，人口2170万，GDP 4.16万亿，著名景点：故宫、长城、天坛",
        "上海": "直辖市，人口2490万，GDP 4.72万亿，著名景点：外滩、东方明珠、迪士尼",
        "深圳": "经济特区，人口1760万，GDP 3.46万亿，著名景点：世界之窗、大梅沙",
        "广州": "省会，人口1880万，GDP 2.88万亿，著名景点：广州塔、白云山",
        "成都": "省会，人口2120万，GDP 2.21万亿，著名景点：宽窄巷子、大熊猫基地",
    }
    return data.get(city, f"暂无 {city} 的信息")


def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "安全限制：只支持数字和基本运算符"
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"


def search_knowledge(query: str) -> str:
    """搜索知识库"""
    knowledge = {
        "python": "Python 是高级编程语言，适合 Web、数据分析、AI。主流框架：Django、FastAPI、Flask",
        "fastapi": "FastAPI 是高性能 Python Web 框架，支持异步，自动生成 OpenAPI 文档，适合 AI 应用后端",
        "agent": "Agent（智能体）是能自主完成任务的 AI 程序，核心：LLM + 工具 + 循环决策 + 记忆",
        "rag": "RAG（检索增强生成）让 AI 基于外部文档回答问题，流程：检索 → 增强 → 生成",
        "langchain": "LangChain 是 LLM 应用开发框架，提供工具调用、记忆、Agent 等组件",
        "docker": "Docker 是容器化平台，用于打包和部署应用，确保环境一致性",
    }
    query_lower = query.lower()
    results = []
    for key, value in knowledge.items():
        if key in query_lower or query_lower in key:
            results.append(value)
    return "\n".join(results) if results else f"知识库中没有找到关于 '{query}' 的信息"


def compare_items(item1: str, item2: str, aspect: str) -> str:
    """比较两个事物"""
    return f"比较 {item1} 和 {item2}（{aspect}方面）：这需要基于已收集的数据进行分析，请结合之前查询的信息给出比较结论。"


def get_recommendation(category: str, preference: str) -> str:
    """获取推荐"""
    recommendations = {
        "旅游": {
            "热门": "推荐：北京（历史文化）、上海（现代都市）、成都（美食休闲）",
            "小众": "推荐：大理（文艺）、泉州（古城）、阿那亚（海边）",
            "亲子": "推荐：上海迪士尼、广州长隆、成都大熊猫基地",
        },
        "学习": {
            "入门": "推荐：Python 基础 → FastAPI → AI 应用开发",
            "进阶": "推荐：LangChain → RAG 项目 → Agent 开发",
            "求职": "推荐：做 2-3 个完整项目，重点 RAG + Agent",
        },
    }
    cat_data = recommendations.get(category, {})
    return cat_data.get(preference, f"暂无 {category}-{preference} 的推荐")


def get_current_time() -> str:
    """获取当前时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# ============================================================
# 工具注册
# ============================================================
tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "查询城市天气", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_city_info", "description": "查询城市综合信息（人口、GDP、景点等）", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "search_knowledge", "description": "搜索技术知识库", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "compare_items", "description": "比较两个事物的某个方面", "parameters": {"type": "object", "properties": {"item1": {"type": "string", "description": "第一个事物"}, "item2": {"type": "string", "description": "第二个事物"}, "aspect": {"type": "string", "description": "比较的方面"}}, "required": ["item1", "item2", "aspect"]}}},
    {"type": "function", "function": {"name": "get_recommendation", "description": "获取推荐（旅游、学习等）", "parameters": {"type": "object", "properties": {"category": {"type": "string", "description": "类别：旅游/学习"}, "preference": {"type": "string", "description": "偏好：热门/小众/亲子/入门/进阶/求职"}}, "required": ["category", "preference"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {}, "required": []}}},
]

available_functions = {
    "get_weather": get_weather,
    "get_city_info": get_city_info,
    "calculator": calculator,
    "search_knowledge": search_knowledge,
    "compare_items": compare_items,
    "get_recommendation": get_recommendation,
    "get_current_time": get_current_time,
}


# ============================================================
# Agent 日志
# ============================================================
class AgentLog:
    """简单的 Agent 日志"""

    def __init__(self):
        self.steps = []
        self.total_tokens = 0
        self.start_time = time.time()

    def log(self, step_type: str, content: str):
        elapsed = round(time.time() - self.start_time, 1)
        entry = {"time": elapsed, "type": step_type, "content": content}
        self.steps.append(entry)

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens

    def summary(self):
        elapsed = round(time.time() - self.start_time, 1)
        print(f"\n📊 执行摘要:")
        print(f"  总步数: {len([s for s in self.steps if s['type'] == 'ACTION'])}")
        print(f"  总 Token: {self.total_tokens}")
        print(f"  总耗时: {elapsed}s")


# ============================================================
# 多工具 Agent（ReAct 模式 + 日志 + 错误处理）
# ============================================================
SYSTEM_PROMPT = """你是一个功能强大的智能助手，使用 ReAct 模式工作。

你可以使用以下工具：
- get_weather: 查询城市天气
- get_city_info: 查询城市综合信息
- calculator: 数学计算
- search_knowledge: 搜索技术知识库
- compare_items: 比较两个事物
- get_recommendation: 获取推荐
- get_current_time: 获取当前时间

工作规则：
1. 每步先思考，再决定是否使用工具
2. 获得足够信息后立即给出最终回答
3. 如果工具返回错误，尝试其他方法
4. 回答要详细、有条理、基于实际数据"""


def multi_tool_agent(user_message: str, max_steps: int = 10):
    """
    多工具 Agent

    综合了前面所有 Demo 的特点：
    - ReAct 模式（显式思考）
    - 多工具选择
    - 日志记录
    - 重复检测
    - Token 统计
    """
    print(f"\n{'=' * 60}")
    print(f"🤖 多工具 Agent")
    print(f"用户: {user_message}")
    print(f"{'=' * 60}")

    log = AgentLog()
    log.log("START", user_message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    recent_calls = []

    for step in range(1, max_steps + 1):
        print(f"\n━━━ Step {step} ━━━")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
            )
        except Exception as e:
            print(f"  ❌ API 调用失败: {e}")
            log.log("ERROR", str(e))
            break

        message = response.choices[0].message

        if response.usage:
            log.add_tokens(response.usage.total_tokens)

        # Thought
        if message.content:
            print(f"  💭 Thought: {message.content}")
            log.log("THOUGHT", message.content)

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                call_sig = f"{func_name}({json.dumps(func_args, ensure_ascii=False)})"

                # 重复检测
                if call_sig in recent_calls[-3:]:
                    print(f"  ⚠️ 重复调用检测: {call_sig}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "重复调用，请直接基于已有信息回答"
                    })
                    log.log("DUPLICATE", call_sig)
                    continue

                recent_calls.append(call_sig)

                # Action
                print(f"  🔧 Action: {call_sig}")
                log.log("ACTION", call_sig)

                func = available_functions.get(func_name)
                if func:
                    try:
                        result = func(**func_args)
                    except Exception as e:
                        result = f"工具执行错误: {e}"
                else:
                    result = f"未知工具: {func_name}"

                # Observation
                print(f"  👁️ Observation: {result}")
                log.log("OBSERVATION", result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            # 最终回答，用流式输出
            stream_response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
            )

            print(f"\n  ✅ Answer: ", end="", flush=True)
            answer = ""
            for chunk in stream_response:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    answer += delta
            print()  # 换行

            log.log("ANSWER", answer)
            log.summary()
            return answer

    print("\n  ⏰ 达到最大步数，强制结束")
    log.summary()
    return None


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 测试 1：复杂的多工具任务
    multi_tool_agent("我想去旅游，帮我比较北京和成都，包括天气、城市信息，然后给我一个推荐")

    # 测试 2：技术问题 + 知识检索
    multi_tool_agent("我想学 AI 应用开发，现在是什么时间？帮我推荐一个学习路线")

    # 测试 3：数据收集 + 计算
    multi_tool_agent("查一下北京和上海的GDP，算一下两个城市GDP的总和")

    # 测试 4：简单问题
    multi_tool_agent("你好，现在几点了？")

    print("\n" + "=" * 60)
    print("完成！这是本章最完整的 Agent 实现")
    print("=" * 60)
    print("""
本 Demo 综合了所有 Agent 知识：

1. ReAct 模式：Thought → Action → Observation 循环
2. 多工具选择：7 个工具，AI 自主选择
3. 日志记录：每步都有记录，方便调试
4. 重复检测：防止 Agent 无限循环
5. 错误处理：API 失败和工具执行错误都有处理
6. Token 统计：监控成本

这就是一个接近生产级别的 Agent 骨架。
实际项目中，你需要：
- 把模拟工具换成真实的 API 调用
- 加入记忆管理（参考 Demo 4）
- 用 FastAPI 包装成 Web 服务
- 加入用户认证和权限控制
""")
