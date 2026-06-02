"""
Demo 2: Tools 工具定义
目标：掌握各种姿势写工具，让 Agent 用得准用得稳

演示内容：
1. 简单 @tool 装饰器
2. 多参数 + Pydantic 验证
3. 自定义 Tool 名字和描述
4. 错误处理（返回字符串 vs raise）
5. 用内置工具（DuckDuckGo 搜索）

不需要 API Key（只演示工具定义，不跑 Agent）
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional


# ============================================================
# 1. 最简单：@tool 装饰器
# ============================================================
print("=" * 60)
print("1. 最简单的 @tool")
print("=" * 60)


@tool
def get_weather(city: str) -> str:
    """查询某个城市的实时天气。

    Args:
        city: 城市名称，例如"北京"
    """
    return f"{city} 25 度，晴"


# 看看 LangChain 自动生成了什么
print(f"name:        {get_weather.name}")
print(f"description: {get_weather.description}")
print(f"args:        {get_weather.args}")

# 手动调用测试
result = get_weather.invoke({"city": "北京"})
print(f"调用结果:    {result}")


# ============================================================
# 2. 多参数 + 类型注解
# ============================================================
print("\n" + "=" * 60)
print("2. 多参数 + 类型注解（基础用法）")
print("=" * 60)


@tool
def search_flight(from_city: str, to_city: str, date: str) -> str:
    """查询机票信息。

    Args:
        from_city: 出发城市，例如"北京"
        to_city: 到达城市，例如"上海"
        date: 出发日期，格式必须为 YYYY-MM-DD，例如"2026-05-01"
    """
    return f"找到 {from_city}→{to_city} {date} 的航班 5 个"


print(f"args: {search_flight.args}")
print(f"调用结果: {search_flight.invoke({'from_city': '北京', 'to_city': '上海', 'date': '2026-05-01'})}")

# ⚠️ 关键：date 的描述里写了"格式 YYYY-MM-DD"，否则 LLM 可能传"明天"、"五一"


# ============================================================
# 3. Pydantic 参数模式（复杂参数推荐）
# ============================================================
print("\n" + "=" * 60)
print("3. Pydantic 参数模式（更严格的参数定义）")
print("=" * 60)


class FlightSearchInput(BaseModel):
    """机票搜索参数"""
    from_city: str = Field(..., description="出发城市，例如'北京'")
    to_city: str = Field(..., description="到达城市，例如'上海'")
    date: str = Field(..., description="日期，格式 YYYY-MM-DD")
    cabin: Optional[str] = Field(default="经济舱", description="舱位类型，可选'经济舱'/'商务舱'/'头等舱'")


@tool(args_schema=FlightSearchInput)
def search_flight_v2(from_city: str, to_city: str, date: str, cabin: str = "经济舱") -> str:
    """查询机票信息（带舱位选择）"""
    return f"找到 {from_city}→{to_city} {date} {cabin}航班"


print(f"args: {search_flight_v2.args}")
# 你会看到每个参数都带了 description，比纯类型注解更详细


# ============================================================
# 4. 自定义 Tool 名字和描述
# ============================================================
print("\n" + "=" * 60)
print("4. 自定义名字和描述")
print("=" * 60)


# 装饰器参数可以覆盖函数名和 docstring
@tool("weather_query", description="查询天气，输入城市名返回当前天气状况")
def _get_weather_internal(city: str) -> str:
    """这个 docstring 会被覆盖"""
    return f"{city} 天气数据"


print(f"name:        {_get_weather_internal.name}")          # 'weather_query'
print(f"description: {_get_weather_internal.description}")   # 装饰器里的


# ============================================================
# 5. 错误处理：返回字符串而不是 raise
# ============================================================
print("\n" + "=" * 60)
print("5. 错误处理（重要）")
print("=" * 60)


@tool
def divide(a: float, b: float) -> str:
    """两个数相除。

    Args:
        a: 被除数
        b: 除数（不能为 0）
    """
    if b == 0:
        # ✅ 推荐：返回错误字符串，让 LLM 看到错误原因后能自我修正
        return "错误：除数不能为 0，请提供非零的除数"
    return str(a / b)


print(f"正常调用: {divide.invoke({'a': 10, 'b': 2})}")
print(f"错误调用: {divide.invoke({'a': 10, 'b': 0})}")
# Agent 看到 "错误：除数不能为 0" 后，会主动修正参数再调一次


# ============================================================
# 6. 复杂工具：模拟一个查订单的 Tool
# ============================================================
print("\n" + "=" * 60)
print("6. 真实场景：查订单工具")
print("=" * 60)

# 模拟订单数据库
FAKE_ORDERS = {
    "ORD-12345": {"status": "已发货", "amount": 199.0, "item": "蓝牙耳机"},
    "ORD-67890": {"status": "已签收", "amount": 599.0, "item": "无线键盘"},
}


@tool
def get_order_info(order_id: str) -> str:
    """查询订单详细信息，包括状态、金额、商品名称。

    Args:
        order_id: 订单号，例如 "ORD-12345"。订单号一般以 "ORD-" 开头。
    """
    order = FAKE_ORDERS.get(order_id)
    if not order:
        return f"未找到订单 {order_id}，请确认订单号是否正确"

    return (
        f"订单 {order_id} 信息：\n"
        f"  - 商品: {order['item']}\n"
        f"  - 金额: {order['amount']} 元\n"
        f"  - 状态: {order['status']}"
    )


print(get_order_info.invoke({"order_id": "ORD-12345"}))
print()
print(get_order_info.invoke({"order_id": "ORD-99999"}))   # 不存在的订单


# ============================================================
# 7. 内置工具：DuckDuckGo 搜索（可选）
# ============================================================
print("\n" + "=" * 60)
print("7. 内置工具：DuckDuckGo 搜索 (可选)")
print("=" * 60)

# 注释掉是因为需要安装额外依赖：pip install duckduckgo-search
# from langchain_community.tools import DuckDuckGoSearchRun
# search = DuckDuckGoSearchRun()
# print(search.invoke("Python 是什么"))

print("（需要 pip install duckduckgo-search）")
print("内置工具用法：")
print("  from langchain_community.tools import DuckDuckGoSearchRun")
print("  search = DuckDuckGoSearchRun()")
print("  search.invoke('查询关键词')")


# ============================================================
# 8. 工具列表 —— 这就是要传给 Agent 的东西
# ============================================================
print("\n" + "=" * 60)
print("8. 组装工具列表")
print("=" * 60)

tools = [
    get_weather,
    search_flight,
    search_flight_v2,
    divide,
    get_order_info,
]

print(f"共有 {len(tools)} 个工具：")
for t in tools:
    print(f"  - {t.name}: {t.description.split(chr(10))[0]}")

# 这个 tools 列表就是 Demo 3 创建 Agent 时要传的


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
工具定义的关键点：

1. ✅ @tool 装饰器自动从 docstring + 类型注解生成工具描述
2. ✅ 复杂参数用 Pydantic + args_schema，每个字段加 description
3. ✅ docstring 第一句要说清楚使用场景，参数要给格式示例
4. ✅ 错误用字符串返回，不要 raise，让 LLM 能自我修正
5. ✅ 一个工具只做一件事，不要塞太多功能
6. ✅ 工具数量控制在 5~10 个内，太多 LLM 选不准

实用技巧：
- 用 tool.invoke({...}) 单独测试工具是否正常
- 用 tool.name / tool.description / tool.args 看 LLM 看到的样子

下一步：把工具组装到 Agent 里跑起来 → Demo 3
""")
