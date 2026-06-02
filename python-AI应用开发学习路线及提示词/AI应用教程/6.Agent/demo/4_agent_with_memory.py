"""
Demo 4: 带记忆的 Agent
目标：实现一个能记住用户信息、支持多轮对话的 Agent

在 Demo 1 的基础上加入：
1. 工作记忆（滑动窗口，控制 Token）
2. 长期记忆（用户偏好持久化）
3. 任务记忆（跟踪多步任务进度）

需要 API Key
"""

import json
import os
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
API_KEY = "sk-2e56e7e02258424eaf6afe699b054031"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


MEMORY_DIR = "agent_memory"


# ============================================================
# 工具函数
# ============================================================
def get_weather(city: str) -> str:
    """查询城市天气"""
    data = {
        "北京": "晴天，35°C", "上海": "多云，22°C",
        "深圳": "阵雨，28°C", "广州": "晴天，30°C",
    }
    return data.get(city, f"暂无 {city} 的天气数据")


def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "查询城市天气", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
]

available_functions = {
    "get_weather": get_weather,
    "calculator": calculator,
}


# ============================================================
# 记忆管理
# ============================================================
class AgentMemory:
    """
    Agent 记忆管理器

    三种记忆：
    1. 短期记忆：当前对话的 messages 列表
    2. 工作记忆：滑动窗口，只保留最近 N 轮
    3. 长期记忆：用户偏好，存到文件
    """

    def __init__(self, user_id: str, max_rounds: int = 5):
        self.user_id = user_id
        self.max_rounds = max_rounds
        self.messages = []  # 短期记忆
        self.memory_file = os.path.join(MEMORY_DIR, f"{user_id}.json")

        # 确保目录存在
        os.makedirs(MEMORY_DIR, exist_ok=True)

        # 加载长期记忆，构建 system prompt
        self._init_system_prompt()

    def _init_system_prompt(self):
        """初始化 system prompt（包含长期记忆）"""
        long_term = self.load_long_term()

        prompt = "你是一个智能助手，可以查天气和做计算。\n"

        if long_term:
            prompt += "\n你记得这个用户的以下信息：\n"
            for key, value in long_term.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n请根据这些信息个性化回答。"

        prompt += "\n\n如果用户告诉你他的个人信息（名字、喜好、城市等），请记住。"

        self.messages = [{"role": "system", "content": prompt}]

    # --- 工作记忆（滑动窗口）---

    def add_message(self, role: str, content):
        """添加消息，自动裁剪"""
        if isinstance(content, str):
            self.messages.append({"role": role, "content": content})
        else:
            # tool_calls 等复杂消息直接添加
            self.messages.append(content) if not isinstance(content, dict) else self.messages.append(content)

        self._trim_messages()

    def add_assistant_message(self, message):
        """添加 AI 回复（可能包含 tool_calls）"""
        self.messages.append(message)
        self._trim_messages()

    def add_tool_result(self, tool_call_id: str, content: str):
        """添加工具结果"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })

    def _trim_messages(self):
        """
        滑动窗口裁剪
        保留 system prompt + 最近 N 轮对话
        """
        if len(self.messages) <= 1:
            return

        system_msg = self.messages[0]
        history = self.messages[1:]

        # 粗略估算：每轮约 2-4 条消息（user + assistant + 可能的 tool）
        max_messages = self.max_rounds * 4

        if len(history) > max_messages:
            history = history[-max_messages:]

        self.messages = [system_msg] + history

    # --- 长期记忆（文件持久化）---

    def save_long_term(self, key: str, value: str):
        """保存长期记忆"""
        memory = self.load_long_term()
        memory[key] = value

        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)

    def load_long_term(self) -> dict:
        """加载长期记忆"""
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def get_messages(self) -> list:
        """获取当前 messages"""
        return self.messages


# ============================================================
# 带记忆的 Agent
# ============================================================

def extract_user_info(user_message: str, ai_response: str) -> dict:
    """
    从对话中提取用户信息（简单的关键词匹配）
    实际项目中可以让 AI 来提取
    """
    info = {}
    message = user_message.lower()

    # 简单的模式匹配
    if "我叫" in user_message:
        name = user_message.split("我叫")[-1].strip().split("，")[0].split(",")[0].split("。")[0]
        if name:
            info["name"] = name

    if "喜欢" in user_message:
        like = user_message.split("喜欢")[-1].strip().split("，")[0].split(",")[0].split("。")[0]
        if like:
            info["preference"] = like

    if "住在" in user_message or "在" in user_message and "住" in user_message:
        for city in ["北京", "上海", "深圳", "广州", "成都"]:
            if city in user_message:
                info["city"] = city
                break

    return info


def agent_with_memory(user_id: str = "user_001"):
    """
    带记忆的 Agent（多轮对话模式）

    特点：
    1. 支持多轮对话（不是一问一答就结束）
    2. 自动提取并保存用户信息（长期记忆）
    3. 滑动窗口控制 Token 成本（工作记忆）
    4. 下次启动时还能记住用户信息
    """
    memory = AgentMemory(user_id=user_id, max_rounds=5)  # 初始化记忆管理器，最多保留 5 轮对话

    # ---- 启动时加载长期记忆（从本地 JSON 文件读取） ----
    long_term = memory.load_long_term()
    if long_term:
        print(f"📝 已加载用户 {user_id} 的长期记忆: {long_term}")
    else:
        print(f"📝 用户 {user_id} 暂无长期记忆")

    print(f"\n{'=' * 60}")
    print("带记忆的 Agent（输入 'quit' 退出）")
    print(f"{'=' * 60}")

    # ---- 外层循环：多轮对话，持续等待用户输入 ----
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！你的信息已保存。")
            break

        if not user_input:
            continue

        memory.add_message("user", user_input)  # 用户消息存入短期记忆

        # ---- 内层循环：Agent 单轮执行，可能多次工具调用 ----
        max_steps = 5
        for step in range(max_steps):
            # 先用非流式请求，判断 AI 是要调工具还是给最终回答
            response = client.chat.completions.create(
                model=MODEL,
                messages=memory.get_messages(),  # 每次带上完整记忆（system + 历史对话）
                tools=tools,
            )

            message = response.choices[0].message

            if message.tool_calls:
                memory.add_assistant_message(message)  # AI 的工具调用请求也要存入记忆

                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    print(f"  [工具] {func_name}({func_args})")

                    func = available_functions.get(func_name)  # 从注册表查找函数
                    result = func(**func_args) if func else f"未知工具: {func_name}"
                    print(f"  [结果] {result}")

                    memory.add_tool_result(tool_call.id, result)  # 工具结果也存入记忆
            else:
                # ---- AI 要给最终回答了，用流式输出 ----
                stream_response = client.chat.completions.create(
                    model=MODEL,
                    messages=memory.get_messages(),
                    stream=True,  # 流式输出，边生成边打印
                )

                print(f"\nAI: ", end="", flush=True)
                ai_response = ""
                for chunk in stream_response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        print(delta, end="", flush=True)
                        ai_response += delta
                print()  # 换行

                memory.add_message("assistant", ai_response)

                # ---- 从对话中提取用户信息，写入长期记忆（持久化到本地文件） ----
                user_info = extract_user_info(user_input, ai_response)
                for key, value in user_info.items():
                    memory.save_long_term(key, value)
                    print(f"  💾 已记住: {key} = {value}")

                break  # 有最终回答了，退出内层循环，等待下一轮输入

        # 调试用：显示当前记忆状态
        print(f"  [记忆] messages 长度: {len(memory.get_messages())}, 长期记忆: {memory.load_long_term()}")


# ============================================================
# 非交互式测试（不需要手动输入）
# ============================================================
def test_memory_agent():
    """
    自动测试带记忆的 Agent
    模拟多轮对话，验证记忆功能
    """
    print(f"\n{'=' * 60}")
    print("自动测试：带记忆的 Agent")
    print(f"{'=' * 60}")

    user_id = "test_user"
    memory = AgentMemory(user_id=user_id, max_rounds=5)

    # 预设的测试对话，模拟用户依次输入（替代手动 input）
    # 前两句提供个人信息 → 测试长期记忆写入
    # 第三句反问名字 → 测试短期记忆（AI 能否记住上下文）
    # 后两句调工具 → 测试 Agent 工具调用 + 记忆共存
    test_conversations = [
        "我叫张三，我住在北京",      # → 触发 extract_user_info 提取 name、city
        "我喜欢 Python",            # → 触发提取 preference
        "我叫什么名字？",            # → AI 应从短期记忆中回忆出"张三"
        "北京天气怎么样？",           # → 触发 get_weather 工具调用
        "帮我算一下 100 * 3.14",     # → 触发 calculator 工具调用
    ]

    # 外层循环：遍历预设对话（等价于 agent_with_memory 的 while True 等待用户输入）
    for user_input in test_conversations:
        print(f"\n你: {user_input}")
        memory.add_message("user", user_input)  # 存入短期记忆

        # 内层循环：Agent 执行任务（工具调用可能多步）
        for step in range(5):
            response = client.chat.completions.create(
                model=MODEL,
                messages=memory.get_messages(),  # 带上完整记忆
                tools=tools,
            )

            message = response.choices[0].message

            if message.tool_calls:
                memory.add_assistant_message(message)
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    func = available_functions.get(func_name)
                    result = func(**func_args) if func else "未知工具"
                    print(f"  [工具] {func_name} → {result}")
                    memory.add_tool_result(tool_call.id, result)
            else:
                # AI 给出最终回答，流式输出
                stream_response = client.chat.completions.create(
                    model=MODEL,
                    messages=memory.get_messages(),
                    stream=True,
                )

                print(f"AI: ", end="", flush=True)
                ai_response = ""
                for chunk in stream_response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        print(delta, end="", flush=True)
                        ai_response += delta
                print()

                memory.add_message("assistant", ai_response)

                # 提取用户信息 → 写入长期记忆（持久化到本地文件）
                user_info = extract_user_info(user_input, ai_response)
                for key, value in user_info.items():
                    memory.save_long_term(key, value)
                    print(f"  💾 已记住: {key} = {value}")
                break  # 有回答了，进入下一轮对话

    # 显示最终记忆状态
    print(f"\n{'=' * 60}")
    print(f"最终长期记忆: {memory.load_long_term()}")
    print(f"messages 长度: {len(memory.get_messages())}")
    print(f"{'=' * 60}")

    # 清理测试文件
    test_file = os.path.join(MEMORY_DIR, f"{user_id}.json")
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists(MEMORY_DIR) and not os.listdir(MEMORY_DIR):
        os.rmdir(MEMORY_DIR)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    print("选择模式：")
    print("1. 自动测试（不需要手动输入）")
    print("2. 交互模式（手动对话）")

    choice = input("输入 1 或 2: ").strip()

    if choice == "2":
        agent_with_memory()
    else:
        test_memory_agent()

    print("""
核心要点：
1. AgentMemory 类统一管理三种记忆
2. 滑动窗口（_trim_messages）控制 Token 成本
3. 长期记忆（save/load_long_term）让 Agent 跨会话记住用户
4. 用户信息自动提取并保存
5. 下次启动时，长期记忆会加入 system prompt
""")
