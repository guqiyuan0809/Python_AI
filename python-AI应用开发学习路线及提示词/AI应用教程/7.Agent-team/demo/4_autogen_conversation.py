"""
Demo 4: AutoGen 多 Agent 对话协作
使用 AutoGen 框架构建一个"代码开发对话"：
程序员 Agent 和 审查员 Agent 通过对话协作完成编程任务。

需要 API Key（配置下方变量）
安装依赖：pip install autogen-agentchat
"""

from autogen import ConversableAgent

# ============================================================
# API 配置（替换为你自己的配置）
# ============================================================

API_KEY = "your-api-key-here"
BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-4o-mini"

llm_config = {
    "model": MODEL,
    "api_key": API_KEY,
    "base_url": BASE_URL,
}

# ============================================================
# 定义 Agent
# ============================================================

coder = ConversableAgent(
    name="程序员",
    system_message=(
        "你是一位 Python 程序员。根据需求编写代码。"
        "代码要简洁、有注释、可运行。"
        "当审查员提出修改意见时，修改代码并重新提交。"
        "当审查员说'通过'时，回复 TERMINATE。"
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

reviewer = ConversableAgent(
    name="审查员",
    system_message=(
        "你是一位代码审查专家。审查程序员提交的代码。"
        "检查：代码正确性、可读性、边界情况处理。"
        "如果代码质量合格，回复'通过'。"
        "如果需要修改，给出具体的修改建议（最多 2 条）。"
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# ============================================================
# 运行对话
# ============================================================

if __name__ == "__main__":
    print("AutoGen 多 Agent 对话协作 Demo")
    print("=" * 50)
    print("模式：两人对话（程序员 ↔ 审查员）")
    print("流程：程序员写代码 → 审查员审查 → 来回修改直到通过")
    print("=" * 50)

    task = "写一个 Python 函数，实现二分查找算法。要求处理空列表和目标不存在的情况。"
    print(f"\n任务：{task}\n")
    print("-" * 50)

    chat_result = reviewer.initiate_chat(
        coder,
        message=f"请完成以下编程任务：\n{task}",
        max_turns=6,
    )

    print("\n" + "=" * 50)
    print("对话结束")
    print(f"总轮次：{len(chat_result.chat_history)}")
    print("=" * 50)
