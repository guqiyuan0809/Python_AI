"""
Demo 3: LangGraph 多 Agent 协作
使用 LangGraph 构建一个带条件路由的多 Agent 工作流：
研究员 → 写作者 → 审核者（审核不通过则回到写作者修改）

需要 API Key（配置下方变量）
安装依赖：pip install langgraph langchain-openai
"""

from typing import TypedDict, Annotated, Literal
import operator
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ============================================================
# API 配置（替换为你自己的配置）
# ============================================================

API_KEY = "your-api-key-here"
BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-4o-mini"

llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
)

# ============================================================
# 定义共享状态
# ============================================================

class TeamState(TypedDict):
    task: str
    research_result: str
    draft: str
    review_feedback: str
    review_passed: bool
    revision_count: int
    final_output: str


# ============================================================
# 定义节点（每个节点 = 一个 Agent）
# ============================================================

def researcher_node(state: TeamState) -> dict:
    """研究员节点：根据任务进行研究"""
    task = state["task"]
    print(f"  [研究员] 正在研究：{task}")

    response = llm.invoke(
        f"你是一位研究分析师。请对以下主题进行简要研究，"
        f"列出 3-5 个关键发现：\n\n{task}"
    )

    print(f"  [研究员] 研究完成 ✓")
    return {"research_result": response.content}


def writer_node(state: TeamState) -> dict:
    """写作者节点：根据研究结果撰写内容"""
    research = state["research_result"]
    feedback = state.get("review_feedback", "")

    prompt = f"你是一位技术写作者。根据以下研究结果，撰写一段 200 字的摘要：\n\n{research}"
    if feedback:
        prompt += f"\n\n请根据以下反馈修改：\n{feedback}"

    print(f"  [写作者] 正在撰写...")
    response = llm.invoke(prompt)
    print(f"  [写作者] 撰写完成 ✓")

    return {
        "draft": response.content,
        "revision_count": state.get("revision_count", 0) + 1,
    }


def reviewer_node(state: TeamState) -> dict:
    """审核者节点：审查文章质量"""
    draft = state["draft"]
    print(f"  [审核者] 正在审核...")

    response = llm.invoke(
        f"你是一位内容审核者。请审查以下文章，判断质量是否合格。\n"
        f"如果合格，回复'通过'。如果不合格，给出具体修改建议。\n\n{draft}"
    )

    passed = "通过" in response.content
    print(f"  [审核者] 审核结果：{'通过 ✓' if passed else '需要修改 ✗'}")

    return {
        "review_feedback": response.content,
        "review_passed": passed,
    }


def output_node(state: TeamState) -> dict:
    """输出节点：生成最终结果"""
    return {"final_output": state["draft"]}


# ============================================================
# 条件路由函数
# ============================================================

def review_decision(state: TeamState) -> Literal["output", "writer"]:
    """审核后的路由：通过则输出，不通过则回到写作者修改"""
    if state["review_passed"] or state.get("revision_count", 0) >= 3:
        return "output"
    return "writer"


# ============================================================
# 构建图
# ============================================================

workflow = StateGraph(TeamState)

# 添加节点
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("output", output_node)

# 添加边
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_conditional_edges("reviewer", review_decision)
workflow.add_edge("output", END)

# 编译图
app = workflow.compile()


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    print("LangGraph 多 Agent 协作 Demo")
    print("=" * 50)
    print("工作流：研究员 → 写作者 → 审核者")
    print("特点：审核不通过会回到写作者修改（最多 3 轮）")
    print("=" * 50)

    initial_state = {
        "task": "AI Agent 在企业中的应用现状",
        "research_result": "",
        "draft": "",
        "review_feedback": "",
        "review_passed": False,
        "revision_count": 0,
        "final_output": "",
    }

    print(f"\n任务：{initial_state['task']}\n")

    result = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print(f"修改轮次：{result['revision_count']}")
    print(f"最终输出：\n{result['final_output'][:500]}")
    print("=" * 50)
