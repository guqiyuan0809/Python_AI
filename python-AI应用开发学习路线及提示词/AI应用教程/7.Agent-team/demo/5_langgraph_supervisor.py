"""
Demo 5: LangGraph Supervisor 模式
使用 LangGraph 实现 Supervisor（主管）模式：
一个 Supervisor Agent 动态决定下一步由哪个 Worker 执行。

需要 API Key（配置下方变量）
安装依赖：pip install langgraph langchain-openai
"""

from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ============================================================
# API 配置（替换为你自己的配置）
# ============================================================

API_KEY = "your-api-key-here"
BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-4o-mini"

llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=BASE_URL)

# ============================================================
# 定义状态
# ============================================================

class SupervisorState(TypedDict):
    task: str
    history: list[str]
    next_worker: str
    final_output: str
    step_count: int


# ============================================================
# Worker 节点
# ============================================================

def researcher_worker(state: SupervisorState) -> dict:
    """研究员 Worker"""
    print("  [Worker: 研究员] 执行中...")
    response = llm.invoke([
        SystemMessage(content="你是研究员，负责收集信息。简要回答，不超过 100 字。"),
        HumanMessage(content=f"任务：{state['task']}\n已有信息：{state['history']}")
    ])
    result = f"[研究员] {response.content}"
    print(f"  [Worker: 研究员] 完成 ✓")
    return {
        "history": state["history"] + [result],
        "step_count": state["step_count"] + 1,
    }


def writer_worker(state: SupervisorState) -> dict:
    """写作者 Worker"""
    print("  [Worker: 写作者] 执行中...")
    response = llm.invoke([
        SystemMessage(content="你是写作者，负责撰写内容。简要回答，不超过 100 字。"),
        HumanMessage(content=f"任务：{state['task']}\n已有信息：{state['history']}")
    ])
    result = f"[写作者] {response.content}"
    print(f"  [Worker: 写作者] 完成 ✓")
    return {
        "history": state["history"] + [result],
        "step_count": state["step_count"] + 1,
    }


def summarizer_worker(state: SupervisorState) -> dict:
    """总结者 Worker"""
    print("  [Worker: 总结者] 执行中...")
    response = llm.invoke([
        SystemMessage(content="你是总结者，负责汇总所有信息生成最终输出。"),
        HumanMessage(content=f"请汇总以下工作成果：\n{state['history']}")
    ])
    print(f"  [Worker: 总结者] 完成 ✓")
    return {
        "final_output": response.content,
        "step_count": state["step_count"] + 1,
    }


# ============================================================
# Supervisor 节点
# ============================================================

def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor：决定下一步由谁执行"""
    print(f"\n  [Supervisor] 分析当前进度（已执行 {state['step_count']} 步）...")

    if state["step_count"] >= 3:
        print("  [Supervisor] 决定：任务完成，交给总结者")
        return {"next_worker": "summarizer"}

    response = llm.invoke([
        SystemMessage(content=(
            "你是项目主管。根据任务和当前进度，决定下一步由谁执行。"
            "只回复一个词：researcher / writer / summarizer"
        )),
        HumanMessage(content=(
            f"任务：{state['task']}\n"
            f"已完成步骤：{state['history']}\n"
            f"已执行 {state['step_count']} 步"
        ))
    ])

    next_w = response.content.strip().lower()
    if next_w not in ("researcher", "writer", "summarizer"):
        next_w = "researcher"

    print(f"  [Supervisor] 决定：下一步由 {next_w} 执行")
    return {"next_worker": next_w}


# ============================================================
# 路由函数
# ============================================================

def route_to_worker(state: SupervisorState) -> str:
    """根据 Supervisor 的决定路由到对应 Worker"""
    return state["next_worker"]


def after_worker(state: SupervisorState) -> str:
    """Worker 执行完后，判断是否结束"""
    if state.get("final_output"):
        return "end"
    return "supervisor"


# ============================================================
# 构建图
# ============================================================

workflow = StateGraph(SupervisorState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_worker)
workflow.add_node("writer", writer_worker)
workflow.add_node("summarizer", summarizer_worker)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_to_worker,
    {"researcher": "researcher", "writer": "writer", "summarizer": "summarizer"}
)

workflow.add_conditional_edges("researcher", after_worker, {"supervisor": "supervisor", "end": END})
workflow.add_conditional_edges("writer", after_worker, {"supervisor": "supervisor", "end": END})
workflow.add_edge("summarizer", END)

app = workflow.compile()


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    print("LangGraph Supervisor 模式 Demo")
    print("=" * 50)
    print("架构：Supervisor 动态分配任务给 Worker")
    print("Worker：研究员 / 写作者 / 总结者")
    print("=" * 50)

    initial_state = {
        "task": "分析 Python 在 AI 开发中的优势",
        "history": [],
        "next_worker": "",
        "final_output": "",
        "step_count": 0,
    }

    print(f"\n任务：{initial_state['task']}\n")

    result = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print(f"总执行步骤：{result['step_count']}")
    print(f"最终输出：\n{result['final_output'][:500]}")
    print("=" * 50)
