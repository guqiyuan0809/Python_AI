# 7.Agent-team Demo 学习指南

## 学习顺序

建议按编号顺序学习，从概念理解到框架实操：

1. **先运行 Demo 1**（不需要 API Key），理解三种协作模式的核心逻辑
2. **再选择一个框架 Demo（2/3/4/5）** 动手实操，推荐从 Demo 2（CrewAI）开始
3. **对比不同框架**，理解各自的设计理念和适用场景

---

### Demo 1: 多 Agent 协作模式演示（纯 Python 模拟）
**文件**: `1_multi_agent_patterns.py`

用纯 Python 模拟串行、并行、层级三种协作模式的核心逻辑。
不依赖任何 AI 框架，帮助你理解多 Agent 协作的本质。

**不需要 API Key**

---

### Demo 2: CrewAI 团队协作
**文件**: `2_crewai_team.py`

使用 CrewAI 框架构建一个"内容创作团队"（研究员 + 写作者），
演示角色定义、任务分配、串行执行的完整流程。

**需要 API Key**

---

### Demo 3: LangGraph 多 Agent 工作流
**文件**: `3_langgraph_multi_agent.py`

使用 LangGraph 构建带条件路由的工作流（研究 → 写作 → 审核），
审核不通过会回到写作者修改，演示循环和条件分支。

**需要 API Key**

---

### Demo 4: AutoGen 对话协作
**文件**: `4_autogen_conversation.py`

使用 AutoGen 框架实现两个 Agent 的对话协作（程序员 ↔ 审查员），
演示 Agent 通过来回对话完成编程任务。

**需要 API Key**

---

### Demo 5: LangGraph Supervisor 模式
**文件**: `5_langgraph_supervisor.py`

使用 LangGraph 实现 Supervisor 模式：一个主管 Agent 动态决定
下一步由哪个 Worker 执行，演示层级协作的完整实现。

**需要 API Key**

---

## 对应理论文档

| Demo | 对应理论文档 | 核心知识点 |
|------|------------|-----------|
| 1_multi_agent_patterns.py | 1.多Agent协作概述.md + 2.多Agent协作模式.md | 串行/并行/层级三种模式 |
| 2_crewai_team.py | 3.CrewAI框架.md | Agent/Task/Crew 概念 |
| 3_langgraph_multi_agent.py | 4.LangGraph多Agent.md | State/Node/Edge + 条件路由 |
| 4_autogen_conversation.py | 5.AutoGen框架.md | 对话驱动协作 |
| 5_langgraph_supervisor.py | 4.LangGraph多Agent.md | Supervisor 层级模式 |

---

## 环境准备

```bash
# Demo 1 无需安装额外依赖

# Demo 2（CrewAI）
pip install crewai crewai-tools

# Demo 3 和 Demo 5（LangGraph）
pip install langgraph langchain-openai

# Demo 4（AutoGen）
pip install autogen-agentchat
```

---

## 核心要点

- **多 Agent 的本质**：把复杂任务拆分给多个专业化的 Agent，通过协作完成
- **三种模式**：串行（流水线）、并行（同时执行）、层级（Manager 协调）
- **框架选择**：快速原型用 CrewAI，复杂工作流用 LangGraph，对话式用 AutoGen
- **生产建议**：LangGraph 更适合生产环境（可控性强、状态管理清晰）
- **核心权衡**：多 Agent 增加了灵活性，但也增加了 Token 消耗和调试难度
