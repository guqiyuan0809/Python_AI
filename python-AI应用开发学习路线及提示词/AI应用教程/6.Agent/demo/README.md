# Agent Demo

## 学习顺序

### Demo 1: 最简 Agent（基础结构）
**文件**: `1_simple_agent.py`

用最少的代码实现完整 Agent：System Prompt + Tools + While 循环。
和 Agent Skills Demo 2 类似，但结构更清晰，组件划分更明确。

**需要 API Key**

---

### Demo 2: ReAct Agent（推理 + 行动）
**文件**: `2_react_agent.py`

在 Demo 1 基础上加入显式思考过程：Thought → Action → Observation 循环。
能看到 AI 每步在想什么，方便理解和调试。加入了重复调用检测。

**需要 API Key**

---

### Demo 3: Plan-and-Execute Agent（先规划再执行）
**文件**: `3_plan_execute_agent.py`

另一种 Agent 架构：先让 AI 制定完整计划，再逐步执行。
适合步骤明确的任务。和 ReAct 形成对比。

**需要 API Key**

---

### Demo 4: 带记忆的 Agent（多轮对话）
**文件**: `4_agent_with_memory.py`

在 Agent 中集成完整的记忆管理：
- 工作记忆（滑动窗口控制 Token）
- 长期记忆（用户偏好持久化到文件）
- 自动提取用户信息并保存

支持交互模式（手动对话）和自动测试模式。

**需要 API Key**

---

### Demo 5: 多工具复杂任务 Agent（综合实战）
**文件**: `5_multi_tool_agent.py`

本章最完整的 Agent 实现，综合所有知识：
- 7 个工具，覆盖不同场景
- ReAct 模式 + 日志记录
- 重复检测 + 错误处理
- Token 统计

接近实际项目水平的 Agent 骨架。

**需要 API Key**

---

## 对应理论文档

| Demo | 对应理论文档 |
|------|------------|
| Demo 1: 最简 Agent | 1.Agent核心概念.md |
| Demo 2: ReAct Agent | 2.ReAct模式详解.md |
| Demo 3: Plan-and-Execute | 3.Agent架构模式.md |
| Demo 4: 带记忆的 Agent | 5.Agent记忆与上下文管理.md |
| Demo 5: 多工具 Agent | 4.Agent工具与环境.md + 6.Agent评估与调试.md |

理论文档 7.LangChain与LangGraph.md 没有对应 Demo，因为本章重点是纯代码理解原理，框架留到实战项目时再学。

---

## 核心要点

- Agent 的本质就是 while 循环 + LLM 决策 + 工具调用
- ReAct 是最常用的模式：Thought → Action → Observation
- Plan-and-Execute 适合步骤明确的任务
- 记忆管理是 Agent 的重要组成部分
- 日志、错误处理、重复检测是 Agent 稳定运行的保障
- 理解了纯代码实现，再用框架（LangChain/LangGraph）会事半功倍
