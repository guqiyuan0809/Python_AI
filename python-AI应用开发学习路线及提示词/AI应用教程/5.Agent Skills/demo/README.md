# Agent Skills Demo

## 学习顺序

### Demo 1: Function Calling 基础（最重要）
**文件**: `1_function_calling_basic.py`

理解 AI 如何决定调用你写的函数，完整的 6 步流程：
定义函数 → 描述给 AI → AI 判断 → 返回调用意图 → 你执行 → AI 生成回答

**需要 API Key**（使用 OpenRouter）

---

### Demo 2: Function Calling 循环（Agent 雏形）
**文件**: `2_function_calling_loop.py`

在 Demo 1 基础上加了循环：AI 可以多次调用工具，自己决定什么时候停。
这就是 Agent 的核心模式。

**需要 API Key**（使用 OpenRouter）

---

### Demo 3: 记忆机制
**文件**: `3_memory.py`

三种记忆方式的演示：短期记忆、工作记忆（滑动窗口）、长期记忆（文件存储）。

**不需要 API Key**，纯逻辑演示。

---

## 核心要点

Agent Skills 的四大能力中：
- **工具调用（Function Calling）**：最核心，Demo 1 和 Demo 2 重点讲
- **记忆**：Demo 3 讲，本质是管理 messages 列表 + 持久化
- **规划**：在 Demo 2 的循环中已经体现（AI 自己决定下一步做什么）
- **流式输出**：你在 Skills 阶段已经学过，不再重复

理解了 Function Calling + 循环，就理解了 Agent 的核心。
