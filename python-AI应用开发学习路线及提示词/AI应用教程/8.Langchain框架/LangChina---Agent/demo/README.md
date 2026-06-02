# LangChain Agent Demo

## 安装依赖

```bash
pip install langchain langchain-core langchain-openai langchain-community langchain-ollama langchain-chroma langchain-text-splitters pypdf pydantic
```

确保 Ollama 已安装并下载 Embedding 模型（Demo 5 用到）：
```bash
ollama pull nomic-embed-text
```

可选（Demo 2 演示内置工具时用）：
```bash
pip install duckduckgo-search
```

## 学习顺序

### Demo 1: Agent 基础概念
**文件**: `1_agent_basics.py`

跑通你的第一个 Agent。两个简单工具（查天气、计算器）+ DeepSeek，直观感受 LLM 自己决策调用工具的过程。重点看 verbose 输出的执行步骤。

**需要 API Key**

---

### Demo 2: Tools 工具定义
**文件**: `2_tools.py`

掌握各种姿势写工具：@tool 装饰器、Pydantic 参数模式、错误处理、内置工具。这一节不跑 Agent，只演示工具定义。

**不需要 API Key**

---

### Demo 3: Agent 与 AgentExecutor
**文件**: `3_agent_executor.py`

深入理解 AgentExecutor 的核心参数：max_iterations、verbose、return_intermediate_steps、流式输出。学习自定义 System Prompt 控制 Agent 行为。

**需要 API Key**

---

### Demo 4: Memory 对话记忆
**文件**: `4_memory.py`

让 Agent 记住对话历史。用 RunnableWithMessageHistory 包装 Agent，按 session_id 隔离不同用户。演示内存存储 vs 文件存储。

**需要 API Key**

---

### Demo 5: 完整 Agent 应用 ⭐
**文件**: `5_complete_agent.py`

把所有东西串起来：**公司客服助手"小 A"**。
- RAG 工具（查公司知识库）
- 业务工具（查订单、计算器、当前时间）
- 多轮对话（带记忆）
- 完整的多轮对话演示

这是一个面试时拿得出手的"完整作品"。

**需要 API Key + 本地 Ollama**

---

## 示例文档

`sample_docs/` 文件夹包含 3 个示例文档（用于 Demo 5 的 RAG）：
- `refund_policy.txt` — 公司退款政策
- `shipping_policy.txt` — 配送政策
- `product_intro.txt` — 产品介绍

做自己的 Agent 项目时，把这个文件夹换成你自己的知识库文档即可。

---

## 对应理论文档

| Demo | 对应理论文档 |
|------|------------|
| Demo 1: Agent 基础 | 1.Agent 基础概念.md |
| Demo 2: Tools | 2.Tools 工具定义.md |
| Demo 3: AgentExecutor | 3.Agent 与 AgentExecutor.md |
| Demo 4: Memory | 4.Memory 对话记忆.md |
| Demo 5: 完整 Agent | 5.完整 Agent 应用.md |

---

## 核心要点

- **Agent 不是 Chain**：Chain 流程写死，Agent 让 LLM 自己决策调用什么工具、几次调用
- **Tool 的 description 决定一切**：LLM 只看 docstring 决定要不要调，参数怎么填
- **AgentExecutor 是循环执行器**：把"决策→调工具→回填→再决策"循环管理起来
- **记忆要靠 RunnableWithMessageHistory**：旧版 ConversationBufferMemory 已过时
- **RAG 也是一个 Tool**：用 create_retriever_tool 包装检索器，让 Agent 自己决定何时查知识库

学完这 5 个 Demo，你就能：
- 独立设计和实现一个 Agent 应用
- 在面试时讲清楚 Chain vs Agent 的区别
- 把已有的 RAG 系统升级成 Agent 形态

---

## 进阶方向

- **LangGraph**：更复杂的 Agent 流程图（条件分支、并发、循环）
- **多 Agent 协作**：Supervisor 模式、Hierarchical Agents
- **MCP（Model Context Protocol）**：标准化工具协议
- **LangSmith**：Agent 调用追踪和性能评估
