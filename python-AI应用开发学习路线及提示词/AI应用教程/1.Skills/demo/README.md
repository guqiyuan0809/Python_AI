# Skills Demo

## 学习顺序

### Demo 1: 基础调用（入门第一步）
**文件**: `1_basic_call.py`

最简单的 AI API 调用，理解核心三步：
创建 client → 构造 messages → 调用 API 取回答

**需要 API Key**（使用 DeepSeek）

---

### Demo 2: System Prompt 角色设定
**文件**: `2_system_prompt.py`

同一个问题，不同的 System Prompt 得到完全不同的回答。
演示四种角色：默认 / Python 工程师 / 编程导师 / 翻译器。
同时理解 temperature 参数对回答的影响。

**需要 API Key**（使用 DeepSeek）

---

### Demo 3: 多轮对话（上下文记忆）
**文件**: `3_multi_turn.py`

理解 AI 的"记忆"是怎么实现的：
不带历史 → AI 不记得你说过什么；带上历史 → AI 有了记忆。
包含交互式多轮对话体验。

**需要 API Key**（使用 DeepSeek）

---

### Demo 4: 流式输出
**文件**: `4_stream.py`

普通模式 vs 流式模式对比：
普通模式等全部生成完才返回，流式模式逐字出现（打字机效果）。
做 RAG 和 Agent 时必须用。

**需要 API Key**（使用 DeepSeek）

---

### Demo 5: 错误处理 + Token 统计
**文件**: `5_error_and_token.py`

生产环境必备：错误处理（Key 过期、超时、模型不存在）、
Token 用量统计与成本计算、换模型只改两行。

**需要 API Key**（使用 DeepSeek）

---

## 对应理论文档

| Demo | 对应理论 |
|------|---------|
| Demo 1 基础调用 | 4.OpenAI SDK详解.md、5.AI API调用.md |
| Demo 2 System Prompt | 1.提示词类型.md、5.AI API调用.md |
| Demo 3 多轮对话 | 4.OpenAI SDK详解.md（messages 结构） |
| Demo 4 流式输出 | 4.OpenAI SDK详解.md（Stream + SSE） |
| Demo 5 错误处理 | 3.Token与计费.md、2.主流大模型对比.md |

---

## 核心要点

Skills 阶段的 5 种 API 调用方式：
- **基础调用**：一问一答，Demo 1 重点讲
- **System Prompt**：角色设定改变 AI 行为，Demo 2 重点讲
- **多轮对话**：维护 messages 列表实现记忆，Demo 3 重点讲
- **流式输出**：逐字返回，做产品必用，Demo 4 重点讲
- **错误处理**：生产环境必备，Demo 5 重点讲

实际项目中这 5 种方式是组合使用的：
```
聊天应用 = System Prompt + 多轮对话 + 流式输出 + 错误处理
RAG 知识库 = System Prompt + 检索文档 + 流式输出 + 错误处理
Agent 智能体 = System Prompt + 多轮对话 + 工具调用 + 流式输出 + 错误处理
```

理解了这 5 个 Demo，就掌握了所有 AI 应用的底层基础。
