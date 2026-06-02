# AI 应用开发学习路线

> 核心理念：不需要深入底层原理，重点是**会用**、**能做项目**、**能部署上线**。

---

## 学习路线图

```
1. Skills（基础技能）           1-2 天     ⭐
   ↓
2. 本地模型部署（Ollama）        1 天       ⭐
   ↓
3. MCP（连接外部工具）           2-3 天     ⭐⭐
   ↓
4. RAG（知识库问答）★核心        2-3 周     ⭐⭐⭐
   ↓
5. Agent Skills（智能体核心能力） 1-2 周     ⭐⭐⭐
   ↓
6. Agent（构建完整智能体）★核心   2-3 周     ⭐⭐⭐
   ↓
7. 多 Agent 协作（了解即可）      3-5 天     ⭐⭐⭐⭐
   ↓
8. 实战项目（求职作品）           2-3 周     ⭐⭐⭐⭐
```

**核心路线约 2-3 个月，目标是做出 2-3 个能展示的项目。**

---

## 1. Skills — AI 基础技能

### 学什么
- 提示词工程（Prompt Engineering）
- AI API 调用（OpenAI / Claude / 国产大模型 API）
- AI 辅助编程（Cursor、Kiro、GitHub Copilot）

### 怎么学
- **提示词**：直接在日常开发中练习，总结出自己的提示词模板
- **API 调用**：用 Python 调用 OpenAI/Claude API，写一个简单的聊天机器人
- **辅助编程**：日常开发中持续使用 AI 编程工具

### 实践项目
```python
# 用 Python 调用 AI API 的最小示例
import openai
client = openai.OpenAI(api_key="your-key")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)
```

### 学习资源
- OpenAI 官方文档：https://platform.openai.com/docs
- Claude 官方文档：https://docs.anthropic.com
- 提示词工程指南：https://www.promptingguide.ai/zh

### 预计时间：1-2 天（你有 Python 基础，这部分会很快）

---

## 2. 本地模型部署 — Ollama

### 学什么
- Ollama（本地运行开源模型的工具）
- 本地模型选择（Llama、Qwen、DeepSeek 等）
- 本地 API 服务搭建

### 怎么学
1. 安装 Ollama：https://ollama.com
2. 拉取模型：`ollama pull qwen2.5:7b`
3. 运行对话：`ollama run qwen2.5:7b`
4. 用 Python 调用本地 API（和 OpenAI 格式兼容）

### 实践项目
```bash
# 安装 Ollama 后
ollama pull qwen2.5:7b
ollama run qwen2.5:7b

# 本地 API 调用
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:7b",
  "messages": [{"role": "user", "content": "你好"}]
}'
```

### 好处
- 免费、无限调用
- 数据不出本地，隐私安全
- 开发调试时省钱

### 预计时间：1 天

---

## 3. MCP — Model Context Protocol

### 学什么
- MCP 是什么：让 AI 连接外部工具和数据的标准协议
- MCP Server：提供工具能力（读文件、查数据库、调 API）
- MCP Client：AI 应用调用 MCP Server

### 怎么学
1. 理解 MCP 架构：Client ↔ Server ↔ 外部工具
2. 使用现成的 MCP Server（文件系统、数据库、GitHub 等）
3. 自己写一个简单的 MCP Server

### 实践项目
```
场景：让 AI 能读取你的项目文件并回答问题
1. 安装 MCP 文件系统 Server
2. 配置 AI 客户端连接 MCP Server
3. AI 就能直接读取你的代码文件了
```

### 学习资源
- MCP 官方文档：https://modelcontextprotocol.io

### 预计时间：2-3 天

---

## 4. RAG — 检索增强生成 ★核心重点

> 这是目前市场需求最大的 AI 应用方向，企业知识库、智能客服、文档问答都是 RAG。
> 要花足够时间，做出一个完整的、能部署的项目。

### 学什么
- RAG 原理：检索相关文档 → 喂给 AI → 生成回答
- 向量数据库（Milvus、Chroma、Pinecone）
- Embedding（文本向量化）
- 文档解析（PDF、Word、Markdown、网页）
- 检索策略（相似度搜索、混合检索、重排序）
- LlamaIndex 框架（专注 RAG 的框架）

### 关键概念
```
用户提问 → 文本向量化 → 在向量数据库中搜索相似文档 → 把文档 + 问题一起发给 AI → AI 基于文档回答

"你好" → [0.2, 0.8, 0.1]    （AI 把文字变成数字列表）
"你好吗" → [0.3, 0.7, 0.2]  （数字越接近，意思越像）
不需要手算，调一个函数就行：similarity = cosine_similarity(a, b)
```

### 怎么学
1. 先用 LlamaIndex 跑通一个最简单的 RAG demo
2. 换成 LangChain 实现同样的功能，对比两个框架
3. 逐步加入高级特性：混合检索、重排序、多轮对话记忆
4. 用 FastAPI 包装成 API 服务，加上流式输出

### 实践项目（求职作品级别）
```
项目：企业知识库问答系统
技术栈：FastAPI + LangChain/LlamaIndex + Chroma/Milvus + OpenAI API

功能：
1. 上传文档（PDF/Markdown/Word）
2. 自动解析、分块、向量化、存入向量数据库
3. 用户提问，检索相关文档，AI 生成回答
4. 支持多轮对话、引用来源
5. 流式输出（SSE）
6. 简单的前端界面（可用 Streamlit 或 Gradio 快速搭建）

部署：Docker 容器化，能在服务器上跑起来
```

### 学习资源
- LlamaIndex 文档：https://docs.llamaindex.ai
- LangChain RAG 教程：https://python.langchain.com/docs/tutorials/rag/
- FastGPT（开源知识库问答，可参考架构）：https://fastgpt.in

### 预计时间：2-3 周（包含做完整项目）

---

## 5. Agent Skills — 智能体核心能力

### 学什么
- **工具调用（Tool Use / Function Calling）**：让 AI 调用函数/API
- **记忆（Memory）**：让 AI 记住上下文和历史
- **规划（Planning）**：让 AI 拆解复杂任务
- **Streaming（流式输出）**：实时返回结果

### 怎么学
1. 用 LangChain 框架学习每个能力
2. 每个能力写一个小 demo
3. 重点理解 Function Calling 的机制，这是 Agent 的基础

### 实践项目
```python
# 工具调用示例：让 AI 能查天气、查数据库、执行代码
def get_weather(city: str) -> str:
    """查询城市天气"""
    return f"{city}今天晴，25°C"

def query_database(sql: str) -> str:
    """执行 SQL 查询"""
    # 实际连接数据库执行
    pass

# 把函数注册为 AI 可调用的工具
# AI 会自动判断何时调用哪个函数
```

### 学习资源
- LangChain 文档：https://python.langchain.com
- OpenAI Function Calling 文档：https://platform.openai.com/docs/guides/function-calling

### 预计时间：1-2 周

---

## 6. Agent — 构建完整智能体 ★核心重点

> Agent 是 AI 应用的未来方向，和 RAG 并列为最重要的技能。
> 要理解 Agent 的工作原理，并做出一个完整的 Agent 项目。

### 学什么
- Agent 架构：感知 → 思考 → 行动 → 观察 循环
- ReAct 模式：推理 + 行动交替进行
- 自主决策：AI 自己决定用什么工具、执行什么步骤
- LangGraph（LangChain 的 Agent 框架，重点学）

### 关键概念
```
用户指令 → AI 思考（我需要做什么？）
         → 选择工具（用搜索？查数据库？写代码？）
         → 执行工具
         → 观察结果（结果对吗？够了吗？）
         → 继续思考或返回最终答案
```

### 怎么学
1. 先用 LangChain Agent 搭建一个简单的 Agent
2. 学习 LangGraph，用它构建更复杂的 Agent 工作流
3. 给 Agent 配备多个工具（搜索、代码执行、文件读写、API 调用）
4. 用 FastAPI 包装成服务

### 实践项目（求职作品级别）
```
项目：智能数据分析 Agent
技术栈：FastAPI + LangChain/LangGraph + OpenAI API

功能：
1. 用户上传 CSV/Excel 数据文件
2. 用自然语言描述分析需求（"帮我分析销售趋势"）
3. Agent 自动：
   - 读取数据
   - 编写并执行 Python 分析代码
   - 生成图表
   - 输出分析报告
4. 支持多轮追问和修改

或者：
项目：代码审查 Agent
1. 读取 Git 提交记录
2. 分析代码变更
3. 自动生成审查意见
4. 输出审查报告
```

### 学习资源
- LangGraph 文档：https://langchain-ai.github.io/langgraph/
- LangChain Agent 教程：https://python.langchain.com/docs/tutorials/agents/

### 预计时间：2-3 周（包含做完整项目）

---

## 7. 多 Agent 协作（了解即可）

> 多 Agent 目前在生产环境落地不多，大部分公司还在单 Agent 阶段。
> 了解概念和框架，不需要花太多时间。

### 学什么
- 多 Agent 架构：不同 Agent 扮演不同角色
- Agent 间通信：消息传递、任务分配
- 协作模式：串行、并行、层级
- 框架：CrewAI、AutoGen、LangGraph 多 Agent

### 怎么学
1. 用 CrewAI 或 LangGraph 搭建一个简单的多 Agent demo
2. 理解不同协作模式的适用场景
3. 不需要做完整项目，有 demo 能讲清楚原理即可

### 学习资源
- CrewAI：https://docs.crewai.com
- AutoGen：https://microsoft.github.io/autogen
- LangGraph 多 Agent：https://langchain-ai.github.io/langgraph/

### 预计时间：3-5 天

---

## 8. 实战项目 — 求职作品

> 这个阶段是把前面学的东西整合成 1-2 个完整的、能部署的、能在面试中展示的项目。

### 推荐项目组合

**项目一：RAG 知识库问答系统（展示 RAG 能力）**
```
技术栈：FastAPI + LangChain + Chroma/Milvus + OpenAI API + Docker
亮点：
- 完整的文档处理 pipeline
- 混合检索 + 重排序
- 流式输出
- Docker 部署
- 简单前端（Streamlit/Gradio）
```

**项目二：智能 Agent 应用（展示 Agent 能力）**
```
技术栈：FastAPI + LangGraph + OpenAI API + Docker
亮点：
- 多工具调用
- 自主决策和规划
- 完整的 Agent 工作流
- API 服务化
```

### 项目要求
- 代码放 GitHub，README 写清楚
- 能 Docker 一键部署
- 有简单的前端可以演示
- 能讲清楚技术选型和架构设计

### 预计时间：2-3 周（整合优化之前的项目）

---

## 补充学习（按需，不是必须）

### AI 工作流平台（Dify/Coze）
- 了解即可，这些是给非开发者用的低代码平台
- 你会写代码，不需要依赖这些工具
- 但了解 Dify 的架构设计对你有参考价值
- 预计时间：1-2 天浏览一下

### AI 前端生成工具（v0.dev / bolt.new）
- 锦上添花，不是核心竞争力
- 做项目需要前端时可以用来快速生成
- 预计时间：用到时再学

### 模型微调（Fine-tuning / LoRA）
- 了解原理即可，大部分 AI 应用开发岗不要求会训练模型
- 知道什么场景需要微调、微调的基本流程
- 不需要自己动手训练
- 预计时间：看 1-2 篇文章理解概念

### AI 开发框架深入
- LangChain、LlamaIndex 在做项目过程中自然会深入
- 不需要单独花时间系统学，边做边查文档

---

## 总体时间规划

| 阶段 | 内容 | 时间 | 难度 | 优先级 |
|------|------|------|------|--------|
| 1 | Skills（基础技能） | 1-2 天 | ⭐ | 必学 |
| 2 | 本地模型部署 | 1 天 | ⭐ | 必学 |
| 3 | MCP | 2-3 天 | ⭐⭐ | 必学 |
| 4 | RAG ★ | 2-3 周 | ⭐⭐⭐ | 核心重点 |
| 5 | Agent Skills | 1-2 周 | ⭐⭐⭐ | 必学 |
| 6 | Agent ★ | 2-3 周 | ⭐⭐⭐ | 核心重点 |
| 7 | 多 Agent 协作 | 3-5 天 | ⭐⭐⭐⭐ | 了解即可 |
| 8 | 实战项目整合 | 2-3 周 | ⭐⭐⭐⭐ | 必做 |

**总计：约 2-3 个月（每天投入 4-6 小时）**

---

## 求职方向参考

你的背景（Python 后端 + AI 应用）适合的岗位：

| 岗位 | 核心要求 | 你的优势 |
|------|---------|---------|
| RAG 工程师 | RAG pipeline、向量数据库、文档处理 | 后端工程化能力 + FastAPI |
| LLM 应用开发 | API 集成、Agent 开发、Prompt Engineering | Python 全栈 + 项目经验 |
| AI 平台开发 | 内部 AI 平台搭建、API 网关、模型管理 | 后端架构经验 + Django/FastAPI |
| AI 全栈工程师 | 前后端 + AI 能力整合 | 后端扎实 + AI 应用能力 |

---

## 学习建议

1. **每个阶段都要有产出**：不是看完文档就行，要有能跑的代码和能演示的项目
2. **RAG 和 Agent 是绝对核心**：这两块各花 2-3 周，做出求职级别的项目
3. **结合你的后端优势**：FastAPI + LangChain + 向量数据库 这个组合面试很加分
4. **不要贪多**：微调、多 Agent 这些了解概念就行，把核心技能做深做透
5. **用 AI 学 AI**：遇到不懂的直接问 AI，效率最高
6. **项目要能部署**：Docker 容器化，GitHub 上有完整代码和文档，面试时能现场演示


---

## 相关知识（做项目时遇到再学）

| 知识 | 说明 | 什么时候会用到 |
|------|------|--------------|
| AI 应用安全 | Prompt 注入防护、内容审核、敏感信息过滤 | 项目上线时必须考虑 |
| 评估与测试 | 评估 RAG 检索质量、Agent 执行效果 | 做 RAG/Agent 项目时 |
| 日志与监控 | LangSmith、日志记录、Token 成本监控 | 生产环境部署时 |
| 异步与并发 | Python asyncio，高并发下的 AI 调用 | FastAPI 项目优化时 |
| 缓存策略 | 相同问题缓存回答，省 Token 省钱 | 生产环境成本优化 |
| 多模态 | 图片、音频、视频输入处理 | 加分项，面试有亮点 |

**这些不需要单独学，做项目过程中遇到了再学就行。**
