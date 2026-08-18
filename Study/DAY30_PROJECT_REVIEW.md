# Day30：项目复盘与金汤令改造准备

## 1. 当前项目已经交付了什么

当前项目不是“调用一次大模型”的 Demo，而是一个可被 Java 业务系统调用的 AI 服务模块：

```text
浏览器 Vue
  -> Nginx（静态资源、统一入口）
  -> 公园 Java 后端（登录态、角色/权限、可信身份头）
  -> Python FastAPI（AI 路由、会话归属、RBAC、RAG、Agent）
  -> MySQL / RabbitMQ / Celery Worker / Milvus / DashScope
```

其中 Java 的角色类似传统业务系统的 BFF/业务网关：浏览器 Token 只在 Java 内校验；Python 不接收浏览器 Token，而是只信任带 `X-Service-API-Key` 的 Java 服务调用，并使用 Java 传来的当前用户身份做二次授权和数据归属校验。

## 2. 已完成的核心能力

| 能力 | 当前实现 | 企业价值 |
| --- | --- | --- |
| 会话与消息 | `chat_session`、`chat_message`，按用户归属校验 | 多轮对话和防止横向越权 |
| 异步任务 | `ai_async_task` + `ai_task_outbox` + Celery | 不阻塞 HTTP，支持可靠投递、重试和状态轮询 |
| RAG | 文档解析、Embedding、Milvus 召回、Reranker、引用审计和阈值拒答 | 用企业知识库回答，降低无依据回答 |
| Tool / Agent | 工具白名单、参数校验、策略拦截、人工确认、循环护栏 | 模型不能直接操作数据库或越权执行动作 |
| Prompt 与评测 | Prompt 版本、评测集、运行报告、候选门禁、发布/回滚审计 | 不凭感觉上线 Prompt 或 Agent 改动 |
| 可观测性 | trace、阶段调用日志、任务/run/prompt 关联查询 | 能定位一次 AI 请求在哪一阶段失败或成本异常 |
| AI 安全 | API Key、可信 Java 代理、RBAC、授权审计 | AI 能力纳入现有业务权限边界 |
| 部署 | Docker 镜像、Compose、Live/Ready、Nginx | 可重复启动并让未就绪实例不接业务流量 |

## 3. Day29 端到端异步链路验收

本次真实验证的是普通会话异步聊天，不是模拟数据：

```text
Vue 点击“提交异步任务”
-> 无 sessionId 时调用 Java POST /python-ai/sessions
-> Java 校验当前登录用户并向 Python 创建归属该用户的会话
-> Vue 保存 Java 返回的 sessionId
-> Java POST /python-ai/sessions/chat/async
-> Python 同一事务写用户消息、ai_async_task、ai_task_outbox
-> Dispatcher 投递 RabbitMQ
-> Celery Worker 执行模型调用并将任务更新为 success/error
-> Vue GET /python-ai/tasks/{taskId} 轮询并展示最终回答
```

这条链路还暴露并修复了两个典型的跨语言 DTO 契约问题：

1. Python 返回 `session_id`，而 Java/Vue 对外约定 `sessionId`；
2. Python 返回 `task_id`，而 Java/Vue 对外约定 `taskId`。

Java DTO 使用 `@JsonAlias("session_id")`、`@JsonAlias("task_id")` 做**反序列化兼容**：能接收 Python 的 snake_case，但再向 Vue 序列化时保持 camelCase。不能直接用 `@JsonProperty`，否则 Java 响应会继续输出 snake_case，Vue 读取 `created.sessionId` 或 `submittedTask.taskId` 就会得到 `undefined`。

本次真实排障证据：Worker 已把任务 `348003351020965888` 标为 `success`，但页面一度显示 `error`；API 日志显示页面实际轮询了 `/api/chat/tasks/undefined`。因此并非第一次轮询过早，而是 Java 对外字段名错误。修复 DTO 并重新构建、重启 Java 后，前端到 Worker 的闭环验证通过。

## 4. ID 的职责不要混用

| ID | 含义 | 典型用途 |
| --- | --- | --- |
| `trace_id` | 一次请求链路标识，通常是 32 位十六进制 | 查 Java/Python/Nginx 调用链和阶段日志 |
| `session_id` | 一个多轮会话 | 找该用户的历史消息和上下文 |
| `message_id` | 会话中的一条消息 | 关联具体问答和模型调用 |
| `task_id` | 一个异步业务任务，当前为雪花 ID | 前端轮询、查任务状态、重试/超时处理 |
| `run_id` | 一次 RAG/Agent/Prompt 评测运行 | 查看固定样本集的详细评测报告 |
| `prompt_version_id` | 一个运行时 Prompt 版本 | 追溯某阶段实际使用的提示词 |

例如 `348003351020965888` 是 `task_id`，不是 `trace_id`。两者可在 `ai_async_task.trace_id` 关联，但用途完全不同。

## 5. 当前发布边界

- 浏览器只访问 Nginx 的 `8088`（正式环境通常是 `80/443`）；
- Nginx 的 `/python-ai/` 只代理 Java `9090`，不提供直达 Python `/api/chat/` 的路径；
- Python `8000` 仅绑定 `127.0.0.1`，防止局域网或公网绕过 Java 登录态与可信代理身份校验；
- API、Worker、Beat 使用同一镜像，但分别以 Web、Celery Worker、Celery Beat 三种进程职责运行；
- 数据库迁移在发布前由 Alembic 显式执行，生产容器关闭自动建表。

## 6. 当前版本的已知边界（不是缺陷掩盖）

1. Nginx 到 Java 当前使用 `host.docker.internal:9090`，适合 Windows 本地联调；真实服务器应改为同一 Docker 网络的 Java 服务名、内网地址或 Kubernetes Service。
2. 前端只是课程验证控制台，不替代公园项目正式前端的菜单、登录和 UI 规范。
3. Java 中园长角色到 AI 权限的映射仍是教学过渡实现；正式接入应使用现有菜单/按钮权限表授予 `python-ai:*` 权限编码。
4. Agent 的示例写工具仍停在 `require_confirm`，没有接入真实业务写库；真正接入前需要定义确认单、幂等键、审批与业务审计。
5. 当前 Outbox 超时扫描没有 Worker heartbeat/lease；长模型调用的更严格治理留给后续生产化升级。

## 7. 接下来进入 Day31 的范围

Day31 不再扩建通用 Python 基础能力，而是盘点金汤令（或公园项目）已有 Coze/AI 功能：入口在哪里、输入输出是什么、哪些数据权限必须保留、哪些能力优先替换为 Python AI 模块。产出会是一份改造清单和边界设计，之后再实现最小可替换功能。
