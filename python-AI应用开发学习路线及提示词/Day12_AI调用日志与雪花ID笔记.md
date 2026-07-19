# Day12 AI 调用日志与雪花 ID 笔记

## 1. 今天为什么要加 AI 调用日志

前面我们已经能完成普通聊天、会话聊天、流式聊天、消息落库和 Java 调 Python AI 服务。

但企业项目中只保存“用户问了什么、AI 回了什么”还不够，还需要知道：

- 本次到底有没有真正调用模型；
- 调用的是哪个模型；
- 本次调用耗时多久；
- 本次消耗了多少 prompt token、completion token、total token；
- 本次调用是成功还是失败；
- 出问题时能不能根据 trace_id、session_id、message_id 反查。

所以 Day12 新增 `ai_call_log` 表，用来记录“模型调用行为”。

## 2. `chat_message` 和 `ai_call_log` 的区别

`chat_message` 更偏业务展示：

- 用户问了什么；
- AI 回答了什么；
- 这条消息属于哪个会话；
- 这条 assistant 消息是 success、pending、streaming 还是 error。

`ai_call_log` 更偏系统排查和成本统计：

- 某一次模型调用的唯一 ID；
- 调用类型，比如 `session_chat`、`session_stream_chat`；
- 调用耗时；
- token 消耗；
- 成功或失败；
- 错误原因。

用 Java 类比：

- `chat_message` 像业务表；
- `ai_call_log` 像调用流水表、接口日志表、成本审计表。

## 3. 为什么这次引入雪花 ID

之前我们大量使用 UUID，例如：

- `trace_id`
- `session_id`
- `message_id`
- `stream_id`

UUID 的优点是简单、分布式环境下重复概率极低。

但调用日志、订单、任务、流水这类数据在企业中经常需要：

- 趋势递增，方便排序和索引；
- 数字型或数字字符串，方便跨系统传递；
- 能在多台机器上生成，避免依赖数据库自增 ID；
- 适合作为业务 ID 对外查询。

所以 Day12 给 `ai_call_log.call_id` 使用了简化版雪花 ID。

## 4. 雪花 ID 的组成

当前学习版雪花 ID 大致由三部分组成：

```text
时间戳部分 + 机器ID部分 + 毫秒内序列号部分
```

含义：

- 时间戳：保证 ID 大体按时间递增；
- 机器 ID：保证多台机器生成 ID 时不冲突；
- 序列号：保证同一台机器同一毫秒内生成多个 ID 也不冲突。

当前代码里 `worker_id=1` 是学习阶段写死的。

企业项目中通常会通过配置文件、环境变量、Nacos、K8s 配置等方式给不同服务实例分配不同 `worker_id`。

## 5. 为什么雪花 ID 里需要锁

Python 服务可能同时处理多个请求。

如果多个请求在同一毫秒内同时生成 ID，就会同时修改 `sequence`。

所以当前实现中使用：

```python
with self.lock:
```

它的作用类似 Java 中对共享变量加锁，保证同一个 Python 进程内生成 ID 时，序列号递增过程不会被并发打乱。

## 6. 今天新增的表：`ai_call_log`

关键字段：

- `call_id`：本次模型调用的业务 ID，使用雪花 ID；
- `trace_id`：请求链路 ID，用于 Java 和 Python 日志串联；
- `session_id`：本次调用属于哪个聊天会话；
- `message_id`：本次调用最终对应哪条 assistant 消息；
- `call_type`：调用类型，例如普通会话聊天、流式会话聊天；
- `model`：本次使用的模型；
- `prompt_tokens`：输入消耗；
- `completion_tokens`：输出消耗；
- `total_tokens`：总消耗；
- `cost_ms`：模型调用耗时；
- `status`：成功或失败；
- `error_message`：失败原因。

## 7. 今天新增的接口

新增分页查询 AI 调用日志接口：

```http
GET /api/chat/call-logs
```

可选筛选条件：

- `page`
- `page_size`
- `trace_id`
- `session_id`
- `status`

这个接口用于排查和统计，不是普通用户聊天页面必须展示的接口。

## 8. 当前实现和企业级差距

当前是学习阶段实现，已经能跑通核心思路，但还有企业级优化空间：

- 当前 `worker_id=1` 写死，后续应改成从配置读取；
- 当前用 `Base.metadata.create_all(...)` 自动建表，企业项目应使用 Alembic 管理数据库迁移；
- 当前只记录了会话普通聊天和会话流式聊天，后续摘要生成、标题生成、RAG 检索、工具调用也应该记录；
- 当前调用日志直接写 MySQL，后续高并发场景可考虑异步写日志或先写消息队列；
- 后续可以增加调用成本金额、用户 ID、业务模块、供应商、重试次数等字段。

## 9. 为什么要做异步任务

同步接口的特点是：

```text
前端/Java 后端发起请求 -> Python AI 服务调用模型 -> 等模型返回 -> 接口才返回
```

如果模型调用很慢，用户和 Java 服务都要一直等。

异步任务的特点是：

```text
前端/Java 后端提交任务 -> Python AI 服务立刻返回 task_id
后台慢慢执行模型调用
前端/Java 后端用 task_id 查询任务状态和结果
```

适合异步处理的场景：

- 长文总结；
- 批量文本分析；
- 简历解析；
- 报表生成；
- 知识库文档导入；
- 多步骤 Agent 任务；
- 执行时间不稳定的大模型调用。

## 10. 今天新增的异步任务表：`ai_async_task`

核心字段：

- `task_id`：异步任务业务 ID，使用雪花 ID；
- `trace_id`：提交任务这次请求的链路 ID；
- `session_id`：任务所属会话；
- `message_id`：任务最终关联的 assistant 消息；
- `task_type`：任务类型，例如 `session_chat`；
- `input_text`：用户提交的问题；
- `result_text`：任务成功后的 AI 输出；
- `status`：任务状态；
- `error_message`：失败原因；
- `prompt_tokens`、`completion_tokens`、`total_tokens`：模型消耗；
- `cost_ms`：模型调用耗时。

任务状态流转：

```text
pending -> running -> success
pending -> running -> error
```

含义：

- `pending`：任务已提交，等待后台执行；
- `running`：后台已经开始执行模型调用；
- `success`：任务完成，并写入结果；
- `error`：任务失败，并记录错误原因。

## 11. 今天新增的异步接口

提交异步会话聊天任务：

```http
POST /api/chat/sessions/chat/async
```

请求体：

```json
{
  "session_id": "会话ID",
  "message": "用户问题",
  "history_limit": 6
}
```

返回：

```json
{
  "task_id": "异步任务ID",
  "status": "pending"
}
```

查询异步任务状态：

```http
GET /api/chat/tasks/{task_id}
```

前端或 Java 后端可以轮询这个接口，直到状态变为 `success` 或 `error`。

## 12. BackgroundTasks 的定位

当前使用的是 FastAPI 自带的 `BackgroundTasks`。

它的优点：

- 写法简单；
- 不需要 Redis、RabbitMQ、Celery；
- 适合学习和轻量后台任务；
- 可以帮助理解异步任务的完整链路。

但它不是严格意义上的企业级分布式任务队列。

它的不足：

- 服务重启时，正在执行的后台任务可能丢失；
- 不适合大量耗时任务；
- 不方便做任务重试、延迟任务、任务优先级；
- 多实例部署时任务调度能力有限。

企业项目中更常见的升级方案：

```text
FastAPI 接收请求 -> 写任务表 -> 投递消息队列 -> Worker 消费任务 -> 更新任务表
```

常见技术：

- Celery + Redis/RabbitMQ；
- RQ + Redis；
- Kafka / RabbitMQ / RocketMQ；
- Java 侧也可以用 MQ 统一调度，Python AI Worker 只消费任务。

## 13. 超时补偿与失败重试

当前学习版又新增了两类任务治理能力：

- 超时扫描：查找长时间停留在 `pending` 或 `running` 的任务，并标记为 `error`；
- 失败重试：只允许 `error` 状态的任务重新回到 `pending`，然后再次交给后台执行。

对应接口：

```http
POST /api/chat/tasks/actions/timeout-scan?timeout_minutes=10
POST /api/chat/tasks/{task_id}/retry?history_limit=6
```

为什么它看起来像消息队列：

```text
任务表 pending -> 后台执行 running -> success/error
失败任务 -> retry -> pending -> 再次执行
长时间无结果 -> timeout scan -> error
```

这个状态机和 Java 中 MQ 的消费、重试、死信处理非常接近，但当前还不是 MQ。

当前 `BackgroundTasks` 更像 Java 的 `@Async` 或本机线程池：任务仍由当前 FastAPI 进程执行，服务重启时内存中的执行任务可能丢失。

企业通常升级为：

```text
FastAPI 接收请求 -> MySQL 创建任务 -> 投递 MQ/Redis Broker
-> 多个 Worker 消费 -> 调用模型 -> 更新任务表
-> 前端轮询任务表或通过 SSE/WebSocket 获取完成通知
```

常见组合：

- 轻量 Python 异步任务：Celery + Redis；
- 跨 Java/Python 服务的业务消息：RabbitMQ、RocketMQ、Kafka；
- 高可靠任务：增加重试次数、退避时间、死信队列、幂等键和状态条件更新。

本次重试不会重复保存用户问题：第一次执行时用户消息已落库；重试时只新建一条 assistant 占位消息。构造模型上下文时会移除历史中最近一条相同问题，再把该问题追加到最后，因此模型始终只收到一次本次提问。

## 14. 当前学习版和生产实现的边界

当前超时扫描和后台执行可能在极端并发下同时更新同一个任务。例如扫描器刚标记超时，模型调用恰好又返回成功。

生产中通常通过以下措施避免状态被覆盖：

- 条件更新：只允许 `running` 状态更新为 `success/error`；
- 乐观锁版本号：更新时校验版本；
- MQ 消费幂等：同一个 `task_id` 重复投递也只能成功处理一次；
- 重试次数与指数退避：不是无限重试；
- 达到阈值后投递死信队列，由人工或补偿程序处理。
