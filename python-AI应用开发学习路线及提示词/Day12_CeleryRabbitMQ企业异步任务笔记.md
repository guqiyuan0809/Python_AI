# Day12 Celery + RabbitMQ 企业异步任务笔记

## 1. 最终企业链路

```text
Java / 前端
    -> FastAPI 异步任务接口
    -> MySQL: chat_message + ai_async_task + ai_task_outbox 同一事务提交
    -> RabbitMQ Broker
    -> Celery Worker 消费任务并调用大模型
    -> MySQL 更新任务、消息、调用日志
    -> Java / 前端用业务 task_id 轮询任务状态
```

三张核心表：

- `chat_message`：保存用户消息和 assistant 占位/结果消息。
- `ai_async_task`：保存业务任务状态，Java 和前端只认 `task_id`。
- `ai_task_outbox`：本地消息表，解决 MySQL 与 RabbitMQ 双写不一致问题。

## 2. 为什么要用 Outbox

不能在数据库事务里直接依赖 MQ 投递成功，否则会出现双写问题：

```text
MySQL 任务已提交 -> 服务在投递 MQ 前崩溃 -> MQ 没有消息 -> 任务永远 pending
```

当前做法：

```text
同一事务写业务数据 + Outbox
事务提交后尝试投递 RabbitMQ
投递失败则 Outbox 保持 pending
Celery Beat 定时扫描 pending Outbox 并补发
```

## 3. RabbitMQ 与 Celery 的关系

Celery 是 Python 异步任务框架，不是 MQ 本身。

RabbitMQ 是真正的消息 Broker，负责接收、保存、分发任务消息。

```text
FastAPI -> celery_app.send_task(...) -> RabbitMQ -> Celery Worker
```

类比 Java：

```text
Spring Controller -> RabbitTemplate.convertAndSend(...) -> RabbitMQ -> @RabbitListener
```

## 4. 消费者幂等

RabbitMQ/Celery 默认要按“至少一次投递”理解，消息可能重复到达 Worker。

当前 Worker 用条件更新领取任务：

```sql
UPDATE ai_async_task
SET status = 'running'
WHERE task_id = ? AND status = 'pending';
```

只有影响行数为 1 的 Worker 才能继续调用模型。重复消息会被忽略。

## 5. 失败重试与指数退避

模型调用失败后：

```text
running -> error
如果 retry_count < max_retries:
    error -> pending
    retry_count + 1
    新建一条 delayed Outbox
    available_at = 当前时间 + 退避秒数
```

退避示例：

```text
第 1 次失败后等待 5 秒
第 2 次失败后等待 10 秒
第 3 次失败后等待 20 秒
最多等待 300 秒
```

达到 `max_retries` 后任务保持 `error`，后续再进入 DLQ/人工处理设计。

## 6. 当前技术选型口径

- Python AI 服务内部：Celery + RabbitMQ + MySQL Outbox。
- 当前 Day12 主链路不使用 Redis，任务结果统一写入 MySQL，避免把缓存组件和 MQ 概念混在一起。
- Java 与 Python 服务：仍通过 HTTP 提交任务和查询状态，不直接共享数据库。
- Windows 本地 Worker：继续使用 `--pool=solo`。

## 7. Alembic 迁移文件

当前阶段保留学习版自动建表兜底，但删除 Day12 过渡手写迁移脚本。

Day12 表结构变更以 Alembic migration 为准：

```text
alembic.ini
alembic/env.py
alembic/versions/20260718_001_day12_celery_rabbitmq_outbox.py
```

迁移文件里主要记录：

- 给 `ai_async_task` 增加 `broker_task_id`、`retry_count`、`max_retries`。
- 创建 `ai_task_outbox` 本地消息表。
- 创建任务查询和补偿扫描需要的索引。
- 在 `downgrade()` 中提供回滚逻辑。

开发阶段可以先看 migration 文件如何表达 DDL；后续新环境使用 `alembic upgrade head` 管理表结构。
