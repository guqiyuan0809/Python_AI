# Day07 预习：会话 ID 与历史对话保存

## 明天的主题

Day07 要把当前的单次 AI 调用升级成有会话概念的聊天服务。

核心目标：

- 理解 `session_id` 是什么
- 理解一段会话和多条消息的关系
- 理解历史对话如何重新组装成 `messages`
- 理解为什么不能无限携带全部历史
- 初步设计聊天会话表和消息表

## 三个 ID 的区别

### session_id

表示一段连续多轮对话。

例如用户打开一个聊天窗口，连续问了 10 个问题，这 10 轮问答都属于同一个 `session_id`。

### trace_id

表示一次 HTTP 请求链路。

例如用户在同一个会话里问第 3 个问题，这一次接口调用会有一个新的 `trace_id`。

### stream_id

表示一次模型流式生成过程。

如果一次请求里调用多个模型，每次模型流式输出可以有不同的 `stream_id`。

关系可以理解为：

```text
session_id > trace_id > stream_id
```

## 为什么要保存历史对话

大模型本身不会永久记住用户之前说过什么。

真实工程中，多轮对话通常是这样实现的：

```text
用户新提问
-> 根据 session_id 查询历史消息
-> 组装成 messages
-> 加上当前问题
-> 调用模型
-> 保存用户问题和模型回答
```

## 为什么不能把全部历史都发给模型

如果每次都把所有历史问答都塞进 `messages`：

```text
prompt_tokens 会越来越大
成本会越来越高
响应会越来越慢
无关历史会干扰模型回答
可能超过模型上下文窗口
```

所以企业项目一般会做：

- 最近 N 轮原文
- 历史摘要
- 关键信息记忆
- RAG 检索相关历史
- token 预算控制

## 最小表结构设计

### chat_session

一条记录表示一个聊天会话。

```text
id
session_id
user_id
title
status
created_at
updated_at
```

### chat_message

一条记录表示一条消息。

```text
id
message_id
session_id
trace_id
stream_id
role
content
model
prompt_tokens
completion_tokens
total_tokens
status
error_message
created_at
```

## role 字段

`role` 用来区分消息来源：

```text
system：系统提示词
user：用户消息
assistant：模型回答
tool：工具调用结果
```

明天重点先掌握：

```text
user
assistant
```

## messages 组装示例

数据库中存储的历史：

```text
user: 什么是 SSE？
assistant: SSE 是服务端推送事件。
user: 它适合 AI 流式输出吗？
assistant: 适合，因为可以持续推送增量内容。
```

下一次用户继续提问：

```text
它和 WebSocket 有什么区别？
```

后端组装：

```python
messages = [
    {"role": "system", "content": "你是一个专业 AI 应用开发老师。"},
    {"role": "user", "content": "什么是 SSE？"},
    {"role": "assistant", "content": "SSE 是服务端推送事件。"},
    {"role": "user", "content": "它适合 AI 流式输出吗？"},
    {"role": "assistant", "content": "适合，因为可以持续推送增量内容。"},
    {"role": "user", "content": "它和 WebSocket 有什么区别？"},
]
```

模型看到这些内容后，就能理解用户说的“它”指的是 SSE。

## 明天的代码目标

先用内存模拟数据库：

```text
session_store = {}
```

实现：

- 创建会话
- 追加用户消息
- 追加 AI 消息
- 根据 session_id 查询历史
- 组装 messages
- 控制最近 N 轮历史

后续再升级为 MySQL / Redis。

## 简历表达方向

学完后可以逐步表达为：

> 设计 AI 聊天会话管理机制，基于 `session_id` 维护多轮对话历史，并在模型调用前动态组装上下文 `messages`，为连续对话、历史恢复和后续 RAG 检索打基础。

## 明天检查问题

1. `session_id`、`trace_id`、`stream_id` 分别表示什么？
2. 为什么多轮对话需要保存 user 和 assistant 两种消息？
3. 为什么不能每次都把所有历史喂给模型？
4. 如果只保留最近 N 轮，会损失什么？
5. 如果要恢复几个月前的会话，后端需要做哪些步骤？
