# Day10 SSE 流式输出与会话落库笔记

## 今日学习主题

- 把普通 SSE 流式接口升级成会话流式聊天接口
- 支持 `session_id`
- 支持用户消息落库
- 支持 assistant 消息状态流转
- 支持流式结束后保存完整回答

## 新增接口

- `POST /api/chat/sessions/stream`
  - 请求体：`SessionStreamChatRequest`
  - 返回类型：`StreamingResponse`
  - 响应格式：`text/event-stream`

## 为什么 SSE 接口不套统一响应体

普通接口是一次性返回 JSON：

```json
{
  "code": 200,
  "message": "success",
  "data": {}
}
```

SSE 接口是一段持续输出的事件流：

```text
data: {"type": "start", ...}

data: {"type": "delta", ...}

data: {"type": "done", ...}
```

所以它不能直接包成 `ApiResponse[T]`，而是通过 `StreamingResponse` 持续返回多条事件。

## 会话流式聊天核心链路

```text
前端 / Java 后端请求
        |
        v
POST /api/chat/sessions/stream
        |
        v
stream_session_chat_events(...)
        |
        |-- 构造 messages：长期摘要 + 短期历史 + 当前问题
        |-- 保存 user 消息
        |-- 创建 assistant pending 消息
        |-- yield start 事件
        |-- 调用模型 stream=True
        |-- 循环读取 delta
        |-- yield delta 事件给前端
        |-- 服务端拼接完整 answer
        |-- 更新 assistant 为 success
        |-- 必要时刷新会话摘要
        |-- yield done 事件
```

## assistant 消息状态流转

```text
pending -> streaming -> success
```

异常时：

```text
pending / streaming -> error
```

## 为什么流式输出也要落库

- 用户后续打开会话时，需要看到完整历史消息。
- 多轮对话需要依赖历史消息构造上下文。
- 错误排查需要通过 `trace_id`、`stream_id`、`message_id` 关联链路。
- 流式输出虽然前端是一段段收到，但数据库最终应该保存完整回答。
- 如果模型服务支持流式 usage 返回，还应该保存 `prompt_tokens`、`completion_tokens`、`total_tokens`，用于成本统计。

## 流式 token 用量保存

普通非流式接口通常可以直接从完整响应里拿到：

```python
response.usage.prompt_tokens
response.usage.completion_tokens
response.usage.total_tokens
```

流式接口不同，模型是一段段返回文本，usage 通常需要额外开启：

```python
stream_options={"include_usage": True}
```

如果模型服务兼容该能力，最后一个流式 chunk 可能只携带 usage，不携带文本内容。

因此代码里要注意：

- 先用 `getattr(chunk, "usage", None)` 判断是否有 usage
- usage chunk 可能没有 `choices`
- 不能直接写 `chunk.choices[0]`，否则最后一个 usage chunk 可能报错
- 拿不到真实 usage 时，不伪造 token，数据库字段保持为空

## 流式中途失败时怎么处理

流式接口可能出现这种情况：

```text
start -> delta -> delta -> error
```

也就是说，前端已经看到了一部分回答，但模型或网络中途失败。

当前处理策略：

- assistant 消息状态更新为 `error`
- `error_message` 保存真实错误原因
- 如果已经输出了半截回答，`content` 保存半截回答
- 如果没有任何输出，`content` 保存错误提示
- 如果异常前已经拿到真实 token usage，则一并写入 token 字段
- 由于状态是 `error`，后续构造上下文时不会把这条失败回答喂给模型

这样做的好处：

- 保留用户已经看到的内容
- 保留失败现场，方便排查
- 不把失败回答当成成功上下文
- 避免消息长期停留在 `streaming` 或 `pending`

## 本次验证结果

使用假模型流式输出验证，不消耗真实模型 token。

```text
events_count= 4
event_types= ['start', 'delta', 'delta', 'done']
message_count= 2
roles= ['user', 'assistant']
statuses= ['success', 'success']
assistant_content= hello, stream saved.
```

说明：

- SSE 事件顺序正确
- 用户消息已落库
- assistant 消息已落库
- assistant 最终状态为 `success`
- 完整流式回答已拼接并保存

补充验证：模拟最后一个流式 chunk 返回 usage 后，assistant 消息可以写入 token 用量。

```text
event_types= ['start', 'delta', 'delta', 'done']
assistant_content= token saved
prompt_tokens= 11
completion_tokens= 7
total_tokens= 18
```

补充验证：模拟流式输出中途异常。

```text
event_types= ['start', 'delta', 'error']
assistant_status= error
assistant_content= partial answer
assistant_error= 模型流式调用失败：RuntimeError
error_partial= partial answer
```

## 客户端如何消费 SSE 流

已新增测试脚本：

```text
D:\Pythoncode\Study\day10_sse_client.py
```

这个脚本用于模拟前端或 Java 后端调用流式接口：

```text
POST /api/chat/sessions/stream
```

它会逐行读取服务端返回的 SSE 数据：

```text
data: {"type": "start", ...}

data: {"type": "delta", ...}

data: {"type": "done", ...}
```

核心理解：

- 服务端用 `yield` 一段段返回 SSE 字符串。
- 客户端用循环一行行读取响应。
- 收到 `delta` 时追加显示内容。
- 收到 `done` 时结束 loading。
- 收到 `error` 时展示错误提示。

## 测试脚本运行方式

先启动 FastAPI 服务：

```powershell
cd D:\Pythoncode\Study
D:\Pythoncode\.venv\Scripts\python.exe run_day04.py
```

再开一个新的 PowerShell 窗口执行：

```powershell
cd D:\Pythoncode\Study
D:\Pythoncode\.venv\Scripts\python.exe day10_sse_client.py --message "帮我解释一下 FastAPI SSE"
```

如果要使用已有会话继续提问：

```powershell
D:\Pythoncode\.venv\Scripts\python.exe day10_sse_client.py --session-id 你的会话ID --message "继续刚才的话题"
```

注意：

- 这个脚本调用的是真实后端接口。
- 如果后端接口真实调用模型，会消耗模型 token。
- 如果服务没启动，会提示连接失败。
