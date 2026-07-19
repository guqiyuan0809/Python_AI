# Day11 - Java 后端调用独立 Python AI 服务

## 今天的架构选择

今天不再把 Java 示例代码放进 `D:\Pythoncode\Study`，而是把 Java 调用层放到真实 Java 项目：

```text
D:\workcode\park_platform
```

Python AI 服务仍然放在：

```text
D:\Pythoncode\Study
```

这样更贴近企业常见结构：

```text
前端 / Apifox
    -> Java Spring Boot 业务后端
    -> Java 内部 PythonAiClient
    -> HTTP 调用 Python FastAPI AI 服务
    -> 大模型 / MySQL 会话记录 / SSE 流式输出
```

核心理解：Java 不直接运行 Python 代码，而是把 Python AI 模块当成一个独立服务，通过 HTTP 接口调用。

## 为什么 Java 和 Python 要分开

Java 后端适合承载：

- 用户、权限、菜单、业务流程
- 原有业务系统集成
- 网关、鉴权、事务、业务数据
- 面向前端的统一接口入口

Python AI 服务适合承载：

- 大模型调用
- Prompt 编排
- 会话上下文管理
- RAG 检索
- 向量库
- Agent / Tool Calling
- SSE 流式输出

所以企业里常见的是：

```text
Java 业务系统
    -> 调用 Python AI 服务
        -> 调用模型、知识库、工具
```

## 本次代码位置

Java 侧代码：

```text
D:\workcode\park_platform\src\main\java\com\yt\parkplat\integration\pythonai
```

主要结构：

```text
integration\pythonai
├── config
│   ├── PythonAiProperties.java
│   └── PythonAiRestTemplateConfig.java
├── client
│   └── PythonAiClient.java
├── controller
│   └── PythonAiProxyController.java
└── dto
    ├── PythonAiApiResponse.java
    ├── PythonAiChatResponse.java
    ├── PythonAiCreateSessionResponse.java
    ├── PythonAiSessionChatProxyRequest.java
    └── PythonAiSessionChatRequest.java
```

Python 侧服务：

```text
D:\Pythoncode\Study\day04_app
```

## 配置层

`application.yml` 新增：

```yaml
python-ai:
  base-url: http://127.0.0.1:8000
  connect-timeout-millis: 3000
  read-timeout-millis: 120000
```

`PythonAiProperties` 负责读取这段配置。

这类似 Python 中的 `settings.py`，也类似 Java 项目里通过配置类读取 `application.yml`。

重点：

- `base-url`：Python FastAPI 服务地址
- `connect-timeout-millis`：连接 Python 服务的最大等待时间
- `read-timeout-millis`：等待 AI 响应的最大时间

AI 接口可能比普通业务接口慢，所以 `readTimeout` 通常会设置得更长。

## DTO 分层

这次最重要的设计点是拆成两个请求 DTO。

`PythonAiSessionChatProxyRequest` 面向前端 / Apifox：

```java
private String sessionId;
private String message;
private Integer historyLimit;
```

这符合 Java 常见的 camelCase 写法，也方便 Swagger / Apifox 展示。

`PythonAiSessionChatRequest` 面向 Python FastAPI：

```java
@JsonProperty("session_id")
private String sessionId;

@JsonProperty("history_limit")
private Integer historyLimit;
```

Python 接口需要的是 snake_case 字段：

```json
{
  "session_id": "xxx",
  "message": "你好",
  "history_limit": 6
}
```

所以 Java Controller 接收前端请求后，会先转换成 Python 内部请求体，再交给 `PythonAiClient` 调用 Python 服务。

## Controller 层

`PythonAiProxyController` 对外暴露 Java 接口：

```text
POST /python-ai/sessions
POST /python-ai/sessions/chat
```

核心代码：

```java
PythonAiSessionChatRequest pythonRequest = new PythonAiSessionChatRequest(
        request.getSessionId(),
        request.getMessage(),
        request.getHistoryLimit()
);
return pythonAiClient.sessionChat(pythonRequest);
```

这一步就是典型的“边界转换”：

```text
前端请求 DTO
    -> Java Controller 校验
    -> 转成 Python 内部 DTO
    -> Java Client 调 Python 服务
```

## Client 层

`PythonAiClient` 类似一个内部 SDK。

它负责：

- 拼接 Python 服务 URL
- 创建 HTTP 请求头
- 发送 JSON 请求体
- 接收 Python 统一响应体
- 把返回结果反序列化成 Java DTO

核心调用：

```java
ResponseEntity<PythonAiApiResponse<T>> response = restTemplate.exchange(
        properties.buildUrl(path),
        HttpMethod.POST,
        entity,
        responseType
);
```

`ParameterizedTypeReference` 用来保留泛型响应类型，例如：

```java
PythonAiApiResponse<PythonAiChatResponse>
```

如果不用它，Java 的泛型擦除会让 Jackson 不容易准确知道 `data` 里面具体应该反序列化成什么类型。

## 和金汤令项目的关系

金汤令项目里已经存在：

```text
AiController
AiService
AiServiceImpl
CozeConfig
/ai/invoke/{ability}
/ai/stream/{ability}
/ai/async/{ability}
```

当前方向不是直接说“原功能就是我写的 Python”，而是更稳妥地表达为：

```text
基于原有低代码 AI 能力，我设计并实现了 Python FastAPI AI 服务重构方案，
让部分 AI 能力可以从 Coze 调用迁移为自研 Python AI 服务调用。
```

后续可以演进成：

```text
Java /ai 接口
    -> AiServiceImpl
    -> 根据 ability/provider 路由
        -> CozeAdapter
        -> PythonAiAdapter
```

## 面试表达

可以这样说：

```text
我把 AI 能力从 Java 业务系统中解耦为独立 Python FastAPI 服务。
Java 侧负责权限、业务编排、统一入口和调用治理；
Python 侧负责大模型调用、会话管理、上下文压缩、RAG 和流式输出。
Java 后端通过 RestTemplate 调用 Python 服务，并通过 DTO 做字段映射和响应体反序列化。
```

也可以结合真实业务背景说：

```text
原系统中的 AI 能力先通过低代码平台快速验证。
为了增强可控性、可扩展性和工程化能力，我基于原有 ability 能力路由，
设计了 Java 调用 Python AI 服务的适配层，为后续替换部分低代码能力做准备。
```

## 今天应该掌握

- Java 调 Python 的本质是 HTTP 服务调用
- Java 业务后端和 Python AI 服务在企业里通常分开部署
- `application.yml` 负责管理 Python 服务地址和超时配置
- `RestTemplate` 可以作为 Java 调 Python 的 HTTP 客户端
- `@JsonProperty` 用来处理 Java camelCase 和 Python snake_case 的字段差异
- 前端 DTO 和内部调用 DTO 最好分开，避免接口边界混乱

## 今日理解检查

### 已完成验证

- 已启动 Python AI 服务和 `park_platform` Java 项目
- 已通过 Java 代理接口完成创建会话
- 已通过 Java 代理接口完成会话多轮对话
- 已验证调用链路是：前端 / Apifox -> Java Controller -> PythonAiClient -> Python FastAPI -> 大模型 / MySQL

### 你的理解总结

`PythonAiProxyController` 是给前端或 Apifox 调用的 Java 接口入口。

`PythonAiClient` 类似一个远程调用客户端，不是传统业务 Service，但作用接近“专门负责调用 Python AI 模块的逻辑层”。它内部通过 `RestTemplate` 发 HTTP 请求，通过 `PythonAiProperties` 获取 Python 服务地址。

拆分两个 DTO 的原因是接口边界不同：

- `PythonAiSessionChatProxyRequest` 面向前端，使用 Java 常见的 camelCase 字段，也负责参数校验和接口文档展示
- `PythonAiSessionChatRequest` 面向 Python FastAPI，通过 `@JsonProperty` 把 Java 字段映射成 Python 接口需要的 snake_case

`RestTemplate` 可以类比 `RedisTemplate`：

- `RedisTemplate` 是 Java 操作 Redis 的模板工具
- `RestTemplate` 是 Java 调用外部 HTTP 服务的模板工具

### 需要微调的表达

面试里不要直接把 `PythonAiClient` 说成“业务逻辑层”，更准确的说法是：

```text
PythonAiClient 是 Java 侧封装的 Python AI 服务调用客户端，职责类似内部 SDK 或适配器，
负责拼接 URL、设置请求头、发送 HTTP 请求、解析统一响应体。
```

这样表达会更工程化，也更容易和后面的 `PythonAiAdapter`、`AiProviderRouter` 设计衔接。

## 后续要升级

- Java 调 Python SSE 流式接口
- Java 侧透传 `trace_id`
- Java 侧统一异常处理和降级
- Python AI 服务鉴权
- 调用失败重试、超时、熔断
- 把这套 adapter/provider 思路迁移到金汤令 AI 能力路由

## 第二阶段：Java 侧调用治理

第一阶段只证明了 Java 可以调用 Python。第二阶段开始处理企业项目真正关心的问题：

- Python 服务没启动怎么办
- Python 返回业务失败怎么办
- 调用超时怎么办
- Java 和 Python 的日志怎么用同一个 ID 串起来
- Java 对前端是否还保持项目统一响应体

### 1. 统一 trace_id 请求头

这次新增了：

```java
public static final String TRACE_ID_HEADER = "X-Trace-Id";
```

Java Controller 收到请求时：

```text
如果前端传了 X-Trace-Id，就沿用前端传入的 ID
如果前端没传，就由 Java 生成一个 UUID
```

然后 Java 调 Python 时，把这个 ID 放进请求头：

```java
headers.set(PythonAiConstants.TRACE_ID_HEADER, traceId);
```

Python 侧原本的中间件会读取 `X-Trace-Id`，并把它写入日志和响应头。

这样一次请求的排查链路就是：

```text
前端 X-Trace-Id
    -> Java Controller
    -> PythonAiClient
    -> Python FastAPI middleware
    -> Python 日志 / Java 响应头
```

### 2. Java 对前端返回项目统一响应体

原来 Java Controller 直接返回：

```java
PythonAiApiResponse<T>
```

这会把 Python 服务的响应体直接暴露给前端。

现在改成返回项目自己的：

```java
ResultResponse<T>
```

核心转换：

```java
return ResultResponse.success(pythonResponse.getData(), pythonResponse.getMessage());
```

这样前端看到的是 `park_platform` 项目统一返回格式，而不是 Python 内部格式。

可以理解成：

```text
PythonAiApiResponse<T>
    是 Java 调 Python 时使用的内部响应体

ResultResponse<T>
    是 Java 对前端暴露的项目统一响应体
```

### 3. PythonAiCallException

这次新增了：

```java
PythonAiCallException
```

它表示 Java 调 Python AI 服务失败。

典型失败场景：

- Python 服务没启动
- Python 接口超时
- Python 返回空响应
- Python 返回 `code != 200`

这些情况都会统一抛出 `PythonAiCallException`，并保留：

- `code`
- `message`
- `traceId`

### 4. 局部异常处理器

这次新增了：

```java
@RestControllerAdvice(assignableTypes = PythonAiProxyController.class)
public class PythonAiExceptionHandler
```

它只处理 `PythonAiProxyController` 抛出的异常，不影响项目里其他 Controller。

这叫局部异常处理，适合教学和渐进式改造真实老项目。

异常响应逻辑：

```java
response.setHeader("X-Trace-Id", ex.getTraceId());
return ResultResponse.error(ex.getCode(), ex.getMessage());
```

也就是说，就算 Python 调用失败，前端仍然可以从响应头拿到 trace_id，再交给后端排查。

### 5. 第二阶段调用链路

成功链路：

```text
前端 / Apifox
    -> Java Controller 生成或读取 trace_id
    -> PythonAiClient 透传 X-Trace-Id
    -> Python FastAPI 返回 PythonAiApiResponse
    -> Java Controller 转成 ResultResponse
    -> 前端收到统一响应体 + X-Trace-Id 响应头
```

失败链路：

```text
前端 / Apifox
    -> Java Controller
    -> PythonAiClient 调 Python 失败
    -> 抛出 PythonAiCallException
    -> PythonAiExceptionHandler 捕获
    -> 返回 ResultResponse.error
    -> 响应头保留 X-Trace-Id
```

### 面试表达

可以这样说：

```text
Java 侧没有直接把 Python 服务响应暴露给前端，而是在代理层做了一次响应体转换。
PythonAiClient 负责调用 Python FastAPI 服务，Controller 负责把 PythonAiApiResponse 转成项目统一的 ResultResponse。
同时我在 Java -> Python 调用链路中透传 X-Trace-Id，Python 服务失败时抛出自定义 PythonAiCallException，
再由局部异常处理器统一转成 Java 项目的错误响应，保证前端体验和后端排查链路一致。
```

## 第三阶段：Java 代理 Python SSE 流式接口

普通接口是一问一答：

```text
前端 -> Java -> Python -> 等完整结果 -> Java 一次性返回前端
```

SSE 流式接口是边生成边返回：

```text
前端
    -> Java SseEmitter 接口
    -> Java 调 Python SSE 接口
    -> Python 一段段返回 data: ...
    -> Java 一段段 emitter.send(...)
    -> 前端持续看到输出
```

### 1. 为什么普通 RestTemplate.exchange 不够

`RestTemplate.exchange(...)` 适合普通 HTTP：

```text
请求发出去
等待完整响应体
一次性反序列化成 Java 对象
```

但 SSE 不是一次性完整 JSON，而是持续返回：

```text
data: {"type":"start", ...}

data: {"type":"delta", ...}

data: {"type":"done", ...}
```

所以 Java 侧需要用更底层的：

```java
restTemplate.execute(...)
```

它允许 Java 拿到响应输入流，然后一行一行读取 Python 返回的 SSE 内容。

### 2. Java 对前端使用 SseEmitter

Controller 新增：

```java
@PostMapping(value = "/sessions/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public SseEmitter sessionStreamChat(...)
```

`SseEmitter` 可以理解成：

```text
Java 后端握着一条可以持续向前端发送数据的连接
```

普通接口是：

```java
return ResultResponse.success(data);
```

流式接口是：

```java
return emitter;
```

后续由异步线程不断：

```java
emitter.send(...)
```

最后：

```java
emitter.complete();
```

### 3. 为什么要线程池

`SseEmitter` 返回后，HTTP 连接会保持住。

读取 Python SSE 流是一个持续过程，如果直接阻塞 Controller 当前线程，会降低 Java 服务处理其他请求的能力。

所以新增：

```java
PythonAiStreamExecutorConfig
```

核心是一个线程池：

```java
@Bean("pythonAiStreamExecutor")
public Executor pythonAiStreamExecutor()
```

Controller 中：

```java
pythonAiStreamExecutor.execute(() ->
        pythonAiClient.streamSessionChat(pythonRequest, traceId, emitter)
);
```

这表示：

```text
Controller 先把 SseEmitter 返回给前端
真正读取 Python 流的任务交给线程池执行
```

### 4. Java 如何读取 Python SSE

`PythonAiClient` 中新增：

```java
streamSessionChat(...)
```

内部调用：

```java
streamPost("/api/chat/sessions/stream", request, traceId, emitter);
```

关键流程：

```text
1. 构造 JSON 请求头
2. 设置 Accept: text/event-stream
3. 使用 RestTemplate.execute 发 POST 请求
4. 把 Java 请求体序列化成 JSON 写入请求流
5. 从 Python 响应 InputStream 中逐行读取
6. 遇到 data: 开头的行就转发给前端
```

### 5. 为什么 Java 这里只转发 data

Python 已经把事件包装成：

```json
{"type":"delta","trace_id":"xxx","delta":"你好"}
```

所以 Java 代理层不需要理解每个事件的业务含义。

Java 只负责：

```text
从 Python 读到 data
通过 SseEmitter 转发给前端
```

这叫代理层职责单一。

如果 Java 代理层开始解析每个 delta，再重新组装业务事件，会导致耦合变重。

### 6. 流式异常处理

普通接口失败可以返回：

```json
{"status":500,"message":"xxx"}
```

但 SSE 连接一旦开始，后端不能再改成普通 JSON 响应。

所以流式失败时，Java 发送一个 SSE error 事件：

```java
emitter.send(SseEmitter.event().name("error").data(errorJson));
emitter.complete();
```

前端收到 error 事件后，就知道流式输出中断了。

### 面试表达

可以这样说：

```text
普通 Java 调 Python 接口使用 RestTemplate.exchange，一次性拿完整响应。
但 SSE 是持续事件流，所以我使用 RestTemplate.execute 读取 Python 返回的 InputStream，
再通过 Spring MVC 的 SseEmitter 将事件持续转发给前端。
为了避免阻塞 Controller 线程，我把读取 Python 流的任务放入独立线程池执行。
同时继续透传 X-Trace-Id，保证普通调用和流式调用都可以做链路追踪。
```

## 后续补充：雪花 ID 教学落点

后续学习异步任务、调用日志表、消息队列和分布式部署时，会补充雪花 ID。

适合使用雪花 ID 的位置：

- AI 异步任务 ID
- Java 调 Python 调用日志 ID
- 消息表业务 ID
- 会话表业务 ID
- 分布式消息 ID

当前 `trace_id` 继续使用 UUID 即可，因为它主要用于链路追踪，不要求趋势递增和数据库索引友好。

## 第三阶段测试结果

已使用 Apifox 测试 Java SSE 代理接口：

```text
POST /python-ai/sessions/stream
```

测试结果：

- 已成功收到多条 `type=delta` 事件
- 已成功收到 `type=done` 事件
- 已确认连接在 `done` 后正常断开
- 已在 Apifox 中使用自定义 JSONPath 完成自动合并

Apifox 自动合并配置：

```text
消息格式：自定义 JSONPath
内容提取路径：$.delta
结束事件字段：$.type
结束事件值：done
错误事件字段：$.type
错误事件值：error
```

注意：不能选择 `OpenAI API 兼容格式`，因为当前接口返回的是自定义 SSE JSON：

```json
{"type":"delta","delta":"..."}
```

而不是 OpenAI 官方常见的：

```json
{"choices":[{"delta":{"content":"..."}}]}
```

## 第三阶段理解检查

### 1. 为什么流式接口用 `RestTemplate.execute`

`RestTemplate.exchange` 更适合普通接口：等待完整响应体，然后一次性反序列化。

SSE 是持续事件流，Java 需要拿到 Python 返回的 `InputStream`，持续读取 `data:` 行，所以使用更底层的 `RestTemplate.execute`。

### 2. `SseEmitter` 的作用

`SseEmitter` 是 Java 和前端之间的 SSE 长连接通道。

Controller 返回 `SseEmitter` 后，连接不会立刻结束，后端可以通过：

```java
emitter.send(...)
```

持续向前端推送事件，最后通过：

```java
emitter.complete()
```

结束连接。

### 3. 为什么 Java 代理层只转发 `data`

Python 已经把每个 SSE 事件封装成 JSON 字符串。

Java 代理层只负责读取 `data:` 后面的内容并转发给前端，不解析 `delta`、`done`、`error` 等业务字段。

这样可以降低 Java 代理层和 Python AI 协议之间的耦合。

### 4. 为什么流式异常不能返回普通统一响应体

普通接口失败时可以返回：

```json
{"status":500,"message":"xxx"}
```

但 SSE 连接一旦开始输出，就不能再切换成普通 JSON 响应。

所以流式异常应发送：

```json
{"type":"error","message":"xxx"}
```

再结束连接。

### 5. 为什么要用 `InputStreamReader + BufferedReader`

Python HTTP 响应底层是字节流 `InputStream`。

SSE 是文本协议，所以 Java 需要：

```text
InputStream
    -> InputStreamReader 按 UTF-8 转成字符流
    -> BufferedReader 按行读取
```

然后筛选 `data:` 开头的行转发给前端。
