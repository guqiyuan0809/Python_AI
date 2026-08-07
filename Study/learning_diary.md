# Python AI 学习日记

## 2026-08-05：RAG 父子切块、Contextual Retrieval 与 Reranker 优化

今天围绕 JVM 知识库文档做了一次完整的 RAG 检索质量优化。最开始使用普通固定切块策略时，系统可以召回到包含正确答案的内容，但 chunk 粒度较粗，`Paragraph:43` 中“堆内存的作用”会和死锁检查、jmap、jconsole 等相邻诊断内容混在一起，导致返回给模型的资料不够干净。

随后尝试语义完整切块策略，虽然段落边界比固定切块更自然，但在当前 JVM 样本集上 Recall/MRR 没有明显优于线上 active 版本，说明不能只因为策略听起来更高级就直接上线，必须通过评测数据判断。

之后采用父子切块策略：子块负责向量召回，父块负责回答阶段回填更完整上下文。进一步把 `child_max_characters` 从 350 调整到 260，`child_overlap_characters` 从 50 调整到 40，`child_min_characters` 从 80 调整到 60，并增加“段内短标题边界识别”，让类似 `**堆** **作用：** ...` 的定义型内容不再黏到上一主题后面。

在 Reranker 优化上，原来精排只传子块原文 `content`，没有利用上下文化阶段生成的 `contextual_summary`。优化后，精排输入改为“检索背景 + 原文子块”，让 Reranker 能看到子块的业务语义背景；回答生成阶段仍然只把真实原文和父块内容作为事实依据，避免把模型生成的背景摘要当成事实引用。

最终在问题“JVM 堆内存的作用是什么？”上，v5 父子切块候选版本把包含 `Paragraph:43` 的正确 chunk 排到 Top1，命中内容为：`**堆** **作用：** 存储对象和成员变量，线程共享，存在线程安全问题，有垃圾回收机制。`

今天形成的工程认知：

- RAG 不是一个黑盒，而是一条可拆分、可替换、可评测的链路。
- 离线阶段包括文档解析、切块、上下文化、Embedding、向量入库。
- 在线阶段包括用户问题、问题向量化、向量粗排、Reranker 精排、上下文组装、模型回答。
- 优化某个节点时，应保持其他节点尽量不变，用同一批样本集对比 Recall@K、MRR@K、无答案误召回率、耗时和成本。
- 不能只看单次接口返回就切换 active，必须通过候选版本评测和人工确认后再发布。

面试表达：

> 我不是简单接入向量库，而是通过检索评测发现固定切块存在上下文混杂、语义切块召回不稳定的问题，于是改造为父子切块，并针对中文技术文档增加短标题边界识别；同时优化 Reranker 输入，将上下文化摘要和子块原文一起参与精排，最终让目标知识点从候选排序靠后提升到 Top1。

### RAG 检索评测集补强

为了避免只凭单条问题判断切块策略优劣，今天把 JVM 文档的检索评测样本从 5 条扩展到 20 条，并按 `normal`、`boundary`、`no_answer` 三类组织：

- `normal` 样本验证常规知识点能否稳定召回，例如虚拟机栈、StringBuilder/StringBuffer、直接内存、反射优化等。
- `boundary` 样本验证容易混淆的相邻章节能否正确区分，例如 Minor GC、G1、双亲委派、字符串拼接、类加载链接阶段等。
- `no_answer` 样本验证知识库没有依据时是否会误召回，例如 ZGC、MaxRAMPercentage、Java 21 虚拟线程等当前文档未覆盖的问题。

样本标注不使用 `chunk_id`，而是使用解析阶段生成的 `segment_index` 作为标准答案。这样即使后续重新切块、重新向量化，评测标准仍然稳定，可以公平比较不同文档版本的检索效果。

新增了两个课程脚本：

- `scripts/seed_jvm_retrieval_eval_samples.py`：幂等补齐 JVM 检索评测样本，重复执行会自动跳过已有问题。
- `scripts/run_jvm_retrieval_eval_compare.py`：固定同一数据集，批量评测 v2/v4/v5 等文档索引版本，并横向输出 Hit@K、Recall@K、Precision@K、MRR@K、无答案误放行率和耗时。

今天形成的工程认知：RAG 优化不能只靠“接口看起来回答不错”，而要把知识库策略发布做成准入流程。固定评测集，替换一个策略节点，横向比较指标，达标后再人工确认切换 active，这才接近企业里的知识库发布方式。

### RAG 无答案拒答阈值

今天继续补齐 Day22 的“无答案兜底”工程能力：检索完成后先读取 Top1 相似度分数，如果 Top1 分数低于阈值，系统直接返回“当前知识库未找到足够依据。”，不再把弱相关资料交给聊天模型。

这一步解决的是 RAG 里的典型幻觉风险：向量库总会返回“最相近”的 chunk，但“最相近”不等于“足够相关”。如果没有阈值，模型可能会基于弱相关资料硬编一个看似合理的答案。

当前实现策略：

- 服务端默认阈值配置为 `RAG_MIN_RELEVANCE_SCORE=0.25`。
- 请求体可以传 `score_threshold` 覆盖默认值，便于 Apifox 或评测脚本做阈值调参。
- 同步 RAG、会话 RAG、异步会话 RAG 都复用同一套阈值逻辑。
- 异步任务的 Outbox payload 会保存 `score_threshold`，人工重试或自动重试时不会丢失首次提交的检索策略。
- 响应中返回 `top_score`、`score_threshold`、`rejected_by_score_threshold`，方便前端和管理端判断本次回答是模型生成还是低分拒答。

面试表达：

> 我在 RAG 链路中没有只依赖 Prompt 让模型自己判断“资料不足”，而是在检索后增加了 Top1 score 阈值。低于阈值时直接拒答并跳过聊天模型调用，既降低幻觉风险，也节省 token 成本。同时这个阈值支持配置和请求级覆盖，并且在异步任务 Outbox 中持久化，保证重试链路和首次执行策略一致。

阈值验证结果：

- 当前 active v2 版本中，16 条可回答样本的 Top1 分数范围为 `0.292474 ~ 0.832519`。
- 4 条无答案样本的 Top1 分数范围为 `0.057576 ~ 0.199985`。
- 因此 `0.20 ~ 0.29` 是当前样本集下比较安全的阈值区间。
- 选择 `0.25` 作为默认值，是因为它位于“无答案最高分”和“可回答最低分”的中间附近，能在当前样本集上做到无答案全部拒答、可回答不误拒。

工程认知：拒答阈值不是固定真理，而是随知识库内容、切块策略、Embedding 模型、Reranker 模型和评测样本变化而变化。企业中应把阈值当成可配置参数，并通过评测集定期校准。

## 2026-08-06：Day23 Tool Calling 工具调用

今天开始学习 Tool Calling。RAG 解决的是“根据知识库资料回答问题”，Tool Calling 解决的是“模型判断是否需要调用后端业务能力”。二者经常一起出现在企业 AI 系统中：RAG 查文档知识，Tool Calling 查实时业务状态或执行受控业务动作。

今天先实现单轮 Tool Calling，不进入 Agent Loop。流程是：

1. 用户提交问题。
2. 模型先根据工具白名单判断是否需要工具。
3. 如果需要工具，模型输出 `tool_name` 和 `arguments`。
4. 后端校验工具名必须在白名单中，参数必须通过 Pydantic 校验。
5. 后端执行只读工具。
6. 再把工具执行结果交给模型生成最终回答。

当前注册了三个只读工具：

- `get_session_status`：根据 `session_id` 查询会话标题、摘要、状态和更新时间。
- `get_async_task_status`：根据 `task_id` 查询异步任务状态、结果、错误、token 和耗时。
- `get_knowledge_document_summary`：根据 `document_id` 查询知识库文档解析、切块、active 版本等基础状态。

后续补充了一个更贴近业务落地的只读工具：

- `get_work_order_analysis_result`：根据工单业务 ID 查询最新的工单结构化分析结果，包括分类、风险等级、摘要、建议和是否需要人工复核。

同时注册了一个高风险动作工具演示：

- `close_work_order_demo`：演示“关闭工单”这类写操作工具。它会出现在工具白名单里，但由于 `read_only=false`、`require_human_confirm=true`、`risk_level=high`，当前执行器会自动拦截，不会真正执行。

工程边界：

- 模型不能直接执行 SQL。
- 模型不能调用未注册工具。
- 模型生成的参数不能直接信任，必须经过 Pydantic 校验。
- 当前工具全部是只读工具，不产生外部副作用。
- 业务查询工具只允许模型提供最小必要参数，例如 `business_id`；业务类型、结果类型和查询范围由后端固定，防止模型越权查询其他业务数据。
- 工具注册表中增加了工具元信息：`tool_type`、`read_only`、`require_human_confirm`、`risk_level`。
- 当前执行器只允许自动执行 `low + read_only + 不需要人工确认` 的工具；后续如果加入写操作工具，必须接入人工确认、权限校验和审计流程。
- 高风险工具即使被模型选中，也会在后端执行前被拦截，不能把安全边界寄托在模型“自觉不调用”上。
- 高风险工具被拦截时，不再让最终回答退化成普通聊天；第二次模型回答必须同时接收第一轮 `decision` 和工具执行/拦截结果，避免模型在没有执行工具的情况下误说“已完成操作”。
- 今天只是单轮工具调用，不做多步循环；多步“思考、调用、再判断、停止”属于 Day24 Agent Loop。

高风险工具拦截验证结果：

- 用户请求“关闭工单 WO_001”。
- 模型识别到该请求匹配 `close_work_order_demo`。
- 后端执行器根据工具元信息判断该工具为 `business_action + high + 非只读 + 需要人工确认`。
- 系统返回 `status=blocked`，没有执行工具 executor，也没有修改任何业务数据。
- 最终回答正确说明“关闭工单操作尚未执行，需要人工确认”，没有再误说“已关闭”。

面试表达：

> 我实现 Tool Calling 时没有让模型直接访问数据库或任意接口，而是通过后端维护工具白名单。模型只负责选择工具和生成参数，后端负责参数校验、权限边界和真实执行。这样既能让大模型使用业务实时数据，又能避免 SQL 注入、越权调用和不可审计的自动操作。

补充可观测性：

- `/api/chat/tool-calling` 已接入统一 `ai_call_log`。
- 成功时记录 `call_type=tool_calling`、`trace_id`、模型、token、耗时和 `status=success`。
- 失败时记录 `status=error`、`error_type`、`error_message` 和耗时，方便后续按 traceId 排查。
- 当前先复用统一调用日志表，不新建专门工具调用明细表；如果 Day26 做更细可观测性，可以再补充 `tool_name`、参数快照、工具耗时、模型决策原文等结构化字段。

工程认知：

Tool Calling 不是“让模型拥有执行权限”，而是“让模型提出调用意图”。真正的权限控制、参数校验、工具执行和审计留痕都在后端完成。这个边界非常重要，否则模型一旦误判或被 prompt 注入诱导，就可能变成不可控的自动操作入口。

今天对 Tool Calling 的理解可以总结为：

> 模型是决策者和参数抽取器，不是 SQL 执行者。模型负责判断一个模糊自然语言请求下一步应该走哪个受控工具，并抽取最小必要参数；后端负责校验工具白名单、校验参数、执行固定查询模板并返回结果。模型不是从自己的训练数据里搜索业务事实，而是通过受控工具读取真实业务系统的数据。

### Day23 完成总结

今天完成了单轮 Tool Calling 的企业化雏形，核心产出包括：

- 新增 `/api/chat/tool-calling` 接口。
- 建立后端工具注册表 `TOOL_REGISTRY`。
- 建立工具参数 DTO 和 Pydantic 校验。
- 支持模型根据自然语言问题选择工具并抽取参数。
- 支持 AI 系统状态查询工具：会话状态、异步任务状态、知识库文档状态。
- 支持业务查询工具：根据工单业务 ID 查询结构化分析结果。
- 增加工具元信息：工具类型、是否只读、是否需要人工确认、风险等级。
- 增加高风险动作工具演示：`close_work_order_demo`。
- 增加执行前安全门：只允许 `low + read_only + 不需要人工确认` 的工具自动执行。
- 修复“工具未执行但模型误说已执行”的问题：最终回答阶段必须同时看到第一轮决策和工具执行/拦截结果。
- 接入统一 `ai_call_log`，记录 Tool Calling 的 token、耗时、状态和异常。

验证过的关键场景：

- 普通解释类问题：模型判断不需要工具。
- 查询异步任务状态：模型选择 `get_async_task_status` 并成功返回真实任务状态。
- 查询工单结构化分析结果：模型选择业务查询工具，并由后端固定查询模板执行。
- 请求关闭工单：模型识别高风险工具，后端返回 `blocked`，最终回答正确说明尚未执行且需要人工确认。

Day23 和 Day24 的边界：

- Day23 是单轮 Tool Calling：只做一次“决策 -> 工具执行/拦截 -> 最终回答”。
- Day24 会升级为 Agent Loop：进入“感知 -> 决策 -> 行动 -> 观察反馈 -> 再决策 -> 停止”的受控循环。
- Day24 必须继续沿用 Day23 的工具白名单、风险元信息、安全门和日志能力，否则循环越多，风险越高。

简历表达：

> 我在项目中实现了受控 Tool Calling 能力。模型只负责根据用户自然语言选择工具和抽取参数，后端通过工具注册表、参数 DTO、风险元信息和执行安全门控制真实执行。低风险只读查询工具可自动调用，高风险业务动作会被 blocked 并要求人工确认。同时调用链路接入 traceId 和调用日志，避免模型越权、误执行和不可审计的问题。

## 2026-08-07：Day24 Agent Loop 循环

今天开始从 Day23 的单轮 Tool Calling 升级到受控 Agent Loop。

Day23 的链路是固定的一次：

```text
用户问题 -> 模型决策 -> 工具执行/拦截 -> 模型最终回答 -> 结束
```

Day24 的链路变成有限循环：

```text
感知用户目标和历史观察
-> 模型决策下一步 action
-> 后端执行工具或拦截
-> 把 observation 放回上下文
-> 模型再次决策
-> 最终回答或达到最大步数停止
```

当前实现的是企业可控版 Agent，而不是完全自由的 Agent：

- 新增 `/api/chat/agent-loop` 接口。
- 请求体支持 `message` 和 `max_steps`。
- `max_steps` 限制为 1 到 5，默认 3，防止无限循环和成本失控。
- 每一轮模型只能选择两种动作：`call_tool` 或 `final_answer`。
- `call_tool` 会复用 Day23 的 `ToolDecision`、工具白名单、参数 DTO 校验和风险安全门。
- 高风险工具仍然只能返回 `blocked`，不能因为进入 Loop 就绕过安全策略。
- 响应会返回 `steps`，记录每一轮的 action、tool_name、arguments、reason、observation 和 final_answer。
- Agent Loop 接入 `ai_call_log`，使用 `call_type=agent_loop` 记录 token、耗时和异常。
- Agent 决策解析增加了 action 归一化：模型如果把动作输出成“调用工具/最终回答”等中文表达，后端会先归一化为 `call_tool/final_answer`，再交给 Pydantic 枚举校验。
- 新增重复工具调用护栏：如果 Agent 再次调用相同工具和相同参数，后端会返回 `status=stopped_by_guardrail` 并停止循环，避免重复打业务接口和浪费 token。
- 将工具执行前的风险判断抽象为 `ToolPolicyChecker`，返回 `allow / block / require_confirm` 策略结果；当前高风险或写操作工具返回 `require_confirm`，不直接执行。
- 统一 Agent observation 格式，将工具原始返回包装成 `success / not_found / require_confirm / blocked / error / stopped_by_guardrail` 等状态，避免下一轮模型从不同工具的私有字段里猜含义。
- 新增终止态 observation 的确定性收口：当 observation 是 `not_found / require_confirm / blocked / error / stopped_by_guardrail` 时，后端直接返回确定性回答，不再额外调用下一轮模型。
- 工具执行异常观察化：工具调用阶段出现参数不合法、工具执行失败等异常时，不直接让整个 Agent Loop 接口失败，而是记录为本轮 `observation.status=error`，再由终止态收口返回。
- 强化多工具串联提示：Agent 可以从上一轮 `observation.data` 中提取字段作为下一轮工具参数，例如先查异步任务得到 `session_id`，再查会话状态，最后汇总回答。

今天的核心认知：

> Agent Loop 不是让模型自由行动，而是让模型在后端设置的边界里反复进行“决策 -> 行动 -> 观察 -> 再决策”。真正的工具执行、权限控制、风险拦截、最大循环次数和审计日志都必须由后端控制。

Day24 当前仍然是同步接口，适合先理解 Loop 结构。后续如果 Agent 步骤变多、耗时变长，应复用之前的异步任务 + Outbox + Worker 机制，把 Agent Loop 做成异步任务并由前端轮询状态。

工程踩坑：

- Prompt 中的 JSON 示例不能写成 `"action": "call_tool 或 final_answer"`，模型可能照抄说明文本，导致后端枚举校验失败。
- 更稳的写法是分别给出 `call_tool` 和 `final_answer` 两个合法 JSON 示例。
- 即使 prompt 写得很清楚，后端也要做轻量兼容和强校验：先归一化常见表达，再用 Pydantic 校验最终 DTO。
- “不要重复调用工具”不能只写在 prompt 里，后端也必须用代码记录已调用过的 `tool_name + arguments`，在重复动作发生前拦截。
- 风险判断不应长期散落在业务 if 中，应抽象成策略校验层。课程阶段规则仍写在代码里，企业中可以进一步接数据库配置、配置中心或规则引擎。
- 工具 executor 可以保留各自的业务返回结构，但进入 Agent Loop 的 observation 应统一包装。这样模型、前端和审计系统看到的是稳定协议，而不是每个工具各说各话。
- 不是所有 observation 都需要再交给模型判断。对于未找到、需人工确认、被拦截、工具异常这类明确终止态，后端确定性收口更省 token，也更不容易误回答。
- 工具执行失败和模型决策失败要区分处理：模型输出 JSON 不合法属于模型决策失败；工具参数或业务查询失败属于 action 的 observation，应进入 steps，方便前端和审计看到 Agent 失败在哪一步。
- Agent Loop 相比单轮 Tool Calling 的核心价值，不是“多调用几次模型”，而是可以让上一轮工具返回的 observation 驱动下一轮决策，实现多个受控工具的串联。

多工具串联验证结果：

- 用户目标：查询异步任务状态，再根据任务里的 `session_id` 查询会话状态，最后汇总回答。
- 第 1 步：Agent 调用 `get_async_task_status`，参数为 `task_id=343403537901817856`。
- 第 1 步 observation 返回 `status=success`，并在 `data.session_id` 中给出 `5360cb51bd804535a8bd20ee58eca528`。
- 第 2 步：Agent 从上一轮 observation 中提取 `session_id`，调用 `get_session_status`。
- 第 2 步 observation 返回会话状态 `active`。
- 第 3 步：Agent 判断信息足够，执行 `final_answer`，汇总任务 `success` 和会话 `active`。

这个验证说明当前 Agent Loop 已经具备“观察结果驱动下一步工具调用”的能力。

### Day24 完成总结

Day24 已完成受控 Agent Loop 的第一版，且已完成多工具串联验证。它和 Day23 的单轮 Tool Calling 的区别不是“多调用几次模型”，而是把上一步工具的 `observation` 作为下一步决策的输入，使 Agent 能在受控边界内完成依赖型任务。

当前交付：

- 新增同步接口 `POST /api/chat/agent-loop`。
- 用 `action=call_tool/final_answer` 把每一轮行为限制为两种明确动作。
- 用 `max_steps=1~5` 限制循环上限，避免无限循环、接口长时间占用和 token 失控。
- 复用 Day23 的工具白名单、Pydantic 参数 DTO、`ToolPolicyChecker` 和高风险工具拦截，确保进入 Loop 不会绕过安全门。
- 将不同工具的返回归一化为统一 `observation` 协议，便于下一轮模型、前端和审计侧读取。
- 对 `not_found`、`require_confirm`、`blocked`、`error`、`stopped_by_guardrail` 做后端确定性终止，不再无意义地多调用模型。
- 用 `tool_name + canonical arguments` 生成调用键，拦截重复的相同工具调用，防止 Agent 卡在循环中。
- 每轮把 action、参数、原因、observation 和最终回答记录在响应 `steps` 中，并通过 `ai_call_log` 记录整次 Agent Loop 的模型 token、耗时和异常。

Day23 与 Day24 的边界：

| 维度 | Day23 Tool Calling | Day24 Agent Loop |
| --- | --- | --- |
| 模型决策次数 | 一次工具决策，随后生成最终回答 | 最多 `max_steps` 次决策 |
| 工具调用次数 | 最多一个 | 可串联多个不同工具 |
| 下轮输入 | 无下一轮 | 上轮 `observation` 进入下一轮上下文 |
| 典型场景 | “查询任务状态” | “查询任务状态，再查询关联会话状态” |
| 风险控制 | 白名单、DTO、策略拦截 | 保留 Day23 控制，额外增加步数和重复调用护栏 |

面试表达：

> 我实现的不是让大模型自由执行的 Agent，而是受控 Agent Loop。模型每轮只能在 `call_tool` 和 `final_answer` 两个动作中选择；真实工具执行仍由后端白名单、Pydantic 参数校验和风险策略层控制。我把工具返回统一为 observation 协议，并将未找到、需人工确认、被拦截和工具异常做成确定性终止态，避免 Agent 在失败后继续消耗 token 或编造结果。对于多步任务，下一轮只能使用上轮 observation 中已返回的事实字段，例如先从任务结果中获得 session_id，再查询会话状态。

当前实现的边界与后续演进：

- 当前接口是同步执行，适合 1~5 步、低耗时、只读工具场景。
- 如果工具链路变长或包含慢模型调用，应复用 `ai_async_task + ai_task_outbox + RabbitMQ + Celery Worker` 改为异步 Agent 任务，由前端轮询。
- 当前工具风险规则写在代码中的 `ToolPolicyChecker`，企业后续可以迁移到数据库配置、配置中心或规则引擎，但“后端硬校验不能交给模型”这一原则不变。
- 当前 `ai_call_log` 已覆盖整次调用；Day26 可进一步记录每一步的决策原文、工具名、脱敏参数快照、工具耗时、observation 状态和终止原因，形成可检索的 Agent Step Trace。
- 写操作工具不能因为 Agent Loop 存在就自动执行，后续必须结合用户身份、权限、人工确认、审批流和审计记录。

Day24 自测清单：

- 正常单工具：查询一个存在的异步任务，返回 `success` 和正确数据。
- 多工具串联：先查任务，再使用返回的 `session_id` 查会话，最后 `final_answer`。
- 不存在数据：传入不存在的 task_id，第一轮得到 `not_found`，后端直接收口，不进入第二轮模型决策。
- 高风险动作：请求关闭工单，得到 `require_confirm` 或 `blocked`，不修改业务数据。
- 重复动作：模型再次发出相同 `tool_name + arguments`，得到 `stopped_by_guardrail`。
- 最大步数：始终未给出 `final_answer` 时，返回 `max_steps_reached`。

Day24 到此结束。下一天从 Day25 开始，不重复基础 Harness 建设；应在已有 Prompt Harness、RAG 检索评测和 Agent Loop 的基础上，系统化整理 Harness 的评测对象、指标、回归准入和自动化运行方式。
