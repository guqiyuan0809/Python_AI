# Day16 PromptOps 与数据库驱动 Harness 笔记

## 1. Day16 主题

Day16 的核心是把 Day15 的离线 harness 从“脚本 + jsonl 文件 + 代码常量”升级为“数据库驱动 + HTTP 可触发 + 异步执行 + 管理端可查询”的 PromptOps 雏形。

核心链路：

```text
ai_prompt_version
    -> 保存 prompt 内容和模型参数

ai_eval_dataset / ai_eval_sample
    -> 保存评测数据集和人工标注样本

HTTP 触发评测任务
    -> ai_async_task + ai_task_outbox
    -> RabbitMQ
    -> Celery Worker 执行 harness
    -> ai_eval_run / ai_eval_case_result
    -> 管理端查询评测结果
```

## 2. 为什么 prompt 和样本要入库

Day15 时，prompt 写在代码里，样本写在 jsonl 文件里。

这种方式适合学习和本地脚本，但不适合长期维护。

企业里 prompt 和样本会经常变化：

- prompt 需要多版本管理。
- prompt 可能需要灰度或审核。
- 模型参数可能需要调整。
- 样本集会持续补充边界样本和错误样本。
- 线上失败样本需要人工清洗后加入评测集。

所以 Day16 增加三张主数据表：

```text
ai_prompt_version
ai_eval_dataset
ai_eval_sample
```

## 3. ai_prompt_version 表

作用：保存某个业务场景下的 prompt 版本。

核心字段：

- `prompt_id`：Prompt 版本业务 ID。
- `prompt_name`：Prompt 名称，例如 `work_order_analysis`。
- `prompt_version`：版本号，例如 `v2`。
- `system_prompt`：系统提示词。
- `user_prompt_template`：用户提示词模板。
- `model`：使用的模型。
- `temperature`：模型温度。
- `max_tokens`：最大输出 token。
- `status`：状态，例如 `draft`、`active`。

企业意义：

> prompt 不再只是代码里的字符串，而是可版本化、可查询、可评测、可上线管理的配置资产。

## 4. ai_eval_dataset 和 ai_eval_sample

`ai_eval_dataset` 保存数据集版本。

例如：

```text
dataset_name = work_order_analysis
dataset_version = work_order_analysis_v1
```

`ai_eval_sample` 保存每条样本。

核心字段：

- `sample_id`
- `dataset_id`
- `dataset_version`
- `sample_type`
- `input_text`
- `expected_json`
- `source_type`
- `source_ref_id`
- `status`

`sample_type` 可以区分：

```text
normal      正常样本
boundary    边界样本
error       错误样本
```

## 5. ai_failure_sample 和 ai_eval_sample 的区别

这两个表不冲突。

| 表 | 作用 |
|---|---|
| `ai_failure_sample` | 线上失败样本池 |
| `ai_eval_sample` | 正式评测样本库 |

关系：

```text
线上结构化输出失败
-> ai_failure_sample
-> 人工脱敏、复盘、标注
-> ai_eval_sample
-> 参与 harness 回归评测
```

所以 `ai_failure_sample` 更像“问题收集箱”，`ai_eval_sample` 更像“标准测试用例库”。

## 6. Harness 的存在感为什么变低了

当 prompt、样本、模型参数都进入数据库后，harness 代码本身会变得稳定。

变动部分：

```text
prompt 内容
prompt 版本
模型参数
样本输入
人工期望结果
样本类型
数据集版本
```

固定部分：

```text
读取 prompt
读取样本
批量调用模型
Pydantic 校验
actual vs expected 打分
统计 token
统计耗时
保存 eval_run
保存 eval_case_result
```

所以 harness 不是没用了，而是从“到处写逻辑的脚本”变成了“稳定的评测执行引擎”。

Java 类比：

```text
规则引擎代码稳定
规则内容存数据库
执行结果可查询
```

## 7. 数据库驱动 harness

Day16 抽出了公共执行器：

```text
day04_app/services/work_order_eval_runner.py
```

它负责：

```text
读取 ai_prompt_version
读取 ai_eval_sample
调用模型
解析结构化输出
计算准确率
生成 report
```

这样 PowerShell 脚本和 Celery Worker 都可以复用同一套执行逻辑。

脚本入口：

```text
evals/run_work_order_eval.py
```

异步 Worker 入口：

```text
day04_app.tasks.ai_tasks.execute_work_order_eval_task
```

## 8. 管理端查询接口

Day16 增加了 prompt、数据集、样本查询接口：

```http
GET /api/chat/prompt-versions
GET /api/chat/eval-datasets
GET /api/chat/eval-samples
```

也增加了评测结果查询接口：

```http
GET /api/chat/eval-runs
GET /api/chat/eval-runs/{run_id}/cases
```

页面关系：

```text
Prompt 列表页
    -> /prompt-versions

数据集列表页
    -> /eval-datasets

样本列表页
    -> /eval-samples

评测运行列表页
    -> /eval-runs

评测详情页
    -> /eval-runs/{run_id}/cases
```

## 9. HTTP 异步触发评测

Day16 新增接口：

```http
POST /api/chat/evals/work-order/run/async
```

请求体：

```json
{
  "prompt_name": "work_order_analysis",
  "prompt_version": "v2",
  "dataset_version": "work_order_analysis_v1"
}
```

返回：

```json
{
  "task_id": "3389xxxx",
  "status": "pending"
}
```

之后继续用已有任务状态接口轮询：

```http
GET /api/chat/tasks/{task_id}
```

任务成功后，再根据 `run_id` 查询评测结果：

```http
GET /api/chat/eval-runs
GET /api/chat/eval-runs/{run_id}/cases
```

## 10. 为什么评测触发要走异步任务

一次评测会批量调用模型。

如果同步接口直接执行：

- HTTP 请求容易超时。
- 前端一直等待，体验不好。
- 失败重试和补偿不好做。
- 无法复用已有 MQ/Worker 能力。

所以企业做法是：

```text
接口只提交任务
后台 Worker 执行评测
前端轮询任务状态
成功后查看评测结果
```

这和之前异步聊天任务一致。

## 11. 评测任务的企业链路

```text
管理端点击“运行评测”
-> FastAPI 创建 ai_async_task
-> 同事务写 ai_task_outbox
-> 投递 RabbitMQ
-> Celery Worker 领取 pending 任务
-> 执行 work_order_eval_runner
-> 写 ai_eval_run / ai_eval_case_result
-> 更新 ai_async_task 为 success
-> 管理端轮询 task_id
-> 根据 run_id 查看评测详情
```

## 12. 本次排查过的 KeyError 问题

异步评测任务第一次测试失败：

```text
error_type = WORKER_EXECUTION_ERROR
error_message = 工单评测异步任务执行异常：KeyError
```

根因：

```python
user_prompt_template.format(content=content)
```

数据库里的 prompt 模板包含 JSON 示例：

```json
{
  "category": "consult|complaint|repair|other"
}
```

Python `str.format(...)` 会把 JSON 示例中的 `{}` 也当成模板占位符解析，于是抛 `KeyError`。

修复方式：

```python
def render_user_prompt(user_prompt_template: str, content: str) -> str:
    if "{content}" in user_prompt_template:
        return user_prompt_template.replace("{content}", content)
    return f"{user_prompt_template}\n\n工单内容：{content}"
```

意义：

> prompt 模板入库后，不能随便用通用格式化方法，否则 JSON 示例、正则、大括号表达式都可能被误解析。

企业里通常会统一模板规范，例如：

```text
${content}
{{content}}
```

然后用专门渲染器处理。

## 13. 当前 Day16 已完成内容

已完成：

- 新增 `ai_prompt_version` 表。
- 新增 `ai_eval_dataset` 表。
- 新增 `ai_eval_sample` 表。
- 初始化当前 prompt v2 和 5 条评测样本。
- harness 从数据库读取 prompt 和样本。
- 抽取 `work_order_eval_runner.py` 公共执行器。
- 新增 prompt / 数据集 / 样本查询接口。
- 新增 eval_run / case_result 查询接口。
- 新增 HTTP 异步触发评测接口。
- 复用 `ai_async_task + ai_task_outbox + RabbitMQ + Celery Worker` 执行评测。
- 修复 prompt 模板 `{}` 导致的 `KeyError`。

## 14. 当前边界

当前仍是 PromptOps 雏形，不是完整平台。

还未完成：

- prompt 新增/编辑/发布接口。
- 样本新增/编辑/禁用接口。
- 失败样本转评测样本。
- prompt 发布审批。
- 上线前质量门禁。
- 多 prompt 对比评测。
- RAG / Tool / Agent Loop 的评测指标。

这些可以后续继续扩展。

## 15. 面试表达

可以这样讲：

> 我把 prompt、模型参数和评测样本从代码/jsonl 文件中抽离出来，设计了 ai_prompt_version、ai_eval_dataset 和 ai_eval_sample 主数据表。Harness 作为稳定执行引擎，从数据库读取 prompt 和样本，批量调用模型并计算结构化输出准确率、token 和耗时。

也可以这样讲：

> 为避免评测任务阻塞管理端请求，我将 harness 执行接入已有异步任务体系。接口提交任务后写入 ai_async_task 和 ai_task_outbox，通过 RabbitMQ 投递给 Celery Worker 执行，结果写入 ai_eval_run 和 ai_eval_case_result，前端通过 task_id 轮询状态并查看评测明细。

还可以这样讲：

> 在 prompt 模板入库过程中，我处理了 JSON 示例大括号与 Python str.format 占位符冲突的问题，改为只替换明确占位符，避免模板渲染误解析。

## 16. 当前项目标准

Day16 内容已经达到项目/面试亮点标准。

可以写进简历：

```text
设计并实现数据库驱动的 Prompt 评测 Harness，支持 prompt 版本、评测数据集、
样本主数据管理，基于 RabbitMQ/Celery 异步触发评测任务，
并将评测运行和样本级结果入库，支撑后续 PromptOps 管理端和上线前质量门禁。
```

