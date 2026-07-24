# Day15 AI 结构化输出稳定性与 Harness 评测笔记

## 1. Day15 主题

Day15 的核心不是继续做普通问答，而是把 AI 输出从“自然语言回答”升级到“可校验、可追踪、可评测、可优化”的企业工程链路。

本日主题：

```text
结构化输出
-> Pydantic DTO 校验
-> 错误类型分类
-> 失败样本沉淀
-> Harness 离线评测
-> Prompt 版本留痕
-> 评测结果入库
```

这一天解决的问题是：

> 模型不是每次都稳定输出我们想要的 JSON，所以企业项目不能只靠“提示词写好一点”，还要有校验、修复、错误分类、样本沉淀和离线评测机制。

## 2. 结构化输出和普通问答的区别

普通问答更关注“回答内容是否像人话”。

结构化输出更关注“返回结果能不能被系统消费”。

例如工单分析接口要求模型返回：

```json
{
  "category": "complaint",
  "risk_level": "high",
  "summary": "游客投诉停车场出口拥堵并发生争吵",
  "suggestions": ["安排现场人员疏导", "客服跟进投诉解释"],
  "need_human_review": true,
  "confidence": 0.85
}
```

这里的重点不是“文字写得漂亮”，而是：

- `category` 必须是固定枚举。
- `risk_level` 必须是固定枚举。
- `suggestions` 必须是数组。
- `need_human_review` 必须是布尔值。
- `confidence` 必须是数值范围。

## 3. Pydantic 在结构化输出中的作用

`WorkOrderAnalysisResult` 继承了 Pydantic 的 `BaseModel`。

它相当于 Java 里的 DTO + 字段校验规则。

核心代码思想：

```python
json_text = extract_json_object(raw_text)
analysis = WorkOrderAnalysisResult.model_validate_json(json_text)
```

含义：

1. `extract_json_object(raw_text)`：从模型原始字符串中抽取 JSON 对象。
2. `model_validate_json(json_text)`：把 JSON 转成 DTO，并校验字段类型、枚举值、范围。

类比 Java：

```java
WorkOrderAnalysisResult dto =
    objectMapper.readValue(jsonText, WorkOrderAnalysisResult.class);

validator.validate(dto);
```

区别是 Pydantic 把“反序列化 + 校验”结合得更紧。

## 4. 输出修复机制

模型第一次返回的结构可能不合法，比如：

- 不是 JSON。
- JSON 外面包了 Markdown。
- 枚举值不在允许范围内。
- `suggestions` 不是数组。
- `confidence` 超出 0 到 1。

当前设计：

```text
第一次调用模型
-> 抽取 JSON
-> Pydantic 校验
-> 如果 JSON/字段错误，允许调用一次 repair prompt
-> 再次校验
-> 仍失败则抛业务异常
```

为什么只修复一次？

因为企业系统不能无限让模型自我修复，否则会造成：

- 接口耗时不可控。
- token 成本不可控。
- 失败问题被隐藏。

## 5. 错误类型 error_type

Day15 增加了固定错误类型，用于区分不同失败来源。

典型错误类型：

```text
MODEL_CALL_FAILED
STRUCTURED_JSON_INVALID
STRUCTURED_FIELD_INVALID
TASK_TIMEOUT
WORKER_EXECUTION_ERROR
```

字段落到了：

- `chat_message.error_type`
- `ai_call_log.error_type`
- `ai_async_task.error_type`

意义：

```text
error_message 是给开发排查细节的
error_type 是给系统聚合统计和治理用的
```

例如后续可以统计：

```sql
SELECT error_type, COUNT(*)
FROM ai_call_log
WHERE status = 'error'
GROUP BY error_type;
```

## 6. ai_failure_sample 失败样本表

`ai_failure_sample` 不是评测结果表，而是线上失败样本池。

它保存：

- trace_id
- task_id
- session_id
- call_type
- model
- schema_type
- schema_version
- error_type
- raw_text
- validation_error

它的定位：

```text
线上结构化输出失败
-> 进入 ai_failure_sample
-> 人工复盘、脱敏、标注
-> 转成 ai_eval_sample
-> 加入 Harness 回归评测
```

所以它和 `ai_eval_case_result` 不冲突。

区别：

| 表 | 作用 |
|---|---|
| `ai_failure_sample` | 线上失败样本池 |
| `ai_eval_case_result` | 某次评测运行中每条样本的执行结果 |

## 7. Harness 是什么

Harness 可以理解为 AI Prompt 的离线评测框架。

它不是线上接口的一部分，不是每次用户请求都执行。

它通常在以下场景使用：

- 修改 prompt 后。
- 更换模型后。
- 修改 DTO 字段后。
- 上线前回归测试。
- 收集新失败样本后。

当前最小 harness 做了三类评测：

```text
Pydantic 格式校验
业务字段准确率
token / 耗时统计
```

当前脚本：

- `evals/run_work_order_eval.py`
- `evals/datasets/work_order_analysis_v1.jsonl`

运行命令：

```powershell
cd D:\Pythoncode\Study
D:\Pythoncode\.venv\Scripts\python.exe evals\run_work_order_eval.py
```

## 8. 三类评测样本

### 正常样本

用于验证主流程是否准确。

例如：

```text
游客咨询今天几点闭园
```

期望：

```json
{
  "category": "consult",
  "risk_level": "low",
  "need_human_review": false
}
```

### 边界样本

用于验证模糊业务边界是否稳定。

例如：

```text
东门闸机扫码后无法通行，多名游客排队，但没有争吵
```

期望是：

```text
repair + medium + need_human_review=true
```

它主要测试：

- medium 和 high 的边界。
- 是否需要人工介入。
- prompt 规则是否足够明确。

### 错误样本

用于验证模型不要乱答，系统能兜底。

例如：

```text
帮我看看这个问题
```

因为没有具体业务信息，所以不能强行分类成投诉或维修。

期望：

```text
other + low + need_human_review=true
```

错误样本还可以包括：

- 空输入。
- 无意义输入。
- Prompt 注入。
- 模型返回非 JSON。
- 字段枚举非法。

## 9. Prompt v2 优化

Harness 暴露出两个问题：

```text
闸机多人排队：模型容易判 high
卫生间异味投诉：模型容易判 low
```

原因是 v1 prompt 只说了字段枚举，没有说清楚风险等级业务边界。

v2 prompt 增加：

```text
low：
普通咨询、信息确认、无现场处置诉求、无服务影响

medium：
投诉或报修需要工作人员跟进，存在卫生、设备、排队、体验影响，
但没有争吵、伤害、安全隐患或明显舆情风险

high：
已经发生争吵冲突、人身安全风险、大面积服务中断、
严重拥堵失控、可能引发舆情，或必须立即升级管理人员处置
```

这一步的核心思想：

> 不是发现模型错了就凭感觉改 prompt，而是根据 harness 暴露的失败样本，把业务边界规则显式写入 prompt，然后重新评测。

## 10. 评测结果留痕

当前最小版评测结果会生成：

```text
evals/reports/wo_eval_xxx.json
```

并进一步升级为数据库表：

```text
ai_eval_run
ai_eval_case_result
```

`ai_eval_run` 保存一次评测汇总：

- prompt_name
- prompt_version
- dataset_version
- sample_count
- schema_valid_rate
- category_accuracy
- risk_level_accuracy
- human_review_accuracy
- avg_total_tokens
- avg_cost_ms

`ai_eval_case_result` 保存每条样本明细：

- run_id
- sample_id
- schema_valid
- category_match
- risk_level_match
- human_review_match
- expected_json
- actual_json
- row_json

## 11. Day15 当前边界

Day15 已经完成“可运行雏形”，但不是完整企业平台。

已完成：

- 结构化输出 DTO。
- JSON 抽取。
- Pydantic 校验。
- 一次输出修复。
- 错误类型分类。
- 失败样本表。
- 最小 harness。
- prompt v2 优化。
- 文件报告。
- 评测结果入库雏形。

还未完成：

- prompt 内容从数据库读取。
- 样本集从数据库读取。
- prompt_version 主数据表。
- eval_dataset / eval_sample 主数据表。
- 管理端接口。
- 前端页面。
- 上线前质量门禁。

这些内容从 Day16 开始继续补。

## 12. 面试表达

可以这样讲：

> 我在 Python AI 服务中实现了结构化输出治理。模型输出不直接信任，而是先抽取 JSON，再通过 Pydantic DTO 做字段、枚举和值范围校验。对于非 JSON 或字段错误，系统只允许一次修复调用，避免无限重试导致接口耗时和 token 成本失控。

也可以这样讲：

> 为了避免 prompt 调整凭感觉，我建设了一个最小离线 harness。它基于固定样本集批量调用模型，统计 schema 通过率、分类准确率、风险等级准确率、人工介入准确率、平均 token 和耗时。每次 prompt 调整后都会留下 eval_run 和 case_result 记录，方便对比历史版本效果。

还可以这样讲：

> 线上结构化输出失败会沉淀到 ai_failure_sample，后续人工脱敏、标注后可加入正式评测样本集，形成“线上失败 -> 样本沉淀 -> prompt 优化 -> harness 回归”的闭环。

## 13. 当前项目/面试标准

Day15 内容已经达到“项目可讲标准”。

可以写进简历的方向：

```text
设计 AI 结构化输出治理链路，基于 Pydantic 实现 JSON 结果强校验，
并沉淀失败样本与离线评测 harness，支持 prompt 版本优化和质量回归。
```

但如果要达到“完整企业平台标准”，还需要继续完成：

- prompt 版本主数据管理。
- 数据集和样本管理。
- 评测结果查询接口。
- prompt 发布流程。
- 上线前评测门禁。

