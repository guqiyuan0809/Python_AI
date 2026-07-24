"""
给 Python AI 课程相关 MySQL 表和字段补充中文备注。

运行命令：
D:\\Pythoncode\\.venv\\Scripts\\python.exe migrate_add_chinese_comments.py

说明：
- 只修改表备注和字段备注，不修改业务数据。
- MySQL 修改字段 COMMENT 需要重新声明字段类型，所以脚本会先读取当前字段定义再拼接 COMMENT。
- 如果某张表或某个字段在当前数据库中不存在，会自动跳过，避免本地版本差异导致脚本中断。
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text

from day04_app.database import engine


TABLE_COMMENTS: dict[str, str] = {
    "chat_session": "AI 会话主表，保存用户一次长期对话的基础信息、标题、摘要和状态",
    "chat_message": "AI 会话消息表，保存用户消息、助手回复、流式状态、token 用量和失败信息",
    "chat_session_summary": "AI 会话摘要表，保存长上下文压缩后的摘要版本和覆盖到的消息位置",
    "ai_call_log": "AI 调用日志表，记录模型调用链路、token、耗时、成功失败和错误类型",
    "ai_async_task": "AI 异步任务表，保存业务 task_id、任务状态、重试次数和最终结果",
    "ai_task_outbox": "AI 任务本地消息表，用于解决数据库事务与 MQ 投递的双写一致性问题",
    "ai_structured_result": "AI 结构化结果表，保存模型输出并通过 DTO 校验后的标准 JSON 结果",
    "ai_failure_sample": "AI 失败样本表，用于沉淀线上结构化输出失败案例并支撑 prompt 优化",
    "ai_prompt_version": "AI Prompt 版本表，用于保存每个业务场景下的 prompt 内容和模型参数",
    "ai_eval_dataset": "AI 评测数据集表，用于管理某个业务评测集合的版本信息",
    "ai_eval_sample": "AI 评测样本表，用于保存人工标注后的标准输入和期望输出",
    "ai_eval_run": "AI 评测运行表，记录一次 harness 对某个 prompt 和数据集的汇总评测指标",
    "ai_eval_case_result": "AI 评测样本结果表，记录一次评测中每条样本的期望值、实际值和命中情况",
}


COLUMN_COMMENTS: dict[str, dict[str, str]] = {
    "chat_session": {
        "id": "数据库自增主键，仅用于内部存储",
        "session_id": "会话业务 ID，接口和 Java 服务使用该 ID 关联会话",
        "user_id": "用户 ID，当前学习阶段可为空，后续接入真实登录后使用",
        "title": "会话标题，可由模型生成，也可由用户手动修改",
        "summary": "当前会话摘要，用于长上下文压缩和后续模型调用",
        "status": "会话状态，active 表示正常，archived 表示已归档",
        "created_at": "会话创建时间",
        "updated_at": "会话最后更新时间",
    },
    "chat_message": {
        "id": "数据库自增主键，仅用于内部存储",
        "message_id": "消息业务 ID，用于定位某一条用户消息或助手回复",
        "session_id": "所属会话 ID，对应 chat_session.session_id",
        "trace_id": "链路追踪 ID，由 Java 或 Python 生成并贯穿一次请求",
        "stream_id": "流式响应 ID，用于定位一次 SSE 流式输出过程",
        "role": "消息角色，user 表示用户，assistant 表示 AI 助手",
        "content": "消息正文内容，保存用户问题、AI 回答或失败时的部分输出",
        "model": "本次消息对应的模型名称",
        "prompt_tokens": "本次模型调用消耗的提示词 token 数",
        "completion_tokens": "本次模型调用生成的回答 token 数",
        "total_tokens": "本次模型调用总 token 数",
        "status": "消息状态，pending、streaming、success、error 等",
        "error_type": "失败类型，用于区分模型调用失败、JSON 不合法、字段校验失败等",
        "error_message": "失败详情，便于排查问题，不直接暴露给普通用户",
        "created_at": "消息创建时间",
    },
    "chat_session_summary": {
        "id": "数据库自增主键，仅用于内部存储",
        "summary_id": "摘要业务 ID，用于唯一定位一条摘要版本",
        "session_id": "所属会话 ID，对应 chat_session.session_id",
        "summary": "压缩后的会话摘要内容，用于长上下文续聊",
        "summary_until_message_id": "摘要已经覆盖到的最后一条消息 ID",
        "version": "同一会话下的摘要版本号，数字越大表示越新",
        "model": "生成摘要时使用的模型名称",
        "status": "摘要生成状态，当前主要为 success，后续可扩展 error",
        "error_message": "摘要生成失败时的错误信息",
        "created_at": "摘要创建时间",
    },
    "ai_call_log": {
        "id": "数据库自增主键，仅用于内部存储",
        "call_id": "AI 调用业务 ID，使用雪花 ID，便于跨服务追踪",
        "trace_id": "链路追踪 ID，用于串联 Java 请求、Python 接口和模型调用",
        "session_id": "会话 ID，用于从调用日志反查对应会话",
        "message_id": "关联的消息 ID，用于从日志定位最终展示内容",
        "call_type": "调用类型，例如普通会话、流式会话、摘要生成、结构化分析",
        "model": "本次调用使用的模型名称",
        "prompt_tokens": "提示词 token 消耗",
        "completion_tokens": "模型输出 token 消耗",
        "total_tokens": "总 token 消耗",
        "cost_ms": "模型调用耗时，单位毫秒",
        "status": "调用状态，success 表示成功，error 表示失败",
        "error_type": "失败类型，用于聚合统计不同错误来源",
        "error_message": "失败详情，用于排查模型、网络、结构化输出等问题",
        "created_at": "日志创建时间",
    },
    "ai_async_task": {
        "id": "数据库自增主键，仅用于内部存储",
        "task_id": "异步任务业务 ID，前端和 Java 后端通过它轮询任务状态",
        "trace_id": "链路追踪 ID，用于串联任务提交、MQ 投递和 Worker 执行",
        "session_id": "任务所属会话 ID",
        "message_id": "任务最终关联的 assistant 消息 ID",
        "broker_task_id": "Celery 投递到 RabbitMQ 后生成的内部消息 ID，仅用于调度排查",
        "task_type": "任务类型，例如 session_chat、work_order_analysis",
        "input_text": "任务输入文本，保存用户本次提交的问题或工单内容",
        "result_text": "任务最终自然语言结果，结构化任务通常另存 ai_structured_result",
        "model": "任务执行使用的模型名称",
        "prompt_tokens": "任务执行消耗的提示词 token 数",
        "completion_tokens": "任务执行生成的回答 token 数",
        "total_tokens": "任务执行总 token 数",
        "cost_ms": "任务执行耗时，单位毫秒",
        "status": "任务状态，pending、running、success、error 等",
        "retry_count": "当前自动重试次数",
        "max_retries": "最大自动重试次数，超过后进入 error 等待人工处理",
        "error_type": "任务失败类型，例如模型失败、结构化字段失败、任务超时",
        "error_message": "任务失败详情",
        "created_at": "任务创建时间",
        "updated_at": "任务最后更新时间",
    },
    "ai_task_outbox": {
        "id": "数据库自增主键，仅用于内部存储",
        "event_id": "本地消息事件 ID，一次任务可因重试生成多条事件",
        "task_id": "关联的异步任务 ID，对应 ai_async_task.task_id",
        "event_type": "事件类型，用于映射到具体 Celery Worker 任务",
        "payload": "投递给 Worker 的最小参数 JSON 字符串",
        "status": "本地消息状态，pending 表示待投递，published 表示已投递",
        "publish_retry_count": "本地消息投递失败后的补偿重试次数",
        "available_at": "消息可投递时间，用于指数退避和延迟重试",
        "error_message": "消息投递失败时的错误信息",
        "published_at": "消息成功投递到 MQ 的时间",
        "created_at": "本地消息创建时间",
        "updated_at": "本地消息最后更新时间",
    },
    "ai_structured_result": {
        "id": "数据库自增主键，仅用于内部存储",
        "result_id": "结构化结果业务 ID，用于定位一份标准 JSON 结果",
        "task_id": "关联的异步任务 ID，同步接口结果可为空",
        "trace_id": "链路追踪 ID，用于关联请求、任务、日志和结构化结果",
        "session_id": "所属会话 ID",
        "message_id": "关联消息 ID",
        "business_type": "业务类型，例如 work_order、contract、audit",
        "business_id": "业务主键 ID，例如真实工单 ID 或合同 ID",
        "schema_type": "结构化结果类型，例如 work_order_analysis",
        "schema_version": "结构化 DTO 版本，用于兼容后续字段升级",
        "result_json": "通过 Pydantic 校验后的标准 JSON 结果",
        "status": "结构化结果状态，success 或 error",
        "error_message": "结构化结果生成或校验失败详情",
        "created_at": "结构化结果创建时间",
        "updated_at": "结构化结果最后更新时间",
    },
    "ai_failure_sample": {
        "id": "数据库自增主键，仅用于内部存储",
        "sample_id": "失败样本业务 ID，后续可进入评测数据集",
        "trace_id": "链路追踪 ID，用于回溯线上失败请求",
        "task_id": "关联的异步任务 ID",
        "session_id": "关联的会话 ID",
        "message_id": "关联的消息 ID",
        "call_type": "失败来源类型，例如结构化分析或异步任务",
        "model": "失败时使用的模型名称",
        "schema_type": "失败样本对应的结构化结果类型",
        "schema_version": "失败样本对应的结构化 DTO 版本",
        "error_type": "失败类型，例如 JSON 不合法或字段校验失败",
        "error_message": "失败详情摘要",
        "raw_text": "模型原始输出文本，用于复盘和 prompt 优化",
        "validation_error": "Pydantic 校验错误详情",
        "created_at": "失败样本创建时间",
    },
    "ai_prompt_version": {
        "id": "数据库自增主键，仅用于内部存储",
        "prompt_id": "Prompt 版本业务 ID，由 prompt 名称和版本组合生成",
        "prompt_name": "Prompt 名称，例如 work_order_analysis",
        "prompt_version": "Prompt 版本号，例如 v1、v2",
        "description": "Prompt 版本说明，用于描述本次规则变化",
        "system_prompt": "系统提示词内容，定义模型角色、输出约束和业务规则",
        "user_prompt_template": "用户提示词模板，使用占位符拼接真实业务输入",
        "model": "该 Prompt 推荐或评测时使用的模型名称",
        "temperature": "模型温度参数，越低输出越稳定",
        "max_tokens": "模型最大输出 token 数",
        "status": "Prompt 状态，draft 表示草稿，active 表示当前可用",
        "created_by": "创建人或初始化来源",
        "created_at": "Prompt 版本创建时间",
        "updated_at": "Prompt 版本最后更新时间",
    },
    "ai_eval_dataset": {
        "id": "数据库自增主键，仅用于内部存储",
        "dataset_id": "评测数据集业务 ID",
        "dataset_name": "评测数据集名称，例如 work_order_analysis",
        "dataset_version": "评测数据集版本，例如 work_order_analysis_v1",
        "description": "评测数据集说明",
        "sample_count": "当前数据集样本数量",
        "status": "数据集状态，active 表示可用于评测",
        "created_by": "创建人或初始化来源",
        "created_at": "数据集创建时间",
        "updated_at": "数据集最后更新时间",
    },
    "ai_eval_sample": {
        "id": "数据库自增主键，仅用于内部存储",
        "sample_id": "评测样本业务 ID",
        "dataset_id": "所属评测数据集 ID，对应 ai_eval_dataset.dataset_id",
        "dataset_version": "所属数据集版本，便于按版本筛选样本",
        "sample_type": "样本类型，normal 正常样本，boundary 边界样本，error 错误样本",
        "input_text": "评测输入文本，即模拟用户提交的工单内容",
        "expected_json": "人工标注的期望输出 JSON",
        "source_type": "样本来源，例如人工录入、jsonl 初始化、失败样本转入",
        "source_ref_id": "样本来源引用 ID，例如失败样本 ID 或文件路径",
        "status": "样本状态，active 表示参与评测",
        "created_by": "创建人或初始化来源",
        "created_at": "样本创建时间",
        "updated_at": "样本最后更新时间",
    },
    "ai_eval_run": {
        "id": "数据库自增主键，仅用于内部存储",
        "run_id": "评测运行 ID，唯一标识一次 harness 执行",
        "prompt_name": "被评测的 prompt 名称",
        "prompt_version": "被评测的 prompt 版本号",
        "dataset_version": "本次评测使用的数据集版本",
        "sample_count": "本次评测样本总数",
        "schema_valid_rate": "结构化 DTO 校验通过率",
        "category_accuracy": "工单分类字段准确率",
        "risk_level_accuracy": "风险等级字段准确率",
        "human_review_accuracy": "是否人工介入字段准确率",
        "avg_total_tokens": "单条样本平均 token 消耗",
        "avg_cost_ms": "单条样本平均耗时，单位毫秒",
        "metrics_json": "完整汇总指标 JSON，便于后续扩展更多指标",
        "created_at": "评测运行记录创建时间",
    },
    "ai_eval_case_result": {
        "id": "数据库自增主键，仅用于内部存储",
        "run_id": "所属评测运行 ID，对应 ai_eval_run.run_id",
        "sample_id": "样本 ID，当前对应 jsonl 样本，后续可关联 ai_eval_sample",
        "schema_valid": "结构化 DTO 校验是否通过，1 表示通过，0 表示失败",
        "category_match": "工单分类是否命中人工期望值，1 表示命中",
        "risk_level_match": "风险等级是否命中人工期望值，1 表示命中",
        "human_review_match": "是否人工介入字段是否命中人工期望值，1 表示命中",
        "total_tokens": "该样本模型调用总 token 消耗",
        "cost_ms": "该样本模型调用耗时，单位毫秒",
        "error_type": "该样本评测失败类型",
        "error_message": "该样本评测失败详情",
        "expected_json": "该样本人工标注的期望结果 JSON 快照",
        "actual_json": "该样本模型实际输出结果 JSON 快照",
        "row_json": "该样本完整评测明细 JSON 快照",
        "created_at": "样本评测结果创建时间",
    },
}


def quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def quote_sql_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def safe_charset_name(value: str | None) -> str:
    if not value:
        return ""
    if not re.fullmatch(r"[0-9A-Za-z_]+", value):
        raise ValueError(f"不安全的字符集或排序规则名称：{value}")
    return value


def build_default_sql(column: dict[str, Any]) -> str:
    default_value = column["COLUMN_DEFAULT"]
    if default_value is None:
        return ""

    default_text = str(default_value)
    upper_default = default_text.upper()

    if upper_default in {"NULL", "CURRENT_TIMESTAMP", "CURRENT_TIMESTAMP()"}:
        return f" DEFAULT {default_text}"
    if upper_default.startswith("CURRENT_TIMESTAMP"):
        return f" DEFAULT {default_text}"

    return f" DEFAULT {quote_sql_string(default_text)}"


def build_column_definition(column: dict[str, Any], comment: str) -> str:
    column_name = column["COLUMN_NAME"]
    column_type = column["COLUMN_TYPE"]
    nullable_sql = "NULL" if column["IS_NULLABLE"] == "YES" else "NOT NULL"
    charset = safe_charset_name(column.get("CHARACTER_SET_NAME"))
    collation = safe_charset_name(column.get("COLLATION_NAME"))
    extra = column.get("EXTRA") or ""

    charset_sql = f" CHARACTER SET {charset}" if charset else ""
    collation_sql = f" COLLATE {collation}" if collation else ""
    default_sql = build_default_sql(column)
    extra_sql = f" {extra}" if extra else ""

    return (
        f"{quote_identifier(column_name)} {column_type}{charset_sql}{collation_sql} "
        f"{nullable_sql}{default_sql}{extra_sql} COMMENT {quote_sql_string(comment)}"
    )


def get_current_database(conn) -> str:
    database_name = conn.execute(text("SELECT DATABASE()")).scalar_one_or_none()
    if not database_name:
        raise RuntimeError("当前连接没有选中数据库，请检查 DATABASE_URL 是否包含数据库名")
    return str(database_name)


def table_exists(conn, schema_name: str, table_name: str) -> bool:
    count = conn.execute(
        text(
            """
            SELECT COUNT(*)
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = :schema_name
              AND TABLE_NAME = :table_name
            """
        ),
        {"schema_name": schema_name, "table_name": table_name},
    ).scalar_one()
    return int(count) > 0


def load_columns(conn, schema_name: str, table_name: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        text(
            """
            SELECT
                COLUMN_NAME,
                COLUMN_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                EXTRA,
                CHARACTER_SET_NAME,
                COLLATION_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = :schema_name
              AND TABLE_NAME = :table_name
            ORDER BY ORDINAL_POSITION
            """
        ),
        {"schema_name": schema_name, "table_name": table_name},
    ).mappings()
    return {str(row["COLUMN_NAME"]): dict(row) for row in rows}


def main() -> None:
    if engine.dialect.name not in {"mysql", "mariadb"}:
        raise RuntimeError("本脚本只适用于 MySQL/MariaDB 字段 COMMENT 语法")

    applied_columns = 0
    skipped_columns = 0

    with engine.begin() as conn:
        # 确保本次连接使用 utf8mb4，避免中文备注在连接层出现编码问题。
        conn.exec_driver_sql("SET NAMES utf8mb4")
        schema_name = get_current_database(conn)

        for table_name, table_comment in TABLE_COMMENTS.items():
            if not table_exists(conn, schema_name, table_name):
                print(f"跳过不存在的表：{table_name}")
                continue

            conn.exec_driver_sql(
                f"ALTER TABLE {quote_identifier(table_name)} COMMENT={quote_sql_string(table_comment)}"
            )
            print(f"已更新表备注：{table_name}")

            existing_columns = load_columns(conn, schema_name, table_name)
            for column_name, column_comment in COLUMN_COMMENTS.get(table_name, {}).items():
                column = existing_columns.get(column_name)
                if not column:
                    skipped_columns += 1
                    print(f"  跳过不存在的字段：{table_name}.{column_name}")
                    continue

                column_definition = build_column_definition(column, column_comment)
                ddl = (
                    f"ALTER TABLE {quote_identifier(table_name)} "
                    f"MODIFY COLUMN {column_definition}"
                )
                # MySQL 修改字段备注必须 MODIFY COLUMN；这里复用当前字段定义，避免破坏字段类型。
                conn.exec_driver_sql(ddl)
                applied_columns += 1
                print(f"  已更新字段备注：{table_name}.{column_name}")

    print(f"中文备注迁移完成：成功字段数={applied_columns}，跳过字段数={skipped_columns}")


if __name__ == "__main__":
    main()
