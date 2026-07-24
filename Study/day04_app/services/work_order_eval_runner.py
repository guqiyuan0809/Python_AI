"""
工单结构化分析评测执行器。

它是 harness 的核心执行引擎：读取数据库中的 prompt 和样本，批量调用模型，计算评测指标。
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any
from uuid import uuid4

from day04_app.common.exceptions import ModelCallException
from day04_app.database import SessionLocal
from day04_app.services.chat_service import analyze_work_order_structured_with_prompt
from day04_app.services.eval_master_service import (
    EvalSampleDTO,
    get_active_prompt_version,
    list_active_eval_samples,
)


CHINA_TZ = timezone(timedelta(hours=8))


def is_match(actual: dict[str, Any], expected: dict[str, Any], field_name: str) -> bool:
    return actual.get(field_name) == expected.get(field_name)


def accuracy(rows: list[dict[str, Any]], field_name: str) -> float:
    if not rows:
        return 0
    return round(sum(1 for row in rows if row[field_name]) / len(rows), 4)


def load_eval_context(prompt_name: str, prompt_version: str, dataset_version: str):
    db = SessionLocal()
    try:
        # 只在这里读取主数据；后续执行模型调用时不依赖打开的数据库连接。
        prompt = get_active_prompt_version(db, prompt_name, prompt_version)
        samples = list_active_eval_samples(db, dataset_version)
        return prompt, samples
    finally:
        db.close()


def run_work_order_eval(
    prompt_name: str,
    prompt_version: str,
    dataset_version: str,
) -> dict[str, Any]:
    """执行一次工单结构化分析评测，并返回完整报告。"""
    run_id = f"wo_eval_{datetime.now(CHINA_TZ).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    prompt, samples = load_eval_context(prompt_name, prompt_version, dataset_version)
    rows: list[dict[str, Any]] = []

    for sample in samples:
        row = run_one_sample(sample, prompt)
        rows.append(row)
        print_json_line(row)

    sample_count = len(rows)
    schema_valid_count = sum(1 for row in rows if row["schema_valid"])
    matched_rows = [row for row in rows if row["schema_valid"]]

    metrics = {
        "sample_count": sample_count,
        "schema_valid_rate": round(schema_valid_count / sample_count, 4) if sample_count else 0,
        "category_accuracy": accuracy(matched_rows, "category_match"),
        "risk_level_accuracy": accuracy(matched_rows, "risk_level_match"),
        "human_review_accuracy": accuracy(matched_rows, "human_review_match"),
        "avg_total_tokens": round(mean([row["total_tokens"] for row in rows]), 2) if rows else 0,
        "avg_cost_ms": round(mean([row["cost_ms"] for row in rows]), 2) if rows else 0,
        "failed_sample_ids": [row["sample_id"] for row in rows if not row["schema_valid"]],
    }
    return {
        "run_id": run_id,
        "run_at": datetime.now(CHINA_TZ).isoformat(),
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "dataset_version": dataset_version,
        "prompt_id": prompt.prompt_id,
        "dataset_source": "database",
        # 保存 prompt 快照，避免后续数据库中的 prompt 被修改后，历史评测无法还原。
        "prompt_snapshot": {
            "system_prompt": prompt.system_prompt,
            "user_prompt_template": prompt.user_prompt_template,
            "model": prompt.model,
            "temperature": prompt.temperature,
            "max_tokens": prompt.max_tokens,
        },
        "metrics": metrics,
        "rows": rows,
    }


def run_one_sample(sample: EvalSampleDTO, prompt) -> dict[str, Any]:
    start_time = time.perf_counter()
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "sample_type": sample.sample_type,
        "schema_valid": False,
        "category_match": False,
        "risk_level_match": False,
        "human_review_match": False,
        "total_tokens": 0,
        "cost_ms": 0,
        "error_type": None,
        "error_message": None,
    }
    try:
        result = analyze_work_order_structured_with_prompt(
            content=sample.input,
            system_prompt=prompt.system_prompt,
            user_prompt_template=prompt.user_prompt_template,
            model=prompt.model,
            temperature=prompt.temperature or 0.1,
            max_tokens=prompt.max_tokens or 600,
        )
        cost_ms = round((time.perf_counter() - start_time) * 1000)
        actual = result.analysis.model_dump()
        expected = sample.expected

        # Pydantic 能返回 result，说明 JSON 格式和硬性字段规则已经通过。
        row["schema_valid"] = True
        row["category_match"] = is_match(actual, expected, "category")
        row["risk_level_match"] = is_match(actual, expected, "risk_level")
        row["human_review_match"] = is_match(actual, expected, "need_human_review")
        row["total_tokens"] = result.total_tokens
        row["cost_ms"] = cost_ms
        row["actual"] = actual
        row["expected"] = expected
    except ModelCallException as exc:
        row["cost_ms"] = round((time.perf_counter() - start_time) * 1000)
        row["error_type"] = exc.error_type
        row["error_message"] = exc.message
    return row


def print_json_line(row: dict[str, Any]) -> None:
    import json

    print(json.dumps(row, ensure_ascii=False))
