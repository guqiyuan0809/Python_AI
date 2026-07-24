"""
Day16 Prompt 与评测样本主数据初始化脚本。

运行命令：
D:\\Pythoncode\\.venv\\Scripts\\python.exe init_day16_prompt_eval_master_data.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from day04_app.database import SessionLocal
from day04_app.models import AiEvalDataset, AiEvalSample, AiPromptVersion
from day04_app.services.chat_service import (
    WORK_ORDER_ANALYSIS_PROMPT_NAME,
    WORK_ORDER_ANALYSIS_SYSTEM_PROMPT,
    WORK_ORDER_ANALYSIS_USER_PROMPT_TEMPLATE,
    WORK_ORDER_ANALYSIS_PROMPT_VERSION,
)
from settings import settings


DATASET_PATH = Path(__file__).parent / "evals" / "datasets" / "work_order_analysis_v1.jsonl"
DATASET_ID = "dataset_work_order_analysis_v1"
DATASET_NAME = "work_order_analysis"
DATASET_VERSION = "work_order_analysis_v1"


def load_jsonl_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(json.loads(line))
    return samples


def infer_sample_type(sample_id: str) -> str:
    if "boundary" in sample_id:
        return "boundary"
    if "error" in sample_id:
        return "error"
    # 当前 5 条样本先按 normal 导入，后续可在管理端调整为 boundary/error。
    return "normal"


def upsert_prompt_version(db) -> None:
    prompt_id = (
        f"prompt_{WORK_ORDER_ANALYSIS_PROMPT_NAME}_{WORK_ORDER_ANALYSIS_PROMPT_VERSION}"
    )
    prompt = db.scalar(select(AiPromptVersion).where(AiPromptVersion.prompt_id == prompt_id))
    now = datetime.now()

    if not prompt:
        prompt = AiPromptVersion(
            prompt_id=prompt_id,
            prompt_name=WORK_ORDER_ANALYSIS_PROMPT_NAME,
            prompt_version=WORK_ORDER_ANALYSIS_PROMPT_VERSION,
            created_at=now,
        )
        db.add(prompt)

    # 重复执行时更新内容，保证数据库中的 prompt 与当前课程代码保持一致。
    prompt.description = "工单结构化分析 prompt，包含分类、风险等级和人工介入规则"
    prompt.system_prompt = WORK_ORDER_ANALYSIS_SYSTEM_PROMPT
    prompt.user_prompt_template = WORK_ORDER_ANALYSIS_USER_PROMPT_TEMPLATE
    prompt.model = settings.dashscope_model
    prompt.temperature = 0.1
    prompt.max_tokens = 600
    prompt.status = "active"
    prompt.created_by = "course_seed"
    prompt.updated_at = now


def upsert_dataset_and_samples(db, samples: list[dict[str, Any]]) -> None:
    dataset = db.scalar(select(AiEvalDataset).where(AiEvalDataset.dataset_id == DATASET_ID))
    now = datetime.now()

    if not dataset:
        dataset = AiEvalDataset(
            dataset_id=DATASET_ID,
            dataset_name=DATASET_NAME,
            dataset_version=DATASET_VERSION,
            created_at=now,
        )
        db.add(dataset)

    dataset.description = "工单结构化分析 v1 评测数据集，来自课程 jsonl 样本"
    dataset.sample_count = len(samples)
    dataset.status = "active"
    dataset.created_by = "course_seed"
    dataset.updated_at = now

    for sample in samples:
        eval_sample = db.scalar(
            select(AiEvalSample).where(AiEvalSample.sample_id == sample["sample_id"])
        )
        if not eval_sample:
            eval_sample = AiEvalSample(
                sample_id=sample["sample_id"],
                dataset_id=DATASET_ID,
                dataset_version=DATASET_VERSION,
                created_at=now,
            )
            db.add(eval_sample)

        # expected_json 保存人工期望结果；后续 harness 会用它与模型 actual 做对比。
        eval_sample.sample_type = infer_sample_type(sample["sample_id"])
        eval_sample.input_text = sample["input"]
        eval_sample.expected_json = json.dumps(sample["expected"], ensure_ascii=False)
        eval_sample.source_type = "jsonl_seed"
        eval_sample.source_ref_id = str(DATASET_PATH)
        eval_sample.status = "active"
        eval_sample.created_by = "course_seed"
        eval_sample.updated_at = now


def main() -> None:
    samples = load_jsonl_samples(DATASET_PATH)
    db = SessionLocal()
    try:
        upsert_prompt_version(db)
        upsert_dataset_and_samples(db, samples)
        db.commit()
        print(
            f"Day16 初始化完成：prompt={WORK_ORDER_ANALYSIS_PROMPT_NAME}:{WORK_ORDER_ANALYSIS_PROMPT_VERSION}，"
            f"dataset={DATASET_VERSION}，samples={len(samples)}"
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
