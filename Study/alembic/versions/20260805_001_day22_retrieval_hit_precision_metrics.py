"""Day22 add Hit@K and Precision@K metrics for retrieval eval.

Revision ID: 20260805_001
Revises: 20260803_002
Create Date: 2026-08-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260805_001"
down_revision: str | None = "20260803_002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _column_names(table_name: str) -> set[str]:
    return {column["name"] for column in inspect(op.get_bind()).get_columns(table_name)}


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    if column.name not in _column_names(table_name):
        op.add_column(table_name, column)


def upgrade() -> None:
    _add_column_if_missing(
        "knowledge_retrieval_eval_run",
        sa.Column(
            "total_expected_segment_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="可回答样本中人工标注的期望原始段总数，用于计算真正的 Recall@K",
        ),
    )
    _add_column_if_missing(
        "knowledge_retrieval_eval_run",
        sa.Column(
            "total_hit_segment_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="Top-K 实际命中的期望原始段去重总数，用于计算真正的 Recall@K",
        ),
    )
    _add_column_if_missing(
        "knowledge_retrieval_eval_run",
        sa.Column(
            "total_retrieved_chunk_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="可回答样本成功检索返回的 chunk 总数，用于计算 Precision@K",
        ),
    )
    _add_column_if_missing(
        "knowledge_retrieval_eval_run",
        sa.Column(
            "total_relevant_retrieved_chunk_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
            comment="可回答样本 Top-K 中包含期望原始段的 chunk 总数，用于计算 Precision@K",
        ),
    )
    _add_column_if_missing(
        "knowledge_retrieval_eval_run",
        sa.Column(
            "hit_at_k",
            sa.Float(),
            nullable=True,
            comment="Hit@K：可回答样本中 Top-K 至少命中一个正确证据的样本比例",
        ),
    )
    _add_column_if_missing(
        "knowledge_retrieval_eval_run",
        sa.Column(
            "precision_at_k",
            sa.Float(),
            nullable=True,
            comment="Precision@K：Top-K 中正确 chunk 数 / 返回 chunk 总数",
        ),
    )

    _add_column_if_missing(
        "knowledge_retrieval_eval_case_result",
        sa.Column(
            "hit_segment_count",
            sa.Integer(),
            nullable=True,
            comment="本样本 Top-K 命中的期望原始段去重数量；无答案样本为空",
        ),
    )
    _add_column_if_missing(
        "knowledge_retrieval_eval_case_result",
        sa.Column(
            "expected_segment_count",
            sa.Integer(),
            nullable=True,
            comment="本样本人工标注的期望原始段数量；无答案样本为空",
        ),
    )
    _add_column_if_missing(
        "knowledge_retrieval_eval_case_result",
        sa.Column(
            "relevant_retrieved_chunk_count",
            sa.Integer(),
            nullable=True,
            comment="本样本 Top-K 中包含期望原始段的 chunk 数量；无答案样本为空",
        ),
    )
    _add_column_if_missing(
        "knowledge_retrieval_eval_case_result",
        sa.Column(
            "precision_at_k",
            sa.Float(),
            nullable=True,
            comment="本样本 Precision@K：正确 chunk 数 / 实际返回 chunk 数；无答案样本为空",
        ),
    )


def downgrade() -> None:
    case_columns = _column_names("knowledge_retrieval_eval_case_result")
    for column_name in (
        "precision_at_k",
        "relevant_retrieved_chunk_count",
        "expected_segment_count",
        "hit_segment_count",
    ):
        if column_name in case_columns:
            op.drop_column("knowledge_retrieval_eval_case_result", column_name)

    run_columns = _column_names("knowledge_retrieval_eval_run")
    for column_name in (
        "precision_at_k",
        "hit_at_k",
        "total_relevant_retrieved_chunk_count",
        "total_retrieved_chunk_count",
        "total_hit_segment_count",
        "total_expected_segment_count",
    ):
        if column_name in run_columns:
            op.drop_column("knowledge_retrieval_eval_run", column_name)
