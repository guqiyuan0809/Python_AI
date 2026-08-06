"""Day22 add RAG retrieval evaluation run records.

Revision ID: 20260803_001
Revises: 20260802_001
Create Date: 2026-08-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260803_001"
down_revision: str | None = "20260802_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_index_if_missing(
    table_name: str,
    index_name: str,
    columns: list[str],
) -> None:
    existing_indexes = {
        index["name"] for index in inspect(op.get_bind()).get_indexes(table_name)
    }
    if index_name not in existing_indexes:
        op.create_index(index_name, table_name, columns)


def upgrade() -> None:
    existing_tables = set(inspect(op.get_bind()).get_table_names())
    if "knowledge_retrieval_eval_run" not in existing_tables:
        op.create_table(
            "knowledge_retrieval_eval_run",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
            sa.Column("run_id", sa.String(length=64), nullable=False, comment="检索评测运行业务唯一 ID"),
            sa.Column("dataset_id", sa.String(length=64), nullable=False, comment="本次使用的检索评测数据集业务 ID 快照"),
            sa.Column("document_id", sa.String(length=64), nullable=False, comment="本次评测的知识库文档业务 ID"),
            sa.Column("document_version_id", sa.String(length=64), nullable=False, comment="本次被测的文档索引版本业务 ID"),
            sa.Column("retrieval_top_k", sa.Integer(), nullable=False, comment="本次检索统一使用的 Top-K"),
            sa.Column("score_threshold", sa.Float(), nullable=True, comment="可选的可回答分数阈值"),
            sa.Column("embedding_model", sa.String(length=128), nullable=True, comment="本次实际调用的查询 Embedding 模型"),
            sa.Column("vector_dimension", sa.Integer(), nullable=True, comment="本次查询向量维度"),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="running", comment="运行状态：running、success、partial_success、error"),
            sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0", comment="参与评测的 active 样本总数"),
            sa.Column("success_count", sa.Integer(), nullable=False, server_default="0", comment="成功完成检索的样本数"),
            sa.Column("error_count", sa.Integer(), nullable=False, server_default="0", comment="检索异常的样本数"),
            sa.Column("answerable_sample_count", sa.Integer(), nullable=False, server_default="0", comment="可回答样本数"),
            sa.Column("answerable_hit_count", sa.Integer(), nullable=False, server_default="0", comment="Top-K 命中正确证据的可回答样本数"),
            sa.Column("recall_at_k", sa.Float(), nullable=True, comment="可回答样本 Recall@K"),
            sa.Column("mrr_at_k", sa.Float(), nullable=True, comment="可回答样本 MRR@K"),
            sa.Column("no_answer_sample_count", sa.Integer(), nullable=False, server_default="0", comment="无答案样本数"),
            sa.Column("no_answer_false_positive_count", sa.Integer(), nullable=True, comment="无答案误放行样本数"),
            sa.Column("no_answer_false_positive_rate", sa.Float(), nullable=True, comment="无答案误放行率"),
            sa.Column("no_answer_avg_top_score", sa.Float(), nullable=True, comment="无答案样本 Top-1 分数均值"),
            sa.Column("elapsed_ms", sa.Integer(), nullable=True, comment="本次评测总耗时，单位毫秒"),
            sa.Column("error_message", sa.Text(), nullable=True, comment="运行级异常信息"),
            sa.Column("created_by", sa.String(length=64), nullable=True, comment="发起评测的人员标识"),
            sa.Column("started_at", sa.DateTime(), nullable=False, comment="评测开始时间"),
            sa.Column("finished_at", sa.DateTime(), nullable=True, comment="评测结束时间"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("run_id", name="uk_krer_run_id"),
            comment="RAG 检索评测运行汇总表",
        )
    _create_index_if_missing("knowledge_retrieval_eval_run", "ix_krer_dataset", ["dataset_id"])
    _create_index_if_missing("knowledge_retrieval_eval_run", "ix_krer_doc_ver", ["document_id", "document_version_id"])
    _create_index_if_missing("knowledge_retrieval_eval_run", "ix_krer_status", ["status"])

    if "knowledge_retrieval_eval_case_result" not in existing_tables:
        op.create_table(
            "knowledge_retrieval_eval_case_result",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
            sa.Column("case_result_id", sa.String(length=64), nullable=False, comment="评测样本结果业务唯一 ID"),
            sa.Column("run_id", sa.String(length=64), nullable=False, comment="所属检索评测运行业务 ID"),
            sa.Column("sample_id", sa.String(length=64), nullable=False, comment="关联的评测样本业务 ID"),
            sa.Column("question_snapshot", sa.Text(), nullable=False, comment="运行时的问题快照"),
            sa.Column("sample_type_snapshot", sa.String(length=32), nullable=False, comment="运行时样本类型快照"),
            sa.Column("expected_answerable_snapshot", sa.Integer(), nullable=False, comment="运行时期望是否可回答快照：1 是，0 否"),
            sa.Column("expected_segment_indexes_json", sa.Text(), nullable=False, comment="运行时期望原始段序号 JSON 快照"),
            sa.Column("retrieved_segment_indexes_json", sa.Text(), nullable=False, comment="实际召回原始段序号 JSON"),
            sa.Column("retrieved_chunks_json", sa.Text(), nullable=False, comment="实际召回 chunk 快照 JSON"),
            sa.Column("first_hit_rank", sa.Integer(), nullable=True, comment="首个命中期望原始段的 chunk 排名"),
            sa.Column("is_hit", sa.Integer(), nullable=True, comment="可回答样本是否命中正确依据：1 是，0 否"),
            sa.Column("top_score", sa.Float(), nullable=True, comment="本条样本 Top-1 相似度分数"),
            sa.Column("is_false_positive", sa.Integer(), nullable=True, comment="无答案样本是否被阈值误判为可回答：1 是，0 否"),
            sa.Column("elapsed_ms", sa.Integer(), nullable=True, comment="本条样本检索耗时，单位毫秒"),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="success", comment="明细状态：success 或 error"),
            sa.Column("error_type", sa.String(length=128), nullable=True, comment="检索失败时的异常类型"),
            sa.Column("error_message", sa.Text(), nullable=True, comment="检索失败时的异常信息"),
            sa.Column("created_at", sa.DateTime(), nullable=False, comment="评测明细创建时间"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("case_result_id", name="uk_krecr_case_id"),
            sa.UniqueConstraint("run_id", "sample_id", name="uk_krecr_run_sample"),
            comment="RAG 检索评测样本结果明细表",
        )
    _create_index_if_missing("knowledge_retrieval_eval_case_result", "ix_krecr_run", ["run_id"])
    _create_index_if_missing("knowledge_retrieval_eval_case_result", "ix_krecr_sample", ["sample_id"])
    _create_index_if_missing("knowledge_retrieval_eval_case_result", "ix_krecr_status", ["status"])


def downgrade() -> None:
    op.drop_table("knowledge_retrieval_eval_case_result")
    op.drop_table("knowledge_retrieval_eval_run")
