"""Day22 add RAG retrieval evaluation datasets.

Revision ID: 20260802_001
Revises: 20260731_002
Create Date: 2026-08-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260802_001"
down_revision: str | None = "20260731_002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_index_if_missing(
    table_name: str,
    index_name: str,
    columns: list[str],
) -> None:
    existing_indexes = {
        index["name"]
        for index in inspect(op.get_bind()).get_indexes(table_name)
    }
    if index_name not in existing_indexes:
        op.create_index(index_name, table_name, columns)


def upgrade() -> None:
    existing_tables = set(inspect(op.get_bind()).get_table_names())
    if "knowledge_retrieval_eval_dataset" not in existing_tables:
        op.create_table(
            "knowledge_retrieval_eval_dataset",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
            sa.Column("dataset_id", sa.String(length=64), nullable=False, comment="检索评测数据集业务唯一 ID"),
            sa.Column("dataset_name", sa.String(length=64), nullable=False, comment="数据集名称，例如 jvm_knowledge_retrieval"),
            sa.Column("dataset_version", sa.String(length=64), nullable=False, comment="数据集版本，例如 v1"),
            sa.Column("document_id", sa.String(length=64), nullable=False, comment="评测范围内的知识库文档业务 ID"),
            sa.Column("description", sa.String(length=500), nullable=True, comment="数据集用途和覆盖范围说明"),
            sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0", comment="当前可参与评测的样本数"),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="draft", comment="数据集状态：draft、active、archived"),
            sa.Column("created_by", sa.String(length=64), nullable=True, comment="创建人标识"),
            sa.Column("created_at", sa.DateTime(), nullable=False, comment="数据集创建时间"),
            sa.Column("updated_at", sa.DateTime(), nullable=False, comment="数据集最后修改时间"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("dataset_id", name="uk_kred_dataset_id"),
            comment="RAG 检索评测数据集表",
        )
    _create_index_if_missing("knowledge_retrieval_eval_dataset", "ix_kred_doc", ["document_id"])
    _create_index_if_missing("knowledge_retrieval_eval_dataset", "ix_kred_name_ver", ["dataset_name", "dataset_version"])
    _create_index_if_missing("knowledge_retrieval_eval_dataset", "ix_kred_status", ["status"])

    if "knowledge_retrieval_eval_sample" not in existing_tables:
        op.create_table(
            "knowledge_retrieval_eval_sample",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
            sa.Column("sample_id", sa.String(length=64), nullable=False, comment="检索评测样本业务唯一 ID"),
            sa.Column("dataset_id", sa.String(length=64), nullable=False, comment="所属检索评测数据集业务 ID"),
            sa.Column("question", sa.Text(), nullable=False, comment="待检索的用户问题"),
            sa.Column("sample_type", sa.String(length=32), nullable=False, server_default="normal", comment="样本类型：normal、boundary、no_answer"),
            sa.Column("expected_answerable", sa.Integer(), nullable=False, server_default="1", comment="是否期望知识库可回答：1 是，0 否"),
            sa.Column("expected_segment_indexes_json", sa.Text(), nullable=False, comment="期望原文段序号 JSON 数组"),
            sa.Column("expected_note", sa.Text(), nullable=True, comment="人工标注理由或期望命中依据说明"),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="active", comment="样本状态：active 或 archived"),
            sa.Column("created_by", sa.String(length=64), nullable=True, comment="标注人标识"),
            sa.Column("created_at", sa.DateTime(), nullable=False, comment="样本创建时间"),
            sa.Column("updated_at", sa.DateTime(), nullable=False, comment="样本最后修改时间"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("sample_id", name="uk_kres_sample_id"),
            comment="RAG 检索评测样本表",
        )
    _create_index_if_missing("knowledge_retrieval_eval_sample", "ix_kres_dataset", ["dataset_id"])
    _create_index_if_missing("knowledge_retrieval_eval_sample", "ix_kres_type", ["sample_type"])
    _create_index_if_missing("knowledge_retrieval_eval_sample", "ix_kres_status", ["status"])


def downgrade() -> None:
    op.drop_table("knowledge_retrieval_eval_sample")
    op.drop_table("knowledge_retrieval_eval_dataset")
