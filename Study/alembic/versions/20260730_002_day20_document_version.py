"""Day20 add knowledge document versions for rebuild and switch.

Revision ID: 20260730_002
Revises: 20260730_001
Create Date: 2026-07-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260730_002"
down_revision: str | None = "20260730_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _add_active_version_column() -> None:
    existing_columns = {
        column["name"]
        for column in inspect(op.get_bind()).get_columns("knowledge_document")
    }
    if "active_version_id" not in existing_columns:
        op.add_column(
            "knowledge_document",
            sa.Column(
                "active_version_id",
                sa.String(length=64),
                nullable=True,
                comment="当前生效的文档版本业务 ID，切换完成后才更新",
            ),
        )


def _create_version_table() -> None:
    op.create_table(
        "knowledge_document_version",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
        sa.Column("version_id", sa.String(length=64), nullable=False, comment="文档版本业务唯一 ID"),
        sa.Column("document_id", sa.String(length=64), nullable=False, comment="所属知识库文档业务 ID"),
        sa.Column("version_number", sa.Integer(), nullable=False, comment="同一文档内递增版本号，从 1 开始"),
        sa.Column("status", sa.String(length=32), nullable=False, comment="版本索引生命周期状态，active 表示当前可被检索"),
        sa.Column("source_sha256", sa.String(length=64), nullable=False, comment="该版本原始文件内容 SHA-256"),
        sa.Column("parser_name", sa.String(length=64), nullable=True, comment="该版本解析时使用的解析器名称"),
        sa.Column("segment_count", sa.Integer(), nullable=False, comment="该版本解析出的原始文本段数量"),
        sa.Column("chunk_config_json", sa.Text(), nullable=True, comment="该版本切块参数快照 JSON"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, comment="该版本生成的检索块数量"),
        sa.Column("embedding_model", sa.String(length=128), nullable=True, comment="该版本向量化使用的 Embedding 模型"),
        sa.Column("embedding_dimension", sa.Integer(), nullable=True, comment="该版本 Embedding 向量维度"),
        sa.Column("vector_collection", sa.String(length=128), nullable=True, comment="该版本向量写入的 Milvus Collection 名称"),
        sa.Column("vector_count", sa.Integer(), nullable=False, comment="该版本已成功写入向量库的 Entity 数量"),
        sa.Column("error_message", sa.Text(), nullable=True, comment="版本构建失败时记录的错误原因"),
        sa.Column("created_at", sa.DateTime(), nullable=False, comment="候选版本创建时间"),
        sa.Column("updated_at", sa.DateTime(), nullable=False, comment="版本构建状态最后更新时间"),
        sa.Column("activated_at", sa.DateTime(), nullable=True, comment="版本切换为 active 的时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version_id"),
        sa.UniqueConstraint("document_id", "version_number", name="uk_knowledge_document_version_document_number"),
        comment="知识库文档索引版本表，支持候选构建和无感切换",
    )


def _create_missing_indexes() -> None:
    document_indexes = {
        index["name"]
        for index in inspect(op.get_bind()).get_indexes("knowledge_document")
    }
    if "ix_knowledge_document_active_version_id" not in document_indexes:
        op.create_index(
            "ix_knowledge_document_active_version_id",
            "knowledge_document",
            ["active_version_id"],
        )

    version_indexes = {
        index["name"]
        for index in inspect(op.get_bind()).get_indexes("knowledge_document_version")
    }
    expected_indexes = {
        "ix_knowledge_document_version_version_id": (["version_id"], True),
        "ix_knowledge_document_version_document_id": (["document_id"], False),
        "ix_knowledge_document_version_status": (["status"], False),
        "ix_knowledge_document_version_source_sha256": (["source_sha256"], False),
    }
    for index_name, (columns, unique) in expected_indexes.items():
        if index_name not in version_indexes:
            op.create_index(index_name, "knowledge_document_version", columns, unique=unique)


def _backfill_legacy_versions() -> None:
    """把 Day18/19 旧文档映射为 v1 候选版本，但不冒充已经通过 Milvus 验证的 active 版本。"""
    op.execute(
        """
        INSERT INTO knowledge_document_version (
            version_id, document_id, version_number, status, source_sha256,
            parser_name, segment_count, chunk_config_json, chunk_count,
            embedding_model, embedding_dimension, vector_collection, vector_count,
            error_message, created_at, updated_at, activated_at
        )
        SELECT
            SHA2(CONCAT(d.document_id, ':legacy:v1'), 256),
            d.document_id,
            1,
            CASE
                WHEN d.status = 'error' OR d.chunk_status = 'error' THEN 'error'
                WHEN d.chunk_status = 'chunked' THEN 'chunked'
                WHEN d.status = 'parsed' THEN 'parsed'
                ELSE 'uploaded'
            END,
            d.content_sha256,
            d.parser_name,
            d.parsed_segment_count,
            d.chunk_config_json,
            d.chunk_count,
            NULL, NULL, NULL, 0,
            COALESCE(d.chunk_error_message, d.error_message),
            d.created_at,
            d.updated_at,
            NULL
        FROM knowledge_document d
        WHERE NOT EXISTS (
            SELECT 1
            FROM knowledge_document_version v
            WHERE v.document_id = d.document_id AND v.version_number = 1
        )
        """
    )


def upgrade() -> None:
    _add_active_version_column()
    table_names = set(inspect(op.get_bind()).get_table_names())
    if "knowledge_document_version" not in table_names:
        _create_version_table()
    _create_missing_indexes()
    _backfill_legacy_versions()


def downgrade() -> None:
    op.drop_table("knowledge_document_version")
    op.drop_column("knowledge_document", "active_version_id")
