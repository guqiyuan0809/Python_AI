"""Day18 add knowledge document metadata table.

Revision ID: 20260728_001
Revises: 20260727_003
Create Date: 2026-07-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260728_001"
down_revision: str | None = "20260727_003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_table() -> None:
    op.create_table(
        "knowledge_document",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
        sa.Column("document_id", sa.String(length=64), nullable=False, comment="知识库文档业务唯一 ID"),
        sa.Column("original_file_name", sa.String(length=255), nullable=False, comment="用户上传时的原始文件名，仅用于展示和审计"),
        sa.Column("file_type", sa.String(length=16), nullable=False, comment="已校验的文件类型，例如 docx/pdf/xlsx"),
        sa.Column("storage_key", sa.String(length=255), nullable=False, comment="服务端原始文件存储键，不向调用方暴露物理路径"),
        sa.Column("file_size", sa.Integer(), nullable=False, comment="原始文件大小，单位字节"),
        sa.Column("content_sha256", sa.String(length=64), nullable=False, comment="原始文件内容 SHA-256，用于完整性校验和重复文件识别"),
        sa.Column("trace_id", sa.String(length=64), nullable=True, comment="上传请求链路追踪 ID"),
        sa.Column("status", sa.String(length=32), nullable=False, comment="文档生命周期状态：uploaded/parsing/parsed/error"),
        sa.Column("parser_name", sa.String(length=64), nullable=True, comment="最近一次成功执行的解析器名称"),
        sa.Column("parsed_segment_count", sa.Integer(), nullable=False, comment="最近一次成功解析得到的有效文本段数量"),
        sa.Column("error_message", sa.Text(), nullable=True, comment="解析失败时记录的错误原因"),
        sa.Column("created_at", sa.DateTime(), nullable=False, comment="文档上传记录创建时间"),
        sa.Column("updated_at", sa.DateTime(), nullable=False, comment="文档记录最后更新时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id"),
        sa.UniqueConstraint("storage_key"),
        comment="知识库文档元数据与解析生命周期主表",
    )


def _create_missing_indexes() -> None:
    expected_indexes = {
        "ix_knowledge_document_document_id": (["document_id"], True),
        "ix_knowledge_document_file_type": (["file_type"], False),
        "ix_knowledge_document_storage_key": (["storage_key"], True),
        "ix_knowledge_document_content_sha256": (["content_sha256"], False),
        "ix_knowledge_document_trace_id": (["trace_id"], False),
        "ix_knowledge_document_status": (["status"], False),
    }
    existing_indexes = {index["name"] for index in inspect(op.get_bind()).get_indexes("knowledge_document")}
    for index_name, (columns, unique) in expected_indexes.items():
        if index_name not in existing_indexes:
            op.create_index(index_name, "knowledge_document", columns, unique=unique)


def upgrade() -> None:
    # 兼容学习环境中 FastAPI startup 已通过 create_all 建表的情形。
    table_names = set(inspect(op.get_bind()).get_table_names())
    if "knowledge_document" not in table_names:
        _create_table()
    _create_missing_indexes()


def downgrade() -> None:
    op.drop_table("knowledge_document")
