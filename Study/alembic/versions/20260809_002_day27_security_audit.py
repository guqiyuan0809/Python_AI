"""Day27 add authorization decision audit table.

Revision ID: 20260809_002
Revises: 20260809_001
Create Date: 2026-08-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260809_002"
down_revision: str | None = "20260809_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ai_security_audit_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
        sa.Column("audit_id", sa.String(length=64), nullable=False, comment="授权审计业务唯一 ID"),
        sa.Column("trace_id", sa.String(length=64), nullable=True, comment="关联本次 HTTP 请求链路 ID"),
        sa.Column("actor_id", sa.String(length=64), nullable=True, comment="调用者身份 ID；认证失败时为空"),
        sa.Column("api_key_id", sa.String(length=64), nullable=True, comment="API Key 管理标识，不保存原始 Key 或 Key 哈希"),
        sa.Column("roles_json", sa.Text(), nullable=False, comment="调用者角色名称 JSON 快照"),
        sa.Column("permission", sa.String(length=255), nullable=False, comment="本次请求要求的权限，多个权限以逗号分隔"),
        sa.Column("http_method", sa.String(length=16), nullable=False, comment="HTTP 请求方法"),
        sa.Column("request_path", sa.String(length=255), nullable=False, comment="请求路径，不包含查询参数和请求正文"),
        sa.Column("decision", sa.String(length=16), nullable=False, comment="授权结论：allow 或 deny"),
        sa.Column("reason", sa.String(length=64), nullable=False, comment="授权结论原因代码，不保存敏感业务内容"),
        sa.Column("resource_type", sa.String(length=64), nullable=True, comment="被访问资源类型，例如 prompt、task、trace"),
        sa.Column("resource_id", sa.String(length=128), nullable=True, comment="来自受信任路径参数的资源业务 ID"),
        sa.Column("created_at", sa.DateTime(), nullable=False, comment="授权决策发生时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("audit_id", name="uk_asal_audit_id"),
    )
    op.create_index("ix_asal_trace", "ai_security_audit_log", ["trace_id"])
    op.create_index("ix_asal_actor", "ai_security_audit_log", ["actor_id"])
    op.create_index("ix_asal_permission", "ai_security_audit_log", ["permission"])
    op.create_index("ix_asal_decision", "ai_security_audit_log", ["decision"])
    op.create_index("ix_asal_created", "ai_security_audit_log", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_asal_created", table_name="ai_security_audit_log")
    op.drop_index("ix_asal_decision", table_name="ai_security_audit_log")
    op.drop_index("ix_asal_permission", table_name="ai_security_audit_log")
    op.drop_index("ix_asal_actor", table_name="ai_security_audit_log")
    op.drop_index("ix_asal_trace", table_name="ai_security_audit_log")
    op.drop_table("ai_security_audit_log")
