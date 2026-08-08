"""Day25 add Agent Loop Harness tables.

Revision ID: 20260807_001
Revises: 20260805_001
Create Date: 2026-08-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "20260807_001"
down_revision: str | None = "20260805_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table_names() -> set[str]:
    return set(inspect(op.get_bind()).get_table_names())


def _create_agent_eval_run() -> None:
    op.create_table(
        "ai_agent_eval_run",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
        sa.Column("run_id", sa.String(length=64), nullable=False, comment="Agent Harness 单次运行业务唯一 ID"),
        sa.Column("agent_name", sa.String(length=64), nullable=False, comment="被评测 Agent 名称"),
        sa.Column("agent_version", sa.String(length=64), nullable=False, comment="被评测 Agent 的实现或提示词版本标签"),
        sa.Column("dataset_version", sa.String(length=64), nullable=False, comment="本次评测使用的数据集版本"),
        sa.Column("agent_snapshot_hash", sa.String(length=64), nullable=False, comment="Agent 提示词和工具白名单快照 SHA-256"),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0", comment="本次实际执行的评测样本数量"),
        sa.Column("status_match_rate", sa.Float(), nullable=False, server_default="0", comment="最终状态命中率"),
        sa.Column("step_sequence_match_rate", sa.Float(), nullable=False, server_default="0", comment="动作和工具调用顺序完整命中率"),
        sa.Column("tool_call_accuracy", sa.Float(), nullable=False, server_default="0", comment="期望工具调用中工具名和参数均命中的比例"),
        sa.Column("observation_status_accuracy", sa.Float(), nullable=False, server_default="0", comment="工具 observation 状态命中率"),
        sa.Column("safety_case_pass_rate", sa.Float(), nullable=False, server_default="0", comment="安全样本完整通过率"),
        sa.Column("full_pass_rate", sa.Float(), nullable=False, server_default="0", comment="所有必填断言同时通过的样本比例"),
        sa.Column("avg_step_count", sa.Float(), nullable=True, comment="每条样本平均 Agent 循环步数"),
        sa.Column("avg_total_tokens", sa.Float(), nullable=True, comment="每条样本平均总 Token 数"),
        sa.Column("avg_cost_ms", sa.Float(), nullable=True, comment="每条样本平均总耗时，单位毫秒"),
        sa.Column("metrics_json", sa.Text(), nullable=False, comment="完整评测指标、Agent 快照和失败样本 JSON"),
        sa.Column("created_at", sa.DateTime(), nullable=False, comment="评测运行创建时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", name="uk_aaer_run_id"),
    )
    op.create_index("ix_aaer_agent_ver", "ai_agent_eval_run", ["agent_name", "agent_version"])
    op.create_index("ix_aaer_dataset", "ai_agent_eval_run", ["dataset_version"])


def _create_agent_eval_case_result() -> None:
    op.create_table(
        "ai_agent_eval_case_result",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
        sa.Column("run_id", sa.String(length=64), nullable=False, comment="所属 Agent Harness 运行业务 ID"),
        sa.Column("sample_id", sa.String(length=64), nullable=False, comment="关联的通用 AI 评测样本业务 ID"),
        sa.Column("sample_type", sa.String(length=32), nullable=False, comment="样本类型，例如 normal、boundary、safety"),
        sa.Column("status_match", sa.Integer(), nullable=False, server_default="0", comment="最终运行状态是否符合人工期望，使用 0 或 1 存储"),
        sa.Column("step_sequence_match", sa.Integer(), nullable=False, server_default="0", comment="动作和工具调用顺序是否完整符合期望，使用 0 或 1 存储"),
        sa.Column("tool_call_match", sa.Integer(), nullable=False, server_default="0", comment="本样本所有期望工具调用是否命中，使用 0 或 1 存储"),
        sa.Column("observation_status_match", sa.Integer(), nullable=False, server_default="0", comment="本样本期望 observation 状态是否全部命中，使用 0 或 1 存储"),
        sa.Column("answer_match", sa.Integer(), nullable=False, server_default="1", comment="回答关键字是否命中，使用 0 或 1 存储"),
        sa.Column("case_pass", sa.Integer(), nullable=False, server_default="0", comment="所有必填断言是否同时通过，使用 0 或 1 存储"),
        sa.Column("actual_step_count", sa.Integer(), nullable=False, server_default="0", comment="实际 Agent 循环步骤数量"),
        sa.Column("total_tokens", sa.Integer(), nullable=True, comment="本样本 Agent 消耗的总 Token 数"),
        sa.Column("cost_ms", sa.Integer(), nullable=True, comment="本样本 Agent 总耗时，单位毫秒"),
        sa.Column("error_type", sa.String(length=64), nullable=True, comment="执行异常类型"),
        sa.Column("error_message", sa.Text(), nullable=True, comment="执行异常简要信息"),
        sa.Column("expected_json", sa.Text(), nullable=False, comment="人工标注的预期最终状态、步骤和安全断言 JSON"),
        sa.Column("actual_json", sa.Text(), nullable=True, comment="实际 Agent 响应快照 JSON"),
        sa.Column("row_json", sa.Text(), nullable=False, comment="包含命中明细和运行快照的完整行 JSON"),
        sa.Column("created_at", sa.DateTime(), nullable=False, comment="样本评测结果创建时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "sample_id", name="uk_aaecr_run_sample"),
    )
    op.create_index("ix_aaecr_run", "ai_agent_eval_case_result", ["run_id"])
    op.create_index("ix_aaecr_sample", "ai_agent_eval_case_result", ["sample_id"])
    op.create_index("ix_aaecr_pass", "ai_agent_eval_case_result", ["case_pass"])


def _create_agent_eval_gate_decision() -> None:
    op.create_table(
        "ai_agent_eval_gate_decision",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="数据库自增主键"),
        sa.Column("gate_id", sa.String(length=64), nullable=False, comment="Agent 评测门禁业务唯一 ID"),
        sa.Column("baseline_run_id", sa.String(length=64), nullable=False, comment="基线 Agent Harness 运行业务 ID"),
        sa.Column("candidate_run_id", sa.String(length=64), nullable=False, comment="候选 Agent Harness 运行业务 ID"),
        sa.Column("agent_name", sa.String(length=64), nullable=False, comment="被比较的 Agent 名称"),
        sa.Column("dataset_version", sa.String(length=64), nullable=False, comment="两次运行共用的数据集版本"),
        sa.Column("decision", sa.String(length=32), nullable=False, comment="准入结论：pass、reject 或 manual_review"),
        sa.Column("comparison_json", sa.Text(), nullable=False, comment="基线和候选指标对比 JSON"),
        sa.Column("reason_json", sa.Text(), nullable=False, comment="门禁结论原因列表 JSON"),
        sa.Column("rule_snapshot_json", sa.Text(), nullable=False, comment="生成本次门禁结论时使用的规则快照 JSON"),
        sa.Column("created_at", sa.DateTime(), nullable=False, comment="Agent 门禁结论创建时间"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("gate_id", name="uk_aaegd_gate_id"),
    )
    op.create_index("ix_aaegd_baseline", "ai_agent_eval_gate_decision", ["baseline_run_id"])
    op.create_index("ix_aaegd_candidate", "ai_agent_eval_gate_decision", ["candidate_run_id"])
    op.create_index("ix_aaegd_agent", "ai_agent_eval_gate_decision", ["agent_name"])
    op.create_index("ix_aaegd_decision", "ai_agent_eval_gate_decision", ["decision"])


def upgrade() -> None:
    table_names = _table_names()
    if "ai_agent_eval_run" not in table_names:
        _create_agent_eval_run()
    if "ai_agent_eval_case_result" not in table_names:
        _create_agent_eval_case_result()
    if "ai_agent_eval_gate_decision" not in table_names:
        _create_agent_eval_gate_decision()


def downgrade() -> None:
    table_names = _table_names()
    if "ai_agent_eval_gate_decision" in table_names:
        op.drop_table("ai_agent_eval_gate_decision")
    if "ai_agent_eval_case_result" in table_names:
        op.drop_table("ai_agent_eval_case_result")
    if "ai_agent_eval_run" in table_names:
        op.drop_table("ai_agent_eval_run")
