"""
Day15 评测结果表临时迁移脚本。

后续切到 Alembic 后，这类脚本会被正式 migration 文件替代。
"""

from sqlalchemy import text

from day04_app.database import engine


DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS ai_eval_run (
        id INT AUTO_INCREMENT PRIMARY KEY,
        run_id VARCHAR(64) NOT NULL UNIQUE,
        prompt_name VARCHAR(64) NOT NULL,
        prompt_version VARCHAR(32) NOT NULL,
        dataset_version VARCHAR(64) NOT NULL,
        sample_count INT NOT NULL DEFAULT 0,
        schema_valid_rate DOUBLE NOT NULL DEFAULT 0,
        category_accuracy DOUBLE NOT NULL DEFAULT 0,
        risk_level_accuracy DOUBLE NOT NULL DEFAULT 0,
        human_review_accuracy DOUBLE NOT NULL DEFAULT 0,
        avg_total_tokens DOUBLE NULL,
        avg_cost_ms DOUBLE NULL,
        metrics_json TEXT NOT NULL,
        created_at DATETIME NOT NULL,
        INDEX ix_ai_eval_run_run_id (run_id),
        INDEX ix_ai_eval_run_prompt_name (prompt_name),
        INDEX ix_ai_eval_run_prompt_version (prompt_version),
        INDEX ix_ai_eval_run_dataset_version (dataset_version)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_eval_case_result (
        id INT AUTO_INCREMENT PRIMARY KEY,
        run_id VARCHAR(64) NOT NULL,
        sample_id VARCHAR(64) NOT NULL,
        schema_valid INT NOT NULL DEFAULT 0,
        category_match INT NOT NULL DEFAULT 0,
        risk_level_match INT NOT NULL DEFAULT 0,
        human_review_match INT NOT NULL DEFAULT 0,
        total_tokens INT NULL,
        cost_ms INT NULL,
        error_type VARCHAR(64) NULL,
        error_message TEXT NULL,
        expected_json TEXT NULL,
        actual_json TEXT NULL,
        row_json TEXT NOT NULL,
        created_at DATETIME NOT NULL,
        INDEX ix_ai_eval_case_result_run_id (run_id),
        INDEX ix_ai_eval_case_result_sample_id (sample_id),
        INDEX ix_ai_eval_case_result_error_type (error_type)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


def main() -> None:
    with engine.begin() as conn:
        for ddl in DDL_STATEMENTS:
            # 每条 DDL 独立执行，CREATE TABLE IF NOT EXISTS 保证重复执行不会报错。
            conn.execute(text(ddl))
    print("Day15 评测结果表迁移完成")


if __name__ == "__main__":
    main()
