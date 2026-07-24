"""
Day16 Prompt 与评测样本主数据表临时迁移脚本。

运行命令：
D:\\Pythoncode\\.venv\\Scripts\\python.exe migrate_day16_prompt_eval_master_tables.py
"""

from sqlalchemy import text

from day04_app.database import engine


DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS ai_prompt_version (
        id INT AUTO_INCREMENT PRIMARY KEY,
        prompt_id VARCHAR(64) NOT NULL UNIQUE,
        prompt_name VARCHAR(64) NOT NULL,
        prompt_version VARCHAR(32) NOT NULL,
        description VARCHAR(255) NULL,
        system_prompt TEXT NOT NULL,
        user_prompt_template TEXT NOT NULL,
        model VARCHAR(64) NULL,
        temperature DOUBLE NULL,
        max_tokens INT NULL,
        status VARCHAR(32) NOT NULL DEFAULT 'draft',
        created_by VARCHAR(64) NULL,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL,
        INDEX ix_ai_prompt_version_prompt_id (prompt_id),
        INDEX ix_ai_prompt_version_prompt_name (prompt_name),
        INDEX ix_ai_prompt_version_prompt_version (prompt_version),
        INDEX ix_ai_prompt_version_status (status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='AI Prompt 版本表'
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_eval_dataset (
        id INT AUTO_INCREMENT PRIMARY KEY,
        dataset_id VARCHAR(64) NOT NULL UNIQUE,
        dataset_name VARCHAR(64) NOT NULL,
        dataset_version VARCHAR(64) NOT NULL,
        description VARCHAR(255) NULL,
        sample_count INT NOT NULL DEFAULT 0,
        status VARCHAR(32) NOT NULL DEFAULT 'active',
        created_by VARCHAR(64) NULL,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL,
        INDEX ix_ai_eval_dataset_dataset_id (dataset_id),
        INDEX ix_ai_eval_dataset_dataset_name (dataset_name),
        INDEX ix_ai_eval_dataset_dataset_version (dataset_version),
        INDEX ix_ai_eval_dataset_status (status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='AI 评测数据集表'
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_eval_sample (
        id INT AUTO_INCREMENT PRIMARY KEY,
        sample_id VARCHAR(64) NOT NULL UNIQUE,
        dataset_id VARCHAR(64) NOT NULL,
        dataset_version VARCHAR(64) NOT NULL,
        sample_type VARCHAR(32) NOT NULL DEFAULT 'normal',
        input_text TEXT NOT NULL,
        expected_json TEXT NOT NULL,
        source_type VARCHAR(32) NOT NULL DEFAULT 'manual',
        source_ref_id VARCHAR(64) NULL,
        status VARCHAR(32) NOT NULL DEFAULT 'active',
        created_by VARCHAR(64) NULL,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL,
        INDEX ix_ai_eval_sample_sample_id (sample_id),
        INDEX ix_ai_eval_sample_dataset_id (dataset_id),
        INDEX ix_ai_eval_sample_dataset_version (dataset_version),
        INDEX ix_ai_eval_sample_sample_type (sample_type),
        INDEX ix_ai_eval_sample_source_type (source_type),
        INDEX ix_ai_eval_sample_source_ref_id (source_ref_id),
        INDEX ix_ai_eval_sample_status (status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='AI 评测样本表'
    """,
]


def main() -> None:
    with engine.begin() as conn:
        for ddl in DDL_STATEMENTS:
            # CREATE TABLE IF NOT EXISTS 保证本地多次运行不会因为表已存在而失败。
            conn.execute(text(ddl))
    print("Day16 Prompt 与评测样本主数据表迁移完成")


if __name__ == "__main__":
    main()
