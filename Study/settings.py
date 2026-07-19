"""
统一配置文件

作用：
1. 统一从 .env 读取配置
2. 让业务代码不再到处写 os.getenv(...)
3. 更接近企业项目里的 settings / config 写法
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=False)


class Settings(BaseSettings):
    app_name: str = "day04-ai-service"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    dashscope_api_key: str
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-plus"

    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str
    db_name: str = "PythonAi"

    # RabbitMQ 承担真正的消息 Broker；任务结果统一落 MySQL，不再依赖 Celery result backend。
    rabbitmq_url: str = "amqp://python_ai:python_ai_123@127.0.0.1:5672//"
    celery_broker_url: str | None = None
    async_task_timeout_minutes: int = 10
    async_task_max_retries: int = 3

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )

    @property
    def celery_broker(self) -> str:
        # 没有单独配置时，开发环境默认使用 RabbitMQ，贴近 Java 企业 MQ 场景。
        return self.celery_broker_url or self.rabbitmq_url

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
