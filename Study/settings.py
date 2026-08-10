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
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ENV_PATH, override=False)


class Settings(BaseSettings):
    app_name: str = "day04-ai-service"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    dashscope_api_key: str
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-plus"
    # Chat 模型与 Embedding 模型职责不同；文档和查询必须使用同一 Embedding 模型版本。
    dashscope_embedding_model: str = "text-embedding-v3"
    # 精排模型独立于 Chat/Embedding；默认使用 DashScope 的文本重排模型。
    dashscope_rerank_model: str = "gte-rerank-v2"
    dashscope_rerank_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    # v2 增加 version_id 过滤字段；旧的空 Collection 保留，避免开发环境直接破坏性删除。
    milvus_collection_name: str = "knowledge_chunk_vectors_v2"
    # RAG 无答案拒答阈值：Top1 分数低于该值时不调用聊天模型，直接返回依据不足。
    rag_min_relevance_score: float | None = 0.25

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

    # 本地兼容模式默认关闭认证；启用后必须配置至少一个仅含 SHA-256 哈希的 API Key 身份。
    security_enabled: bool = False
    security_api_keys_json: str = "[]"
    # 金汤令 Java 调用 Python 的服务间凭据；它不代表最终登录用户。
    security_service_api_keys_json: str = "[]"

    # 原始知识库文件只允许落在服务端控制的目录，不能使用客户端传入的文件路径。
    knowledge_upload_dir: Path = PROJECT_ROOT / "data" / "knowledge_uploads"
    knowledge_upload_max_bytes: int = 20 * 1024 * 1024

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

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
