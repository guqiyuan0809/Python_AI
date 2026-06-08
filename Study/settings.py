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

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
