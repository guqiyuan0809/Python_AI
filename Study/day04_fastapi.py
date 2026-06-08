"""
Day 04: 第一个 FastAPI AI 接口

学习目标：
1.
2. 理解 FastAPI 为什么适合做 AI 接口服务学会把之前的 safe_chat 封装成 HTTP 接口
3. 看到 Python AI 模块如何被 Java 后端调用

运行前准备：
1. 确保 Study/.env 已存在，并配置了 DASHSCOPE_API_KEY
2. 安装依赖：
   pip install fastapi uvicorn openai python-dotenv pydantic-settings

启动命令：
uvicorn day04_fastapi:app --reload --host 127.0.0.1 --port 8000

启动后访问：
- Swagger 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health
"""

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field

from settings import settings


app = FastAPI(
    title="Day04 AI Service",
    description="第一个基于 FastAPI 的 AI 接口服务",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    """
    这是请求体模型。
    FastAPI 会自动根据它做参数校验和文档生成。
    """
    message: str = Field(..., min_length=1, description="用户输入的问题")


class ChatResponse(BaseModel):
    """
    这是响应体模型。
    后面 Java 或前端调用时，看到的就是这类 JSON 结构。
    """
    success: bool
    answer: str | None = None
    error_type: str | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def validate_config() -> None:
    if not settings.dashscope_api_key:
        raise ValueError(
            "没有读取到通义千问 API Key。请先在 Study/.env 中配置 DASHSCOPE_API_KEY"
        )


def create_client(timeout: float = 30.0) -> OpenAI:
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        timeout=timeout,
    )


def safe_chat(message: str) -> dict:
    """
    这里延续 Day03 的工程化思路：
    统一配置、统一调用、统一异常处理、统一结构化返回。
    """
    client = create_client(timeout=30.0)
    try:
        response = client.chat.completions.create(
            model=settings.dashscope_model,
            messages=[
                {"role": "system", "content": "你是一个简洁专业的 Python AI 助手。"},
                {"role": "user", "content": message},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return {
            "success": True,
            "answer": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    except Exception as exc:
        return {
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


@app.on_event("startup")
def startup_check() -> None:
    validate_config()


@app.get("/health")
def health() -> dict:
    """
    健康检查接口。
    真实项目里通常会给 Nginx、Docker、K8s 或监控系统调用。
    """
    return {
        "success": True,
        "message": "service is running",
        "model": settings.dashscope_model,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    最小聊天接口。
    Java 后端、前端、Postman 都可以通过 HTTP 调这个接口。
    """
    result = safe_chat(request.message)
    return ChatResponse(**result)

@app.get("/pathvarible/{useid}")
def pathvarible(useid: str = 1):
    return {"message": "路由参数", "useid": useid}
