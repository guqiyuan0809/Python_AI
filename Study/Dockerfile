FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先复制依赖清单，便于 Docker 在业务代码变更时复用依赖层缓存。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini ./
COPY alembic ./alembic
COPY settings.py run_day04.py ./
COPY day04_app ./day04_app
COPY scripts ./scripts

RUN mkdir -p /app/data/knowledge_uploads

EXPOSE 8000

# 健康检查使用 live 探针；依赖就绪由 /health/ready 另行检查。
CMD ["uvicorn", "day04_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
