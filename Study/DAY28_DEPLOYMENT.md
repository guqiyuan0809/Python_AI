# Day28：本地生产部署演练

本阶段先在本机模拟生产环境，不要求购买服务器或域名。

## 1. 准备配置

复制 `.env.example` 为 `.env`，填写本机的 DashScope、MySQL 和安全配置。`.env` 不得提交 Git。

生产容器使用 `APP_ENV=production` 和 `AUTO_CREATE_TABLES=false`。数据库结构由 Alembic 管理，不由 Web 进程启动时隐式创建。

注意：容器里的 `127.0.0.1` 只代表该容器自己，不是 Windows 宿主机。因此 Docker Desktop 默认把 `DB_HOST` 覆盖为 `host.docker.internal`；若部署到 RDS 或 MySQL 容器，请设置非敏感地址变量 `DOCKER_DB_HOST` 为对应私网地址/Compose 服务名。账号和密码仍只放 `.env`。

## 2. 启动基础设施

```powershell
docker compose -f compose.yaml up -d rabbitmq milvus-etcd milvus-minio milvus
```

## 3. 执行数据库迁移

在 Python 虚拟环境或一次性应用容器中执行：

```powershell
& D:\Pythoncode\.venv\Scripts\python.exe -m alembic upgrade head
```

## 4. 构建并启动应用

```powershell
docker compose -f compose.yaml -f compose.deploy.yaml build ai-api
docker compose -f compose.yaml -f compose.deploy.yaml up -d ai-api ai-worker ai-beat
```

`ai-worker` 和 `ai-beat` 复用同一个镜像，分别承担异步消费和定时投递；Beat 计划文件保存到独立卷，避免容器重建导致调度状态丢失。

## 5. 检查服务

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
docker compose -f compose.yaml -f compose.deploy.yaml logs -f ai-api
```

停止应用：

```powershell
docker compose -f compose.yaml -f compose.deploy.yaml stop ai-api ai-worker ai-beat
```

本阶段只完成容器编排和启动边界；Nginx、HTTPS、CI/CD 和公网发布在后续阶段学习。

> Day29 起，浏览器统一访问 Nginx；前端静态资源由 Nginx 返回，`/python-ai/` 请求转发给 Java。Python `8000` 仅绑定宿主机回环地址，由 Java 使用，不再作为前端入口。
