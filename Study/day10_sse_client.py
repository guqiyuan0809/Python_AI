"""
Day10 SSE 客户端测试脚本

作用：
1. 模拟前端 / Java 后端调用会话流式聊天接口
2. 逐行读取 SSE 返回的 start / delta / done / error 事件
3. 帮助理解服务端 StreamingResponse 是怎样被客户端持续消费的

运行前请先启动 FastAPI 服务：
python run_day04.py

示例：
python day10_sse_client.py --message "帮我解释一下 FastAPI SSE"
python day10_sse_client.py --session-id 你的会话ID --message "继续刚才的话题"
"""

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"


def post_json(path: str, body: dict, timeout: float = 60.0):
    request = Request(
        url=f"{BASE_URL}{path}",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urlopen(request, timeout=timeout)


def create_session() -> str:
    # 没有传 session_id 时，先创建一个新会话，方便直接测试流式接口。
    with post_json("/api/chat/sessions", {}) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["data"]["session_id"]


def print_sse_stream(session_id: str, message: str, history_limit: int) -> None:
    body = {
        "session_id": session_id,
        "message": message,
        "history_limit": history_limit,
    }

    print(f"session_id: {session_id}")
    print("开始读取 SSE 事件...\n")

    with post_json("/api/chat/sessions/stream", body, timeout=120.0) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue

            # SSE 数据行以 data: 开头，后面才是真正的 JSON 字符串。
            if not line.startswith("data:"):
                print(line)
                continue

            event_data = line.removeprefix("data:").strip()
            event = json.loads(event_data)
            event_type = event.get("type")

            if event_type == "delta":
                print(event.get("delta", ""), end="", flush=True)
            else:
                print(f"\n[{event_type}] {event}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day10 SSE 流式接口测试客户端")
    parser.add_argument("--session-id", default="", help="已有会话 ID，不传则自动创建")
    parser.add_argument("--message", required=True, help="用户输入的问题")
    parser.add_argument("--history-limit", type=int, default=6, help="携带最近多少条历史")
    args = parser.parse_args()

    try:
        session_id = args.session_id or create_session()
        print_sse_stream(session_id, args.message, args.history_limit)
    except HTTPError as exc:
        print(f"HTTP 调用失败：{exc.code} {exc.reason}", file=sys.stderr)
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
    except URLError as exc:
        print(f"连接失败：{exc.reason}", file=sys.stderr)
        print("请确认 FastAPI 服务已经启动：python run_day04.py", file=sys.stderr)


if __name__ == "__main__":
    main()
