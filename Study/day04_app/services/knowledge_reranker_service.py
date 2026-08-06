"""RAG 精排服务：使用独立 rerank 模型重排向量粗排候选，不混淆聊天模型职责。"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from day04_app.common.exceptions import ModelCallException
from settings import settings


@dataclass(frozen=True)
class RerankResult:
    """一个粗排候选的精排结果；index 对应传入 documents 的下标。"""

    index: int
    relevance_score: float


def rerank_chunks(question: str, documents: list[str], top_k: int) -> list[RerankResult]:
    """调用 DashScope 原生 Rerank API，只返回排序下标与分数，不改写原文。"""
    if not documents:
        return []
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    try:
        response = httpx.post(
            settings.dashscope_rerank_url,
            headers={
                "Authorization": f"Bearer {settings.dashscope_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.dashscope_rerank_model,
                "input": {"query": question, "documents": documents},
                "parameters": {
                    "top_n": min(top_k, len(documents)),
                    "return_documents": False,
                },
            },
            timeout=45.0,
        )
        response.raise_for_status()
        raw_results = response.json().get("output", {}).get("results", [])
        results = [
            RerankResult(
                index=int(result["index"]),
                relevance_score=float(result["relevance_score"]),
            )
            for result in raw_results
            if 0 <= int(result["index"]) < len(documents)
        ]
        if not results:
            raise ValueError("Rerank 服务未返回有效排序结果")
        return results
    except ModelCallException:
        raise
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        response_text = exc.response.text[:500] if exc.response is not None else ""
        raise ModelCallException(
            message=(
                f"Reranker 调用失败：HTTP {status_code}，"
                f"响应：{response_text}"
            )
        ) from exc
    except Exception as exc:
        raise ModelCallException(message=f"Reranker 调用失败：{type(exc).__name__}") from exc
