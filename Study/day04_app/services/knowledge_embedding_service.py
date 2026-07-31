"""Embedding 原理验证服务；Day20 接入向量数据库前不持久化向量。"""

from __future__ import annotations

import math

from openai import OpenAI

from day04_app.common.exceptions import ERROR_TYPE_MODEL_CALL_FAILED, ModelCallException
from settings import settings


def _create_embedding_client() -> OpenAI:
    """Embedding 与 Chat 复用同一个兼容 API 地址，但调用不同的接口和模型。"""
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        timeout=30.0,
    )


def calculate_cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """计算余弦相似度；只比较方向是否接近，不直接比较向量长度。"""
    if len(vector_a) != len(vector_b) or not vector_a:
        raise ValueError("两个向量必须非空且维度相同")

    dot_product = sum(value_a * value_b for value_a, value_b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("向量范数不能为 0")
    # 浮点计算可能略超出 [-1, 1]，裁剪后满足接口契约。
    return max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))


def compare_text_embeddings(text_a: str, text_b: str) -> tuple[str, int, float]:
    """用同一模型批量生成两段文本的向量，并仅返回可解释的比较指标。"""
    model, vectors = generate_text_embeddings([text_a, text_b])
    similarity = calculate_cosine_similarity(vectors[0], vectors[1])
    return model, len(vectors[0]), similarity


def generate_text_embeddings(texts: list[str], batch_size: int = 10) -> tuple[str, list[list[float]]]:
    """批量生成向量并只返回内存结果；调用方决定是否交给向量数据库持久化。"""
    if not texts:
        raise ValueError("texts 不能为空")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    client = _create_embedding_client()
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            # 每批 data 的 index 相对当前批次从 0 开始，因此先排序再追加以保持输入顺序。
            response = client.embeddings.create(
                model=settings.dashscope_embedding_model,
                input=batch_texts,
                encoding_format="float",
            )
            response_data = sorted(response.data, key=lambda item: item.index)
            if len(response_data) != len(batch_texts):
                raise ValueError("Embedding 服务未返回完整向量")
            vectors.extend(item.embedding for item in response_data)

        vector_dimensions = {len(vector) for vector in vectors}
        if len(vector_dimensions) != 1 or 0 in vector_dimensions:
            raise ValueError("Embedding 服务返回的向量维度不一致或为空")
        return settings.dashscope_embedding_model, vectors
    except ModelCallException:
        raise
    except Exception as exc:
        raise ModelCallException(
            message=f"Embedding 向量生成失败：{type(exc).__name__}",
            error_type=ERROR_TYPE_MODEL_CALL_FAILED,
        ) from exc
