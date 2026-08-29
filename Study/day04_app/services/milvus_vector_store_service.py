"""Milvus Collection 初始化与后续向量存取的统一基础服务。"""

from __future__ import annotations

from typing import Any

from pymilvus import DataType, MilvusClient

from settings import settings


EMBEDDING_DIMENSION = 1024
VECTOR_FIELD_NAME = "embedding"
VECTOR_METRIC_TYPE = "COSINE"
VECTOR_INDEX_TYPE = "AUTOINDEX"
VECTOR_INDEX_NAME = "embedding_auto_index"


def create_milvus_client() -> MilvusClient:
    """集中创建客户端，后续写入、检索和健康检查都复用同一连接配置。"""
    return MilvusClient(uri=settings.milvus_uri)


def _validate_existing_collection(description: dict[str, Any]) -> None:
    """已存在的同名 Collection 必须满足当前代码契约，防止错误维度污染数据。"""
    fields = {
        field["name"]: field
        for field in description.get("fields", [])
    }
    required_fields = {"chunk_id", "document_id", "version_id", VECTOR_FIELD_NAME}
    missing_fields = required_fields - fields.keys()
    if missing_fields:
        raise ValueError(f"Milvus Collection 缺少字段：{sorted(missing_fields)}")

    vector_params = fields[VECTOR_FIELD_NAME].get("params", {})
    actual_dimension = int(vector_params.get("dim", 0))
    if actual_dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Milvus 向量维度不匹配：期望 {EMBEDDING_DIMENSION}，实际 {actual_dimension}"
        )


def _ensure_vector_index(client: MilvusClient, collection_name: str) -> None:
    """Collection 可加载前必须有向量索引；开发环境用 AUTOINDEX，生产再按压测选择 HNSW/IVF。"""
    index_names = client.list_indexes(
        collection_name=collection_name,
        field_name=VECTOR_FIELD_NAME,
    )
    if index_names:
        return
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD_NAME,
        index_type=VECTOR_INDEX_TYPE,
        metric_type=VECTOR_METRIC_TYPE,
        index_name=VECTOR_INDEX_NAME,
    )
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )


def ensure_knowledge_chunk_collection() -> dict[str, Any]:
    """幂等创建并加载知识库向量 Collection，确保后续 query/search 可执行。"""
    client = create_milvus_client()
    collection_name = settings.milvus_collection_name
    try:
        if not client.has_collection(collection_name=collection_name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
                description="知识库 chunk Embedding 向量索引",
            )
            schema.add_field(
                field_name="chunk_id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=64,
                description="对应 MySQL knowledge_document_chunk 的业务 ID",
            )
            schema.add_field(
                field_name="document_id",
                datatype=DataType.VARCHAR,
                max_length=64,
                description="所属知识库文档业务 ID，用于检索过滤",
            )
            schema.add_field(
                field_name="version_id",
                datatype=DataType.VARCHAR,
                max_length=64,
                description="所属索引版本业务 ID，用于新旧版本并存和检索过滤",
            )
            schema.add_field(
                field_name=VECTOR_FIELD_NAME,
                datatype=DataType.FLOAT_VECTOR,
                dim=EMBEDDING_DIMENSION,
                description="text-embedding-v3 生成的 1024 维向量",
            )
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )

        description = client.describe_collection(collection_name=collection_name)
        _validate_existing_collection(description)
        _ensure_vector_index(client, collection_name)
        # Collection 创建或服务重启后可能处于未加载状态，query/search 前必须加载到查询节点。
        client.load_collection(collection_name=collection_name)
        return {
            "collection_name": collection_name,
            "vector_dimension": EMBEDDING_DIMENSION,
            "vector_field": VECTOR_FIELD_NAME,
            "metric_type": VECTOR_METRIC_TYPE,
            "index_type": VECTOR_INDEX_TYPE,
            "field_names": [field["name"] for field in description.get("fields", [])],
        }
    finally:
        client.close()


def ensure_session_memory_collection() -> dict[str, Any]:
    """幂等创建会话长期记忆 collection，和企业知识库物理隔离。

    知识库检索按 document/version 治理；会话记忆必须按 session/user/tenant 可见范围过滤。
    二者不能共用 Collection，否则很容易在查询时发生资料越界或召回语义串扰。
    """

    client = create_milvus_client()
    collection_name = settings.session_memory_collection_name
    try:
        if not client.has_collection(collection_name=collection_name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
                description="受治理的会话长期记忆向量索引",
            )
            schema.add_field(field_name="memory_id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field(field_name="session_id", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="memory_type", datatype=DataType.VARCHAR, max_length=32)
            schema.add_field(field_name="status", datatype=DataType.VARCHAR, max_length=32)
            schema.add_field(field_name=VECTOR_FIELD_NAME, datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)
            client.create_collection(collection_name=collection_name, schema=schema)
        _ensure_vector_index(client, collection_name)
        client.load_collection(collection_name=collection_name)
        return {
            "collection_name": collection_name,
            "vector_dimension": EMBEDDING_DIMENSION,
            "vector_field": VECTOR_FIELD_NAME,
            "metric_type": VECTOR_METRIC_TYPE,
        }
    finally:
        client.close()


def upsert_session_memory_vector(*, record: dict[str, Any]) -> None:
    """按 memory_id 幂等写入一条已治理的长期记忆向量。"""

    vector = record.get(VECTOR_FIELD_NAME)
    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION:
        raise ValueError("会话记忆向量维度与 Milvus Collection 契约不一致")
    ensure_session_memory_collection()
    client = create_milvus_client()
    try:
        client.upsert(collection_name=settings.session_memory_collection_name, data=[record])
        client.flush(collection_name=settings.session_memory_collection_name)
    finally:
        client.close()


def search_session_memory_vectors(
    *,
    question_vector: list[float],
    session_id: str,
    user_id: str | None,
    tenant_id: str | None,
    top_k: int,
) -> list[dict[str, Any]]:
    """按可信范围过滤后检索长期记忆；过滤先于向量相似度返回结果。"""

    if len(question_vector) != EMBEDDING_DIMENSION:
        raise ValueError("会话记忆查询向量维度不匹配")
    import json

    # 第一版只允许当前 session；未来扩展跨会话记忆时必须同时收紧 user/tenant 条件。
    filters = [f"session_id == {json.dumps(session_id, ensure_ascii=False)}", 'status == "active"']
    if user_id is not None:
        filters.append(f"user_id == {json.dumps(user_id, ensure_ascii=False)}")
    if tenant_id is not None:
        filters.append(f"tenant_id == {json.dumps(tenant_id, ensure_ascii=False)}")
    client = create_milvus_client()
    try:
        client.load_collection(collection_name=settings.session_memory_collection_name)
        result = client.search(
            collection_name=settings.session_memory_collection_name,
            data=[question_vector],
            anns_field=VECTOR_FIELD_NAME,
            filter=" and ".join(filters),
            limit=top_k,
            output_fields=["memory_id", "session_id", "user_id", "tenant_id", "memory_type", "status"],
            search_params={"metric_type": VECTOR_METRIC_TYPE},
        )
        return result[0] if result else []
    finally:
        client.close()


def upsert_chunk_vectors(
    *,
    records: list[dict[str, Any]],
) -> None:
    """按 chunk_id 幂等写入向量；重复构建同一版本不会产生重复 Entity。"""
    if not records:
        return
    client = create_milvus_client()
    try:
        client.upsert(
            collection_name=settings.milvus_collection_name,
            data=records,
        )
        # Milvus 写入与后续统计/检索之间显式 flush，避免误把未落盘数据当成缺失。
        client.flush(collection_name=settings.milvus_collection_name)
    finally:
        client.close()


def count_vectors_by_version(version_id: str) -> int:
    """按版本统计向量数量，用于构建完成后的数量校验。"""
    client = create_milvus_client()
    try:
        # 单独调用该方法时也保证 Collection 可查询，不能假设调用方已经完成加载。
        client.load_collection(collection_name=settings.milvus_collection_name)
        result = client.query(
            collection_name=settings.milvus_collection_name,
            filter=f'version_id == "{version_id}"',
            output_fields=["count(*)"],
        )
        if not result:
            return 0
        return int(result[0].get("count(*)", 0))
    finally:
        client.close()


def search_chunk_vectors(
    *,
    version_id: str,
    question_vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """在指定文档版本内执行向量 Top-K 检索，只返回跨库关联所需的最小字段。"""
    if len(question_vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"查询向量维度不匹配：期望 {EMBEDDING_DIMENSION}，实际 {len(question_vector)}"
        )

    # active_version_id 来自 MySQL 内部状态，不接受调用方直接拼接 Milvus 过滤条件。
    escaped_version_id = version_id.replace("\\", "\\\\").replace('"', '\\"')
    client = create_milvus_client()
    try:
        _ensure_vector_index(client, settings.milvus_collection_name)
        client.load_collection(collection_name=settings.milvus_collection_name)
        result = client.search(
            collection_name=settings.milvus_collection_name,
            data=[question_vector],
            anns_field=VECTOR_FIELD_NAME,
            filter=f'version_id == "{escaped_version_id}"',
            limit=top_k,
            output_fields=["chunk_id", "document_id", "version_id"],
            search_params={"metric_type": VECTOR_METRIC_TYPE},
        )
        return result[0] if result else []
    finally:
        client.close()


def search_chunk_vectors_for_versions(
    *,
    version_ids: list[str],
    question_vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """在多个*已由项目确认*的文档版本中执行一次统一向量召回。

    多文档 RAG 不能简单循环调用 ``search_chunk_vectors``：那会对同一个问题重复
    Embedding，并让每篇文档各自截断 Top-K，最终无法按全局相关度排序。本函数只接受
    服务层从 MySQL active 指针取出的版本 ID；HTTP 调用方不能直接传 Milvus 过滤条件。
    """

    if not version_ids:
        return []
    if len(question_vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"查询向量维度不匹配：期望 {EMBEDDING_DIMENSION}，实际 {len(question_vector)}"
        )
    # 使用 JSON 字符串字面量构造 Milvus ``in`` 条件，避免业务 ID 中的引号破坏过滤表达式。
    # version_ids 来自 MySQL 状态，不是用户直接提交给 Milvus 的自由表达式。
    import json

    unique_version_ids = list(dict.fromkeys(version_ids))
    version_literals = ", ".join(json.dumps(version_id, ensure_ascii=False) for version_id in unique_version_ids)
    client = create_milvus_client()
    try:
        _ensure_vector_index(client, settings.milvus_collection_name)
        client.load_collection(collection_name=settings.milvus_collection_name)
        result = client.search(
            collection_name=settings.milvus_collection_name,
            data=[question_vector],
            anns_field=VECTOR_FIELD_NAME,
            filter=f"version_id in [{version_literals}]",
            limit=top_k,
            output_fields=["chunk_id", "document_id", "version_id"],
            search_params={"metric_type": VECTOR_METRIC_TYPE},
        )
        return result[0] if result else []
    finally:
        client.close()
