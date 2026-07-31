"""知识库文档解析阶段的 DTO。

无论原文件是 Word、PDF 还是 Excel，后续 RAG 流程都只消费这里定义的统一结构。
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class DocumentUploadResponse(BaseModel):
    """文件安全落盘后的最小响应；后续解析、切块都只使用 document_id。"""

    document_id: str = Field(..., description="服务端生成的文档业务 ID")
    original_file_name: str = Field(..., description="用户上传时的原始文件名，仅用于展示")
    file_type: str = Field(..., description="已校验的文件类型，例如 docx/pdf/xlsx")
    file_size: int = Field(..., ge=1, description="实际写入的文件大小，单位字节")
    status: str = Field(..., description="文档生命周期状态，当前成功上传后为 uploaded")


class ParsedDocumentSegment(BaseModel):
    """一段可定位的原始文本；它还不是 Day19 的检索切块。"""

    segment_index: int = Field(..., ge=0, description="该段在原始文档中的从 0 开始序号")
    text: str = Field(..., min_length=1, description="从原文件解析出的原始文本")
    location: str = Field(..., min_length=1, description="来源定位，例如第 3 页或 Sheet:规则/Row:8")
    # 文档类型各自的补充信息统一放这里，避免为了页码、表名不断给主 DTO 加字段。
    metadata: dict[str, Any] = Field(default_factory=dict, description="来源定位的结构化补充信息")


class DocumentParseResponse(BaseModel):
    """一次解析完成后的摘要响应，不直接返回全部原文段以避免大文档响应过大。"""

    document_id: str = Field(..., description="已解析的文档业务 ID")
    status: str = Field(..., description="解析完成后为 parsed，失败时为 error")
    parser_name: str = Field(..., description="本次实际执行的解析器名称")
    parsed_segment_count: int = Field(..., ge=0, description="本次持久化的有效原始文本段数量")
    preview_segments: list[ParsedDocumentSegment] = Field(
        default_factory=list,
        description="前五个文本段预览；完整结果后续通过分页查询接口获取",
    )


class ParsedDocument(BaseModel):
    """一份文件经解析后的统一输出，供后续切块、向量化和引用来源使用。"""

    document_id: str = Field(..., min_length=1, description="知识库文档业务 ID，后续关联文档表")
    file_name: str = Field(..., min_length=1, description="用户上传时的原始文件名")
    file_type: str = Field(..., min_length=1, description="标准化后的文件类型，例如 docx/pdf/xlsx")
    parser_name: str = Field(..., min_length=1, description="实际执行解析的解析器名称")
    segments: list[ParsedDocumentSegment] = Field(default_factory=list, description="按原文件顺序输出的文本段")
    metadata: dict[str, Any] = Field(default_factory=dict, description="文档级元数据，例如页数、工作表名称")


class ChunkSourceReference(BaseModel):
    """检索切块关联的一条原始文档段来源。"""

    segment_index: int = Field(..., ge=0, description="原始文档段序号")
    location: str = Field(..., min_length=1, description="原始文档中的可追溯位置")


class KnowledgeTextChunk(BaseModel):
    """Day19 的检索前文本切块；Embedding 与向量库后续只处理它。"""

    document_id: str = Field(..., min_length=1, description="所属知识库文档业务 ID")
    chunk_index: int = Field(..., ge=0, description="该切块在文档内的从 0 开始顺序")
    content: str = Field(..., min_length=1, description="长度受控且带上下文重叠的检索文本")
    char_count: int = Field(..., ge=1, description="切块字符数，当前阶段用字符数控制长度")
    source_references: list[ChunkSourceReference] = Field(
        default_factory=list,
        description="该切块覆盖的原始段来源，后续用于回答引用和问题排查",
    )


class EmbeddingSimilarityTestRequest(BaseModel):
    """Embedding 原理验证入参；仅用于对比两段文本，不写入任何知识库数据。"""

    text_a: str = Field(..., min_length=1, max_length=3000, description="待比较的第一段文本")
    text_b: str = Field(..., min_length=1, max_length=3000, description="待比较的第二段文本")


class EmbeddingSimilarityTestResponse(BaseModel):
    """向量只在内存中参与计算，接口不会返回或持久化完整浮点数组。"""

    embedding_model: str = Field(..., description="本次生成两侧向量使用的 Embedding 模型")
    vector_dimension: int = Field(..., ge=1, description="模型输出的向量维度")
    cosine_similarity: float = Field(..., ge=-1, le=1, description="两段文本向量的余弦相似度")


class InMemoryChunkSearchRequest(BaseModel):
    """Day19 教学检索入参；只在指定文档的少量 chunk 中做内存全量扫描。"""

    question: str = Field(..., min_length=1, max_length=1000, description="用户需要检索的自然语言问题")
    top_k: int = Field(3, ge=1, le=10, description="返回相似度最高的前 K 个 chunk")


class InMemoryChunkSearchItem(BaseModel):
    """内存检索命中的一个 chunk 及其可追溯来源。"""

    document_id: str = Field(..., description="命中文档业务 ID")
    chunk_id: str = Field(..., description="MySQL 与向量数据库共用的 chunk 业务 ID")
    chunk_index: int = Field(..., ge=0, description="命中 chunk 的文档内顺序")
    score: float = Field(..., ge=-1, le=1, description="问题向量与 chunk 向量的余弦相似度")
    content: str = Field(..., description="命中 chunk 的原文内容")
    source_references: list[ChunkSourceReference] = Field(
        default_factory=list,
        description="命中 chunk 可回溯的原始文档段来源",
    )


class InMemoryChunkSearchResponse(BaseModel):
    """Day19 内存 Top-K 演示响应；不包含或持久化任何浮点向量。"""

    embedding_model: str = Field(..., description="本次查询与候选块共同使用的 Embedding 模型")
    vector_dimension: int = Field(..., ge=1, description="本次向量维度")
    candidate_count: int = Field(..., ge=1, description="本次在内存中比较的候选 chunk 数量")
    items: list[InMemoryChunkSearchItem] = Field(default_factory=list, description="按相似度降序排列的 Top-K 结果")


class MilvusChunkSearchRequest(BaseModel):
    """Day20 真实 Milvus 检索入参；调用方只指定文档，不允许越过 active 版本直接检索。"""

    question: str = Field(..., min_length=1, max_length=1000, description="用户需要检索的自然语言问题")
    top_k: int = Field(3, ge=1, le=10, description="返回相似度最高的前 K 个 chunk")


class MilvusChunkSearchItem(InMemoryChunkSearchItem):
    """真实向量检索命中项；score 由 Milvus 的 COSINE 距离直接返回。"""

    version_id: str = Field(..., description="本次命中的 active 文档版本业务 ID")


class MilvusChunkSearchResponse(BaseModel):
    """Day20 真实 Milvus Top-K 检索响应，不包含任何向量数组。"""

    document_id: str = Field(..., description="检索的知识库文档业务 ID")
    active_version_id: str = Field(..., description="本次实际参与检索的 active 文档版本 ID")
    embedding_model: str = Field(..., description="问题向量使用的 Embedding 模型")
    vector_dimension: int = Field(..., ge=1, description="查询向量维度")
    vector_collection: str = Field(..., description="执行检索的 Milvus Collection")
    items: list[MilvusChunkSearchItem] = Field(default_factory=list, description="按相似度降序排列的 Top-K 结果")


class CreateDocumentVersionRequest(BaseModel):
    """创建候选索引版本的请求；当前复用已解析原文，仅用于调整切块和向量索引。"""

    change_note: str = Field(..., min_length=1, max_length=500, description="本次候选版本变更原因，便于后续审计和回滚")


class CreateDocumentVersionResponse(BaseModel):
    """候选版本创建结果；新版本未完成构建和激活前不会参与线上检索。"""

    document_id: str = Field(..., description="所属知识库文档业务 ID")
    version_id: str = Field(..., description="新创建的候选文档版本业务 ID")
    version_number: int = Field(..., ge=1, description="文档内递增版本号")
    status: str = Field(..., description="初始状态为 parsed，表示可开始按版本切块")
    change_note: str = Field(..., description="调用方提交的本次变更说明")


class VersionChunkResponse(BaseModel):
    """候选版本独立切块结果；只写入指定 version_id，不影响 active 版本。"""

    document_id: str = Field(..., description="所属知识库文档业务 ID")
    version_id: str = Field(..., description="已完成切块的候选版本 ID")
    status: str = Field(..., description="成功时为 chunked")
    chunk_count: int = Field(..., ge=0, description="该候选版本独立保存的 chunk 数量")
    chunk_config: dict[str, int] = Field(..., description="本次版本切块参数快照")
    preview_chunks: list[KnowledgeTextChunk] = Field(default_factory=list, description="前 3 条候选版本 chunk 预览")


class DocumentVersionIndexResponse(BaseModel):
    """指定文档版本完成 Milvus 索引构建后的摘要。"""

    version_id: str = Field(..., description="已构建向量索引的文档版本业务 ID")
    document_id: str = Field(..., description="所属文档业务 ID")
    status: str = Field(..., description="成功时为 indexed，尚未切换为 active")
    chunk_count: int = Field(..., ge=0, description="该版本的 MySQL chunk 数量")
    vector_count: int = Field(..., ge=0, description="Milvus 中校验到的向量数量")
    embedding_model: str = Field(..., description="本次构建使用的 Embedding 模型")
    embedding_dimension: int = Field(..., ge=1, description="本次构建向量维度")
    vector_collection: str = Field(..., description="实际写入的 Milvus Collection")


class ActivateDocumentVersionRequest(BaseModel):
    """切换当前检索版本的人工审计信息；接入认证后应从登录上下文获取操作者。"""

    activated_by: str = Field(..., min_length=1, max_length=64, description="执行版本切换的人员标识")
    activation_note: str = Field(..., min_length=1, max_length=500, description="确认向量数量和检索质量后的切换说明")


class ActivateDocumentVersionResponse(BaseModel):
    """版本切换结果；Milvus 不复制数据，查询入口只改 MySQL active 指针。"""

    document_id: str = Field(..., description="所属文档业务 ID")
    active_version_id: str = Field(..., description="切换后当前可服务的文档版本 ID")
    previous_version_id: str | None = Field(None, description="切换前的 active 版本 ID，首次上线时为空")
    status: str = Field(..., description="新版本状态，成功时为 active")
    activated_at: str = Field(..., description="切换完成时间")


class DocumentChunkRequest(BaseModel):
    """一次切块任务的参数；后端会把最终取值保存为快照。"""

    max_characters: int = Field(500, ge=100, le=4000, description="单个 chunk 最大字符数")
    overlap_characters: int = Field(80, ge=0, le=1000, description="相邻 chunk 的重叠字符数")
    boundary_search_characters: int = Field(120, ge=1, le=1000, description="向前寻找语义边界的最大字符数")

    @model_validator(mode="after")
    def validate_overlap(self) -> "DocumentChunkRequest":
        # Pydantic 在进入业务服务前完成跨字段校验，类似 Java DTO 上的自定义校验器。
        if self.overlap_characters >= self.max_characters:
            raise ValueError("overlap_characters 必须小于 max_characters")
        return self


class DocumentChunkResponse(BaseModel):
    """一次切块完成后的摘要；完整 chunk 不在此接口一次性返回。"""

    document_id: str = Field(..., description="已切块的文档业务 ID")
    chunk_status: str = Field(..., description="成功时为 chunked")
    chunk_count: int = Field(..., ge=0, description="本次生成并持久化的 chunk 数量")
    chunk_config: dict[str, int] = Field(..., description="本次实际使用的切块参数快照")
    preview_chunks: list[KnowledgeTextChunk] = Field(
        default_factory=list,
        description="前 3 个 chunk 预览；完整结果后续提供分页查询接口",
    )
