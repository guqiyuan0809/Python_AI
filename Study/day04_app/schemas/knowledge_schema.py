"""知识库文档解析阶段的 DTO。

无论原文件是 Word、PDF 还是 Excel，后续 RAG 流程都只消费这里定义的统一结构。
"""

from typing import Any, Literal

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
    parent_index: int | None = Field(
        None,
        ge=0,
        description="父子切块模式下所属父块顺序；普通切块为空",
    )
    content: str = Field(..., min_length=1, description="长度受控且带上下文重叠的检索文本")
    char_count: int = Field(..., ge=1, description="切块字符数，当前阶段用字符数控制长度")
    source_references: list[ChunkSourceReference] = Field(
        default_factory=list,
        description="该切块覆盖的原始段来源，后续用于回答引用和问题排查",
    )


class KnowledgeParentTextChunk(BaseModel):
    """父块是回答阶段的完整上下文，不直接参与向量粗排。"""

    document_id: str = Field(..., min_length=1, description="所属知识库文档业务 ID")
    parent_index: int = Field(..., ge=0, description="父块在文档版本内的顺序")
    content: str = Field(..., min_length=1, description="由原始段拼接的完整父级原文")
    char_count: int = Field(..., ge=1, description="父块原文字符数")
    source_references: list[ChunkSourceReference] = Field(
        default_factory=list,
        description="父块覆盖的原始文档段来源",
    )


class ParentChildTextChunkBuildResult(BaseModel):
    """父子切块结果：父块回填回答上下文，子块用于向量检索。"""

    parent_chunks: list[KnowledgeParentTextChunk] = Field(default_factory=list)
    child_chunks: list[KnowledgeTextChunk] = Field(default_factory=list)


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
    use_reranker: bool = Field(False, description="是否对 Milvus 粗排结果执行 Reranker 精排")
    rerank_top_n: int = Field(20, ge=1, le=50, description="启用精排时先从 Milvus 粗排召回的候选数量")

    @model_validator(mode="after")
    def validate_rerank_size(self) -> "MilvusChunkSearchRequest":
        if self.use_reranker and self.rerank_top_n < self.top_k:
            raise ValueError("启用 Reranker 时 rerank_top_n 必须大于等于 top_k")
        return self


class MilvusChunkSearchItem(InMemoryChunkSearchItem):
    """真实向量检索命中项；score 由 Milvus 的 COSINE 距离直接返回。"""

    version_id: str = Field(..., description="本次命中的 active 文档版本业务 ID")
    parent_chunk_id: str | None = Field(
        None,
        description="父子切块模式下所属父块 ID；普通切块为空",
    )
    vector_score: float | None = Field(
        None,
        ge=-1,
        le=1,
        description="Reranker 启用时保留的 Milvus 原始相似度分数",
    )
    rerank_score: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Reranker 相关性分数；未启用精排时为空",
    )
    parent_content: str | None = Field(
        None,
        description="父子切块命中后回填的完整父块原文；只在回答资料组装时使用",
    )
    parent_source_references: list[ChunkSourceReference] = Field(
        default_factory=list,
        description="父块覆盖的原文来源，仅用于回答引用；检索评测仍使用子块来源",
    )


class MilvusChunkSearchResponse(BaseModel):
    """Day20 真实 Milvus Top-K 检索响应，不包含任何向量数组。"""

    document_id: str = Field(..., description="检索的知识库文档业务 ID")
    active_version_id: str = Field(..., description="本次实际参与检索的 active 文档版本 ID")
    embedding_model: str = Field(..., description="问题向量使用的 Embedding 模型")
    vector_dimension: int = Field(..., ge=1, description="查询向量维度")
    vector_collection: str = Field(..., description="执行检索的 Milvus Collection")
    items: list[MilvusChunkSearchItem] = Field(default_factory=list, description="按相似度降序排列的 Top-K 结果")


class LlamaIndexLawRetrievalRequest(MilvusChunkSearchRequest):
    """Day31 法规检索适配层入参。

    与现有 Milvus 检索使用相同参数，保证框架接入前后可以在同一 active 文档版本上对照。
    """


class LlamaIndexLawSourceNode(BaseModel):
    """LlamaIndex NodeWithScore 的安全可读投影，不直接向客户端暴露框架内部对象。"""

    node_id: str = Field(..., description="LlamaIndex TextNode ID；与项目 chunk_id 保持一致")
    document_id: str = Field(..., description="节点所属知识库文档 ID")
    version_id: str = Field(..., description="节点所属 active 文档版本 ID")
    chunk_index: int = Field(..., ge=0, description="命中子 chunk 的文档内顺序")
    parent_chunk_id: str | None = Field(None, description="父子切块模式下的父块 ID")
    score: float = Field(..., ge=-1, le=1, description="LlamaIndex NodeWithScore 的最终相关性分数")
    vector_score: float | None = Field(None, ge=-1, le=1, description="Milvus 原始余弦相似度")
    rerank_score: float | None = Field(None, ge=0, le=1, description="启用 Reranker 后的相关性分数")
    content: str = Field(..., description="节点可提供给后续 Query Engine 的文本；父子模式下为父块全文")
    source_locations: list[str] = Field(default_factory=list, description="原始法规文档中的页码、段落等来源位置")


class LlamaIndexLawRetrievalResponse(BaseModel):
    """LlamaIndex 法规检索预览：只验证框架节点适配，尚未调用 LLM 生成回答。"""

    framework: str = Field("llamaindex", description="本次编排框架")
    retrieval_backend: str = Field("project_milvus", description="实际向量检索后端")
    document_id: str = Field(..., description="检索的知识库文档 ID")
    active_version_id: str = Field(..., description="本次使用的 active 文档版本 ID")
    embedding_model: str = Field(..., description="查询向量使用的 Embedding 模型")
    vector_dimension: int = Field(..., ge=1, description="查询向量维度")
    node_count: int = Field(..., ge=0, description="LlamaIndex Retriever 返回的 NodeWithScore 数量")
    nodes: list[LlamaIndexLawSourceNode] = Field(default_factory=list, description="转换后的法规知识节点")


class LlamaIndexChunkPreviewRequest(BaseModel):
    """Day31 LlamaIndex 文档切块预览参数；不会写入线上候选版本。"""

    chunk_size: int = Field(512, ge=128, le=4096, description="LlamaIndex SentenceSplitter 的 token 预算")
    chunk_overlap: int = Field(64, ge=0, le=1024, description="相邻节点重叠的 token 预算")

    @model_validator(mode="after")
    def validate_overlap(self) -> "LlamaIndexChunkPreviewRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self


class LlamaIndexChunkPreviewNode(BaseModel):
    """LlamaIndex TextNode 的稳定 HTTP 投影。"""

    node_id: str = Field(..., description="LlamaIndex 节点 ID")
    content: str = Field(..., description="节点文本")
    document_id: str = Field(..., description="知识库文档业务 ID")
    source_segment_index: int | None = Field(None, description="原始解析段序号")
    source_location: str | None = Field(None, description="原始文档位置")
    start_char_idx: int | None = Field(None, ge=0, description="节点在原始段中的起始字符位置")
    end_char_idx: int | None = Field(None, ge=0, description="节点在原始段中的结束字符位置")


class LlamaIndexChunkPreviewResponse(BaseModel):
    """LlamaIndex 文档切块预览响应；结果未写入 MySQL/Milvus。"""

    framework: str = Field("llamaindex", description="本次文档处理框架")
    document_id: str = Field(..., description="知识库文档业务 ID")
    source_segment_count: int = Field(..., ge=0, description="参与切块的原始段数量")
    node_count: int = Field(..., ge=0, description="LlamaIndex 生成的节点数量")
    chunk_size: int = Field(..., ge=1, description="实际使用的 token 切块预算")
    chunk_overlap: int = Field(..., ge=0, description="实际使用的 token 重叠预算")
    persisted: bool = Field(False, description="本接口仅预览，不持久化节点")
    nodes: list[LlamaIndexChunkPreviewNode] = Field(default_factory=list, description="前 N 个节点预览")


class DocumentVersionChunkSearchResponse(BaseModel):
    """发布前版本验证检索结果；version_id 可为候选 indexed 版本，不代表线上 active 版本。"""

    document_id: str = Field(..., description="检索的知识库文档业务 ID")
    version_id: str = Field(..., description="本次实际参与验证检索的文档版本 ID")
    embedding_model: str = Field(..., description="问题向量使用的 Embedding 模型")
    vector_dimension: int = Field(..., ge=1, description="查询向量维度")
    vector_collection: str = Field(..., description="执行检索的 Milvus Collection")
    items: list[MilvusChunkSearchItem] = Field(default_factory=list, description="按相似度降序排列的 Top-K 结果")


class ParentChildChunkRequest(BaseModel):
    """父子切块参数：先构造较完整父块，再切成更小的向量检索子块。"""

    parent_max_characters: int = Field(1800, ge=500, le=8000, description="父块推荐字符预算")
    parent_min_characters: int = Field(600, ge=100, le=4000, description="父块最小推荐字符数")
    child_max_characters: int = Field(260, ge=100, le=1000, description="子块推荐字符预算")
    child_overlap_characters: int = Field(40, ge=0, le=300, description="子块超长硬切时的重叠字符数")
    child_min_characters: int = Field(60, ge=1, le=1000, description="子块最小推荐字符数")
    child_semantic_overflow_characters: int = Field(60, ge=0, le=300, description="子块语义完整性溢出预算")

    @model_validator(mode="after")
    def validate_parent_child_sizes(self) -> "ParentChildChunkRequest":
        if self.parent_min_characters > self.parent_max_characters:
            raise ValueError("parent_min_characters 必须不大于 parent_max_characters")
        if self.child_min_characters > self.child_max_characters:
            raise ValueError("child_min_characters 必须不大于 child_max_characters")
        if self.child_overlap_characters >= self.child_max_characters:
            raise ValueError("child_overlap_characters 必须小于 child_max_characters")
        return self


class ContextualIndexBuildRequest(BaseModel):
    """上下文化索引构建请求；只允许对已完成父子切块的候选版本执行。"""

    context_model: str | None = Field(None, max_length=64, description="生成子块背景说明的模型，默认使用服务配置模型")
    context_max_tokens: int = Field(180, ge=50, le=500, description="每个子块背景说明的最大输出 Token")


class ParentChildChunkResponse(BaseModel):
    document_id: str = Field(..., description="所属知识库文档业务 ID")
    version_id: str = Field(..., description="已完成父子切块的候选版本 ID")
    status: str = Field(..., description="成功时为 chunked")
    parent_chunk_count: int = Field(..., ge=0, description="生成的父块数量")
    child_chunk_count: int = Field(..., ge=0, description="生成的子块数量，也将成为向量数量")
    chunk_config: dict[str, int | str] = Field(default_factory=dict, description="父子切块参数快照")
    preview_parent_chunks: list[KnowledgeParentTextChunk] = Field(default_factory=list, description="前 2 个父块预览")
    preview_child_chunks: list[KnowledgeTextChunk] = Field(default_factory=list, description="前 3 个子块预览")


class CreateRetrievalEvalDatasetRequest(BaseModel):
    """创建一份 RAG 检索评测数据集；当前一个数据集对应一个知识库文档。"""

    dataset_name: str = Field(..., min_length=1, max_length=64, description="数据集名称，例如 jvm_knowledge_retrieval")
    dataset_version: str = Field(..., min_length=1, max_length=64, description="数据集版本，例如 v1")
    document_id: str = Field(..., min_length=1, max_length=64, description="待评测的知识库文档业务 ID")
    description: str | None = Field(None, max_length=500, description="数据集用途和覆盖范围说明")
    created_by: str | None = Field(None, min_length=1, max_length=64, description="创建人标识")


class CreateRetrievalEvalSampleRequest(BaseModel):
    """标注一条检索样本；期望段落使用原始 segment_index，不能使用易变化的 chunk_id。"""

    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    sample_type: str = Field("normal", pattern="^(normal|boundary|no_answer)$", description="样本类型")
    expected_answerable: bool = Field(..., description="知识库是否应能回答该问题")
    expected_segment_indexes: list[int] = Field(default_factory=list, description="人工标注的期望原文段序号")
    expected_note: str | None = Field(None, max_length=2000, description="标注理由或命中依据说明")
    created_by: str | None = Field(None, min_length=1, max_length=64, description="标注人标识")

    @model_validator(mode="after")
    def validate_expected_segments(self) -> "CreateRetrievalEvalSampleRequest":
        if self.expected_answerable and not self.expected_segment_indexes:
            raise ValueError("可回答样本必须标注至少一个 expected_segment_indexes")
        if not self.expected_answerable and self.expected_segment_indexes:
            raise ValueError("不可回答样本不能标注 expected_segment_indexes")
        if self.sample_type == "no_answer" and self.expected_answerable:
            raise ValueError("no_answer 样本的 expected_answerable 必须为 false")
        return self


class RetrievalEvalDatasetItem(BaseModel):
    dataset_id: str = Field(..., description="检索评测数据集业务 ID")
    dataset_name: str = Field(..., description="数据集名称")
    dataset_version: str = Field(..., description="数据集版本")
    document_id: str = Field(..., description="关联知识库文档业务 ID")
    description: str | None = Field(None, description="数据集说明")
    sample_count: int = Field(..., ge=0, description="当前样本数")
    status: str = Field(..., description="数据集状态")
    created_by: str | None = Field(None, description="创建人")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")


class RetrievalEvalSampleItem(BaseModel):
    sample_id: str = Field(..., description="检索评测样本业务 ID")
    dataset_id: str = Field(..., description="所属数据集业务 ID")
    question: str = Field(..., description="评测问题")
    sample_type: str = Field(..., description="样本类型")
    expected_answerable: bool = Field(..., description="是否期望知识库可回答")
    expected_segment_indexes: list[int] = Field(default_factory=list, description="期望原文段序号")
    expected_note: str | None = Field(None, description="标注理由")
    status: str = Field(..., description="样本状态")
    created_by: str | None = Field(None, description="标注人")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")


class RetrievalEvalDatasetPageResponse(BaseModel):
    total: int = Field(..., ge=0, description="符合条件的数据集总数")
    page: int = Field(..., ge=1, description="当前页码")
    page_size: int = Field(..., ge=1, description="当前页大小")
    items: list[RetrievalEvalDatasetItem] = Field(default_factory=list, description="数据集列表")


class RetrievalEvalSamplePageResponse(BaseModel):
    dataset_id: str = Field(..., description="所属检索评测数据集业务 ID")
    total: int = Field(..., ge=0, description="样本总数")
    page: int = Field(..., ge=1, description="当前页码")
    page_size: int = Field(..., ge=1, description="当前页大小")
    items: list[RetrievalEvalSampleItem] = Field(default_factory=list, description="样本列表")


class RunRetrievalEvalRequest(BaseModel):
    """对固定文档版本运行一次检索评测；候选 indexed 版本不会被激活。"""

    dataset_id: str = Field(..., min_length=1, max_length=64, description="参与评测的数据集业务 ID")
    document_version_id: str = Field(..., min_length=1, max_length=64, description="被测文档版本业务 ID，可为 indexed 或 active")
    retrieval_top_k: int = Field(5, ge=1, le=10, description="所有样本统一使用的 Top-K")
    score_threshold: float | None = Field(
        None,
        ge=-1,
        le=1,
        description="可选拒答阈值；仅用于计算 no_answer 样本误放行率",
    )
    created_by: str | None = Field(None, min_length=1, max_length=64, description="评测发起人标识")


    use_reranker: bool = Field(False, description="是否对 Milvus 粗排结果执行 Reranker 精排")
    rerank_top_n: int = Field(20, ge=1, le=50, description="Reranker 精排前保留的粗排候选数")
    sample_limit: int | None = Field(
        None,
        ge=1,
        le=100,
        description="可选：本次只评测前 N 条 active 样本，用于学习阶段低成本试跑；正式准入应为空",
    )

    @model_validator(mode="after")
    def validate_rerank_size(self) -> "RunRetrievalEvalRequest":
        if self.use_reranker and self.rerank_top_n < self.retrieval_top_k:
            raise ValueError("启用 Reranker 时 rerank_top_n 必须大于等于 retrieval_top_k")
        return self


class RetrievalEvalRunItem(BaseModel):
    use_reranker: bool = Field(False, description="本次是否启用 Reranker 精排")
    rerank_top_n: int | None = Field(None, description="本次精排使用的粗排候选数")
    reranker_model: str | None = Field(None, description="本次使用的 Reranker 模型")
    """一次检索评测运行的汇总结果。"""

    run_id: str = Field(..., description="检索评测运行业务 ID")
    dataset_id: str = Field(..., description="使用的数据集业务 ID")
    document_id: str = Field(..., description="被测知识库文档业务 ID")
    document_version_id: str = Field(..., description="被测文档索引版本业务 ID")
    retrieval_top_k: int = Field(..., ge=1, description="本次统一的 Top-K")
    score_threshold: float | None = Field(None, description="本次使用的可选拒答阈值")
    embedding_model: str | None = Field(None, description="本次实际使用的查询 Embedding 模型")
    vector_dimension: int | None = Field(None, description="本次查询向量维度")
    status: str = Field(..., description="运行状态：running、success、partial_success、error")
    sample_count: int = Field(..., ge=0, description="参与评测的样本总数")
    success_count: int = Field(..., ge=0, description="成功完成检索的样本数")
    error_count: int = Field(..., ge=0, description="检索异常的样本数")
    answerable_sample_count: int = Field(..., ge=0, description="可回答样本数")
    answerable_hit_count: int = Field(..., ge=0, description="Top-K 命中正确证据的可回答样本数")
    total_expected_segment_count: int = Field(..., ge=0, description="可回答样本期望原始段总数")
    total_hit_segment_count: int = Field(..., ge=0, description="Top-K 命中的期望原始段总数")
    total_retrieved_chunk_count: int = Field(..., ge=0, description="可回答样本返回 chunk 总数")
    total_relevant_retrieved_chunk_count: int = Field(..., ge=0, description="可回答样本返回的正确 chunk 总数")
    hit_at_k: float | None = Field(None, description="Hit@K")
    recall_at_k: float | None = Field(None, description="Recall@K")
    precision_at_k: float | None = Field(None, description="Precision@K")
    mrr_at_k: float | None = Field(None, description="MRR@K")
    no_answer_sample_count: int = Field(..., ge=0, description="无答案样本数")
    no_answer_false_positive_count: int | None = Field(None, description="无答案误放行样本数")
    no_answer_false_positive_rate: float | None = Field(None, description="无答案误放行率")
    no_answer_avg_top_score: float | None = Field(None, description="无答案样本 Top-1 分数均值")
    elapsed_ms: int | None = Field(None, description="评测总耗时，单位毫秒")
    error_message: str | None = Field(None, description="运行级错误信息")
    created_by: str | None = Field(None, description="评测发起人")
    started_at: str = Field(..., description="开始时间")
    finished_at: str | None = Field(None, description="结束时间")


class RetrievalEvalRunPageResponse(BaseModel):
    total: int = Field(..., ge=0, description="符合条件的评测运行总数")
    page: int = Field(..., ge=1, description="当前页码")
    page_size: int = Field(..., ge=1, description="当前页大小")
    items: list[RetrievalEvalRunItem] = Field(default_factory=list, description="评测运行列表")


class RetrievalEvalRetrievedChunkItem(BaseModel):
    """评测运行时一个召回 chunk 的最小快照，不复制原文以控制明细体积。"""

    rank: int = Field(..., ge=1, description="Milvus 命中排名，从 1 开始")
    chunk_id: str = Field(..., description="命中的 chunk 业务 ID")
    chunk_index: int = Field(..., ge=0, description="chunk 在文档版本中的顺序")
    score: float = Field(..., ge=-1, le=1, description="Milvus 返回的相似度分数")
    segment_indexes: list[int] = Field(default_factory=list, description="该 chunk 覆盖的原始段序号")


class RetrievalEvalCaseResultItem(BaseModel):
    """单条问题在指定文档版本上的检索评测结果。"""

    case_result_id: str = Field(..., description="评测样本结果业务 ID")
    run_id: str = Field(..., description="所属评测运行 ID")
    sample_id: str = Field(..., description="关联评测样本 ID")
    question: str = Field(..., description="运行时问题快照")
    sample_type: str = Field(..., description="样本类型")
    expected_answerable: bool = Field(..., description="是否期望知识库可回答")
    expected_segment_indexes: list[int] = Field(default_factory=list, description="人工标注的期望原始段")
    retrieved_segment_indexes: list[int] = Field(default_factory=list, description="实际召回的原始段")
    retrieved_chunks: list[RetrievalEvalRetrievedChunkItem] = Field(default_factory=list, description="按排名保存的召回块快照")
    first_hit_rank: int | None = Field(None, description="首个正确依据的 chunk 排名")
    is_hit: bool | None = Field(None, description="可回答样本是否命中正确依据")
    hit_segment_count: int | None = Field(None, description="Top-K 命中的期望原始段去重数量")
    expected_segment_count: int | None = Field(None, description="人工标注的期望原始段数量")
    relevant_retrieved_chunk_count: int | None = Field(None, description="Top-K 中包含期望原始段的 chunk 数量")
    precision_at_k: float | None = Field(None, description="本样本 Precision@K")
    top_score: float | None = Field(None, description="Top-1 相似度分数")
    is_false_positive: bool | None = Field(None, description="无答案样本是否被阈值误判为可回答")
    elapsed_ms: int | None = Field(None, description="本条样本检索耗时，单位毫秒")
    status: str = Field(..., description="明细状态：success 或 error")
    error_type: str | None = Field(None, description="失败异常类型")
    error_message: str | None = Field(None, description="失败异常信息")
    created_at: str = Field(..., description="明细创建时间")


class RetrievalEvalCaseResultPageResponse(BaseModel):
    run_id: str = Field(..., description="所属评测运行 ID")
    total: int = Field(..., ge=0, description="评测明细总数")
    page: int = Field(..., ge=1, description="当前页码")
    page_size: int = Field(..., ge=1, description="当前页大小")
    items: list[RetrievalEvalCaseResultItem] = Field(default_factory=list, description="评测明细列表")


class DocumentSegmentItem(BaseModel):
    """原始解析段的分页展示 DTO；用于人工标注检索评测依据。"""

    segment_index: int = Field(..., ge=0, description="原始文档段序号，是跨切块版本的稳定标注锚点")
    content: str = Field(..., description="解析得到的原始文本")
    location: str = Field(..., description="原文定位，例如 Paragraph:43")


class DocumentSegmentPageResponse(BaseModel):
    document_id: str = Field(..., description="知识库文档业务 ID")
    total: int = Field(..., ge=0, description="原始文本段总数")
    page: int = Field(..., ge=1, description="当前页码")
    page_size: int = Field(..., ge=1, description="当前页大小")
    items: list[DocumentSegmentItem] = Field(default_factory=list, description="原始文本段列表")


class RagContextPreviewRequest(BaseModel):
    """Day21 RAG 上下文预览入参；仅用于观察检索资料如何进入模型 Prompt。"""

    question: str = Field(..., min_length=1, max_length=1000, description="需要基于知识库回答的问题")
    retrieval_top_k: int = Field(5, ge=1, le=10, description="先从 Milvus 召回的候选 chunk 数量")
    max_context_characters: int = Field(
        4000,
        ge=1000,
        le=12000,
        description="资料包最大字符数；当前阶段按字符近似控制，后续升级为 Token 预算",
    )
    use_reranker: bool = Field(False, description="是否对 Milvus 粗排候选执行 Reranker 精排")
    rerank_top_n: int = Field(20, ge=1, le=50, description="启用精排时的 Milvus 粗排候选数")
    score_threshold: float | None = Field(
        None,
        ge=-1,
        le=1,
        description="可选拒答阈值；Top1 分数低于该值时不组装资料包也不调用聊天模型",
    )

    @model_validator(mode="after")
    def validate_rerank_size(self) -> "RagContextPreviewRequest":
        if self.use_reranker and self.rerank_top_n < self.retrieval_top_k:
            raise ValueError("启用 Reranker 时 rerank_top_n 必须大于等于 retrieval_top_k")
        return self


class RagContextReference(BaseModel):
    """RAG 资料包中一个可被模型引用的来源编号。"""

    source_id: str = Field(..., description="模型回答中使用的引用编号，例如 S1")
    document_id: str = Field(..., description="来源知识库文档业务 ID")
    version_id: str = Field(..., description="来源文档的 active 版本 ID")
    chunk_id: str = Field(..., description="来源检索块业务 ID")
    chunk_index: int = Field(..., ge=0, description="来源 chunk 的文档内顺序")
    score: float = Field(..., ge=-1, le=1, description="Milvus 返回的余弦相似度")
    locations: list[str] = Field(default_factory=list, description="原文段落、页码或表格行等可追溯位置")


class RagContextPreviewResponse(BaseModel):
    """仅用于开发核验的 RAG 资料包，生产回答接口不会直接暴露内部 Prompt。"""

    document_id: str = Field(..., description="检索的知识库文档业务 ID")
    active_version_id: str = Field(..., description="本次检索使用的 active 文档版本 ID")
    embedding_model: str = Field(..., description="查询向量使用的 Embedding 模型")
    vector_dimension: int = Field(..., ge=1, description="查询向量维度")
    retrieved_chunk_count: int = Field(..., ge=0, description="Milvus 实际召回并回填成功的 chunk 数量")
    included_chunk_count: int = Field(..., ge=0, description="在上下文字符预算内实际写入资料包的 chunk 数量")
    omitted_chunk_count: int = Field(..., ge=0, description="因上下文预算不足而未写入资料包的 chunk 数量")
    top_score: float | None = Field(None, description="本次检索 Top1 分数；用于观察无答案拒答阈值")
    score_threshold: float | None = Field(None, description="本次实际使用的拒答阈值")
    rejected_by_score_threshold: bool = Field(False, description="是否因为 Top1 分数低于阈值触发拒答")
    context_char_count: int = Field(..., ge=0, description="最终资料包字符数，包含来源元数据")
    references: list[RagContextReference] = Field(default_factory=list, description="资料编号与真实来源的映射")
    context: str = Field(..., description="将作为后续模型调用输入的带编号资料包，仅供开发阶段检查")


class RagAnswerRequest(BaseModel):
    """Day21 RAG 问答请求；当前为单轮知识库问答，暂不混入会话历史改写。"""

    question: str = Field(..., min_length=1, max_length=1000, description="用户需要基于知识库回答的问题")
    retrieval_top_k: int = Field(5, ge=1, le=10, description="从 Milvus 召回的候选 chunk 数量")
    max_context_characters: int = Field(4000, ge=1000, le=12000, description="送入模型的资料包最大字符数")
    use_reranker: bool = Field(False, description="是否启用 Reranker 精排")
    rerank_top_n: int = Field(20, ge=1, le=50, description="启用精排时的 Milvus 粗排候选数")
    score_threshold: float | None = Field(
        None,
        ge=-1,
        le=1,
        description="可选拒答阈值；Top1 分数低于该值时直接返回知识库依据不足",
    )


class KnowledgeDomainRequest(BaseModel):
    """一次多文档查询中已经被业务层允许访问的知识域。

    这是 Day31 的开发/教学 DTO：调用方显式提供已经通过权限和数据范围过滤的文档集合，
    方便先观察 LlamaIndex RouterRetriever 的行为。正式接金汤令后应由服务端根据
    tenant/park/enterprise 与角色数据范围查询 domain-document 关系，而不是相信前端传入
    的 ``document_ids``。
    """

    domain_id: str = Field(..., min_length=1, max_length=64, description="业务知识域 ID，例如 safety_compliance")
    description: str = Field(..., min_length=1, max_length=300, description="供 RouterRetriever 识别领域语义的受控说明")
    document_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=30,
        description="该知识域中已通过业务权限过滤的知识文档 ID 集合",
    )
    keywords: list[str] = Field(
        default_factory=list,
        max_length=30,
        description="首版确定性路由关键词；后续可在同一位置替换为评测后 LLM Selector",
    )

    @model_validator(mode="after")
    def validate_document_ids(self) -> "KnowledgeDomainRequest":
        normalized_document_ids = list(dict.fromkeys(document_id.strip() for document_id in self.document_ids if document_id.strip()))
        if not normalized_document_ids:
            raise ValueError("知识域至少包含一个有效 document_id")
        self.document_ids = normalized_document_ids
        self.keywords = list(dict.fromkeys(keyword.strip() for keyword in self.keywords if keyword.strip()))
        return self


class MultiDocumentRagRequest(BaseModel):
    """多文档知识域路由请求，共用预览和正式回答的检索治理参数。"""

    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    domains: list[KnowledgeDomainRequest] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="已授权的知识域；RouterRetriever 只会从这里选择一个领域",
    )
    default_domain_id: str | None = Field(
        None,
        min_length=1,
        max_length=64,
        description="没有关键词命中时的兜底知识域；建议配置为已授权的综合安全知识域",
    )
    retrieval_top_k: int = Field(5, ge=1, le=10, description="选中知识域内跨文档全局召回的 Top-K")
    max_context_characters: int = Field(4000, ge=1000, le=12000, description="实际送入模型资料包的最大字符数")
    use_reranker: bool = Field(False, description="是否对全局召回候选统一执行 Reranker 精排")
    rerank_top_n: int = Field(20, ge=1, le=50, description="启用精排时的全局粗排候选数")
    score_threshold: float | None = Field(None, ge=-1, le=1, description="可选拒答阈值")

    @model_validator(mode="after")
    def validate_route_request(self) -> "MultiDocumentRagRequest":
        domain_ids = [domain.domain_id for domain in self.domains]
        if len(set(domain_ids)) != len(domain_ids):
            raise ValueError("domains 中 domain_id 不能重复")
        if self.default_domain_id is not None and self.default_domain_id not in domain_ids:
            raise ValueError("default_domain_id 必须属于 domains")
        if self.use_reranker and self.rerank_top_n < self.retrieval_top_k:
            raise ValueError("启用 Reranker 时 rerank_top_n 必须大于等于 retrieval_top_k")
        return self


class MultiDocumentRagContextPreviewResponse(BaseModel):
    """跨文档回答前的证据预览；不调用聊天模型。"""

    framework: str = Field("llamaindex", description="负责路由和检索后编排的框架")
    orchestration: str = Field("RouterRetriever + RetrieverQueryEngine", description="本次使用的 LlamaIndex 编排组件")
    retrieval_backend: str = Field("project_milvus", description="实际向量检索与版本治理后端")
    selected_domain_id: str = Field(..., description="RouterRetriever 实际选择的知识域")
    selected_document_ids: list[str] = Field(default_factory=list, description="该领域允许参与检索的文档 ID")
    active_version_by_document_id: dict[str, str] = Field(default_factory=dict, description="本次参与检索的每篇文档 active 版本")
    route_reason: str = Field(..., description="路由理由，例如 deterministic_domain_keyword_match")
    embedding_model: str = Field(..., description="查询向量模型")
    vector_dimension: int = Field(..., ge=1, description="查询向量维度")
    retrieved_chunk_count: int = Field(..., ge=0, description="选中领域跨文档全局召回的 chunk 数")
    included_chunk_count: int = Field(..., ge=0, description="实际进入资料包的 chunk 数")
    omitted_chunk_count: int = Field(..., ge=0, description="因父块去重或预算未进入资料包的 chunk 数")
    top_score: float | None = Field(None, description="全局 Top1 分数")
    score_threshold: float | None = Field(None, description="本次拒答阈值")
    rejected_by_score_threshold: bool = Field(False, description="是否因相关度门禁拒答")
    context_char_count: int = Field(..., ge=0, description="资料包字符数")
    references: list[RagContextReference] = Field(default_factory=list, description="实际资料编号与跨文档来源映射")
    context: str = Field(..., description="仅开发观察：最终将被送入模型的资料包")


class MultiDocumentRagAnswerResponse(BaseModel):
    """多文档受治理 RAG 回答，不伪造唯一的 document_id/version_id。"""

    framework: str = Field("llamaindex", description="负责路由和检索后编排的框架")
    orchestration: str = Field("RouterRetriever + RetrieverQueryEngine", description="本次使用的 LlamaIndex 编排组件")
    retrieval_backend: str = Field("project_milvus", description="实际向量检索与版本治理后端")
    answer: str = Field(..., min_length=1, description="模型回答；事实结论应带 [S1] 等来源引用")
    references: list[RagContextReference] = Field(default_factory=list, description="模型实际引用且已校验的跨文档来源")
    selected_domain_id: str = Field(..., description="RouterRetriever 实际选择的知识域")
    selected_document_ids: list[str] = Field(default_factory=list, description="本次允许检索的文档 ID")
    active_version_by_document_id: dict[str, str] = Field(default_factory=dict, description="本次参与检索的每篇文档 active 版本")
    route_reason: str = Field(..., description="路由理由")
    retrieved_chunk_count: int = Field(..., ge=0, description="跨文档全局召回数")
    included_chunk_count: int = Field(..., ge=0, description="实际送入模型的资料数")
    omitted_chunk_count: int = Field(..., ge=0, description="未进入资料包的 chunk 数")
    top_score: float | None = Field(None, description="全局 Top1 分数")
    score_threshold: float | None = Field(None, description="本次拒答阈值")
    rejected_by_score_threshold: bool = Field(False, description="是否触发相关度拒答")
    prompt_tokens: int | None = Field(None, ge=0, description="模型输入 Token 数")
    completion_tokens: int | None = Field(None, ge=0, description="模型输出 Token 数")
    total_tokens: int | None = Field(None, ge=0, description="模型总 Token 数")
    cost_ms: int = Field(..., ge=0, description="端到端处理耗时，单位毫秒")


class RagAnswerResponse(BaseModel):
    """RAG 基线回答；references 仅包含模型实际引用且已校验存在的资料来源。"""

    answer: str = Field(..., min_length=1, description="模型基于参考资料生成的回答，事实结论应带 [S1] 等引用")
    references: list[RagContextReference] = Field(default_factory=list, description="模型实际引用的可追溯资料")
    document_id: str = Field(..., description="检索的知识库文档业务 ID")
    active_version_id: str = Field(..., description="本次检索使用的 active 文档版本 ID")
    retrieved_chunk_count: int = Field(..., ge=0, description="Milvus 召回并成功回填的 chunk 数量")
    included_chunk_count: int = Field(..., ge=0, description="实际放入模型资料包的 chunk 数量")
    omitted_chunk_count: int = Field(..., ge=0, description="因上下文预算未进入模型资料包的 chunk 数量")
    top_score: float | None = Field(None, description="本次检索 Top1 分数")
    score_threshold: float | None = Field(None, description="本次实际使用的拒答阈值")
    rejected_by_score_threshold: bool = Field(False, description="是否因为 Top1 分数低于阈值触发拒答")
    prompt_tokens: int | None = Field(None, ge=0, description="模型输入 Token 数")
    completion_tokens: int | None = Field(None, ge=0, description="模型输出 Token 数")
    total_tokens: int | None = Field(None, ge=0, description="本次模型调用总 Token 数")
    cost_ms: int = Field(..., ge=0, description="仅模型生成阶段耗时，单位毫秒")


class LlamaIndexRagAnswerResponse(RagAnswerResponse):
    """Day31 QueryEngine 回答的稳定 HTTP 投影。

    保留 RagAnswerResponse 以便基线与候选链路按同一质量、成本和来源字段对比；
    额外字段只说明本次由哪个框架层完成编排，并不代表替换底层 Milvus 治理。
    """

    framework: str = Field("llamaindex", description="负责检索后编排的框架")
    retrieval_backend: str = Field("project_milvus", description="实际向量检索与版本治理后端")
    orchestration: str = Field("RetrieverQueryEngine", description="实际执行的 LlamaIndex 编排组件")


class ContextualIndexTaskSubmitResponse(BaseModel):
    """知识库上下文化索引异步提交结果。"""

    task_id: str = Field(..., description="异步任务业务 ID")
    version_id: str = Field(..., description="待构建上下文化索引的候选版本 ID")
    status: str = Field(..., description="初始任务状态，正常为 pending")


class SessionRagAnswerRequest(BaseModel):
    """会话内 RAG 问答请求；当前阶段不做查询改写，message 直接作为知识库检索问题。"""

    document_id: str = Field(..., min_length=1, description="要检索的知识库文档业务 ID")
    message: str = Field(..., min_length=1, max_length=1000, description="用户本次提问")
    retrieval_top_k: int = Field(5, ge=1, le=10, description="Milvus 召回候选 chunk 数量")
    max_context_characters: int = Field(4000, ge=1000, le=12000, description="送入模型的资料包最大字符数")
    use_reranker: bool = Field(False, description="是否启用 Reranker 精排")
    rerank_top_n: int = Field(20, ge=1, le=50, description="启用精排时的 Milvus 粗排候选数")
    score_threshold: float | None = Field(
        None,
        ge=-1,
        le=1,
        description="可选拒答阈值；Top1 分数低于该值时直接返回知识库依据不足",
    )


class AsyncSessionRagTaskRequest(BaseModel):
    """异步会话 RAG 入参；提交后由通用异步任务查询接口返回最终回答。"""

    document_id: str = Field(..., min_length=1, description="要检索的知识库文档业务 ID")
    message: str = Field(..., min_length=1, max_length=1000, description="用户本次提问")
    retrieval_top_k: int = Field(5, ge=1, le=10, description="Milvus 召回候选 chunk 数量")
    max_context_characters: int = Field(4000, ge=1000, le=12000, description="送入模型的资料包最大字符数")
    use_reranker: bool = Field(False, description="是否启用 Reranker 精排")
    rerank_top_n: int = Field(20, ge=1, le=50, description="启用精排时的 Milvus 粗排候选数")
    score_threshold: float | None = Field(
        None,
        ge=-1,
        le=1,
        description="可选拒答阈值；Top1 分数低于该值时直接返回知识库依据不足",
    )


class AsyncRagTaskSubmitResponse(BaseModel):
    """异步 RAG 提交结果；task_id 是前端与 Java 轮询的业务任务 ID。"""

    task_id: str = Field(..., description="异步任务业务 ID")
    status: str = Field(..., description="初始状态，正常为 pending")


class SessionRagAnswerResponse(RagAnswerResponse):
    """会话 RAG 回答，额外返回消息 ID 供前端渲染并回查引用。"""

    session_id: str = Field(..., description="所属会话业务 ID")
    user_message_id: str = Field(..., description="本次已持久化的用户消息 ID")
    assistant_message_id: str = Field(..., description="本次已持久化的 RAG 回答消息 ID")


class RagAnswerReferenceItem(RagContextReference):
    """持久化后的 RAG 回答引用记录。"""

    reference_id: str = Field(..., description="引用记录业务 ID")
    assistant_message_id: str = Field(..., description="产生本条引用的 assistant 消息 ID")
    created_at: str = Field(..., description="引用记录创建时间")


class RagAnswerReferenceListResponse(BaseModel):
    """查询某条会话 RAG 回答引用的响应。"""

    session_id: str = Field(..., description="所属会话业务 ID")
    assistant_message_id: str = Field(..., description="RAG 回答消息 ID")
    items: list[RagAnswerReferenceItem] = Field(default_factory=list, description="该回答实际使用的资料来源")


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
    """切换当前检索版本的人工审计信息。"""

    activated_by: str | None = Field(
        None,
        max_length=64,
        description="已废弃兼容字段；服务端始终使用认证 Principal，客户端值不会生效",
    )
    activation_note: str = Field(..., min_length=1, max_length=500, description="确认向量数量和检索质量后的切换说明")


class ActivateDocumentVersionResponse(BaseModel):
    """版本切换结果；Milvus 不复制数据，查询入口只改 MySQL active 指针。"""

    document_id: str = Field(..., description="所属文档业务 ID")
    active_version_id: str = Field(..., description="切换后当前可服务的文档版本 ID")
    previous_version_id: str | None = Field(None, description="切换前的 active 版本 ID，首次上线时为空")
    status: str = Field(..., description="新版本状态，成功时为 active")
    activated_at: str = Field(..., description="切换完成时间")


class DocumentChunkRequest(BaseModel):
    """一次语义切块任务的参数；后端会把最终取值保存为版本快照。"""

    max_characters: int = Field(500, ge=100, le=4000, description="单个 chunk 的推荐字符预算")
    overlap_characters: int = Field(80, ge=0, le=1000, description="仅超长语义单元硬切时保留的重叠字符数")
    boundary_search_characters: int = Field(120, ge=1, le=1000, description="超长句子硬切时向前寻找断句点的窗口")
    min_chunk_characters: int = Field(120, ge=1, le=4000, description="避免产生过短尾块的最小推荐字符数")
    semantic_overflow_characters: int = Field(80, ge=0, le=1000, description="为保持语义完整允许超过推荐预算的字符数")

    @model_validator(mode="after")
    def validate_overlap(self) -> "DocumentChunkRequest":
        # Pydantic 在进入业务服务前完成跨字段校验，类似 Java DTO 上的自定义校验器。
        if self.overlap_characters >= self.max_characters:
            raise ValueError("overlap_characters 必须小于 max_characters")
        if self.min_chunk_characters > self.max_characters:
            raise ValueError("min_chunk_characters 必须不大于 max_characters")
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
