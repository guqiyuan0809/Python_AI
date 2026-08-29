"""
数据库表模型

类似 Java 项目中的 Entity。
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from day04_app.database import Base


class ChatSession(Base):
    __tablename__ = "chat_session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )


class ChatMessage(Base):
    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    # 会话轮次：一条 user 输入和对应的最终 assistant 输出共享同一个 turn_no。
    # Agent Loop 内部的 step 不写在这里，避免把十步工具推理误算成十轮对话。
    turn_no: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stream_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="success")
    error_type: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AiRagAnswerReference(Base):
    """RAG 回答实际引用的知识来源快照，独立于聊天正文以支持审计和前端回溯。"""

    __tablename__ = "ai_rag_answer_reference"
    __table_args__ = (
        UniqueConstraint("reference_id", name="uk_airar_ref_id"),
        UniqueConstraint("assistant_message_id", "source_id", name="uk_airar_msg_source"),
        Index("ix_airar_session", "session_id"),
        Index("ix_airar_assistant", "assistant_message_id"),
        Index("ix_airar_doc_ver", "document_id", "version_id"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    reference_id: Mapped[str] = mapped_column(
        String(64),
        comment="RAG 回答引用记录业务唯一 ID",
    )
    session_id: Mapped[str] = mapped_column(
        String(64),
        comment="所属会话业务 ID",
    )
    assistant_message_id: Mapped[str] = mapped_column(
        String(64),
        comment="产生该引用的 assistant 消息业务 ID",
    )
    source_id: Mapped[str] = mapped_column(
        String(16),
        comment="模型回答中的资料编号，例如 S1",
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        comment="引用知识库文档业务 ID",
    )
    version_id: Mapped[str] = mapped_column(
        String(64),
        comment="引用时生效的知识库文档版本 ID",
    )
    chunk_id: Mapped[str] = mapped_column(
        String(64),
        comment="引用的知识库检索块业务 ID",
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        comment="引用 chunk 在文档版本内的顺序",
    )
    score: Mapped[float] = mapped_column(
        Float,
        comment="本次检索时 Milvus 返回的相似度分数快照",
    )
    locations_json: Mapped[str] = mapped_column(
        Text,
        comment="引用来源位置快照 JSON，例如段落或页码",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="引用记录创建时间",
    )


class KnowledgeDocument(Base):
    """知识库文档主表，管理原始文件元数据和解析生命周期。"""

    __tablename__ = "knowledge_document"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    # 对外唯一业务 ID；Java、前端、异步解析任务都只能使用它定位文档。
    document_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        comment="知识库文档业务唯一 ID",
    )
    # 当前对检索请求生效的版本 ID；候选版本构建期间不修改它，避免服务短暂读不到旧知识。
    active_version_id: Mapped[str | None] = mapped_column(
        String(64),
        index=True,
        nullable=True,
        comment="当前生效的文档版本业务 ID，切换完成后才更新",
    )
    original_file_name: Mapped[str] = mapped_column(
        String(255),
        comment="用户上传时的原始文件名，仅用于展示和审计",
    )
    file_type: Mapped[str] = mapped_column(
        String(16),
        index=True,
        comment="已校验的文件类型，例如 docx/pdf/xlsx",
    )
    # 保存相对存储键而非绝对路径，使开发、测试、生产环境的磁盘根目录可以不同。
    storage_key: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        comment="服务端原始文件存储键，不向调用方暴露物理路径",
    )
    file_size: Mapped[int] = mapped_column(
        Integer,
        comment="原始文件大小，单位字节",
    )
    content_sha256: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="原始文件内容 SHA-256，用于完整性校验和重复文件识别",
    )
    trace_id: Mapped[str | None] = mapped_column(
        String(64),
        index=True,
        nullable=True,
        comment="上传请求链路追踪 ID",
    )
    # uploaded/parsing/parsed/error；后续异步解析任务据此更新状态。
    status: Mapped[str] = mapped_column(
        String(32),
        default="uploaded",
        index=True,
        comment="文档生命周期状态：uploaded/parsing/parsed/error",
    )
    parser_name: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="最近一次成功执行的解析器名称",
    )
    parsed_segment_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="最近一次成功解析得到的有效文本段数量",
    )
    # not_started/chunking/chunked/error；独立于解析状态，避免“已解析”被切块过程覆盖。
    chunk_status: Mapped[str] = mapped_column(
        String(32),
        default="not_started",
        index=True,
        comment="切块生命周期状态：not_started/chunking/chunked/error",
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="最近一次成功切块得到的检索块数量",
    )
    chunk_config_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="最近一次成功切块使用的参数快照 JSON",
    )
    chunk_error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="切块失败时记录的错误原因",
    )
    chunked_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="最近一次成功完成切块的时间",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="解析失败时记录的错误原因",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="文档上传记录创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
        comment="文档记录最后更新时间",
    )


class KnowledgeDocumentVersion(Base):
    """知识库文档版本主表，支持候选索引构建、验证和无感切换。"""

    __tablename__ = "knowledge_document_version"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    version_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        comment="文档版本业务唯一 ID",
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="所属知识库文档业务 ID",
    )
    version_number: Mapped[int] = mapped_column(
        Integer,
        comment="同一文档内递增版本号，从 1 开始",
    )
    # uploaded/parsing/parsed/chunking/chunked/indexing/indexed/active/retired/error。
    status: Mapped[str] = mapped_column(
        String(32),
        index=True,
        comment="版本索引生命周期状态，active 表示当前可被检索",
    )
    source_sha256: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="该版本原始文件内容 SHA-256",
    )
    rebuild_note: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="创建候选索引版本的变更说明，例如调整切块参数或更新原文件",
    )
    parser_name: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="该版本解析时使用的解析器名称",
    )
    segment_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="该版本解析出的原始文本段数量",
    )
    chunk_config_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="该版本切块参数快照 JSON",
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="该版本生成的检索块数量",
    )
    embedding_model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="该版本向量化使用的 Embedding 模型",
    )
    embedding_dimension: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="该版本 Embedding 向量维度",
    )
    vector_collection: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="该版本向量写入的 Milvus Collection 名称",
    )
    vector_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="该版本已成功写入向量库的 Entity 数量",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="版本构建失败时记录的错误原因",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="候选版本创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
        comment="版本构建状态最后更新时间",
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="版本切换为 active 的时间",
    )


class KnowledgeDocumentVersionActivationAudit(Base):
    """文档索引版本发布审计，记录每次 active 指针切换的事实。"""

    __tablename__ = "knowledge_document_version_activation_audit"
    # MySQL 索引名上限为 64 字符；审计表名较长，因此使用稳定短名，不能依赖 SQLAlchemy 默认命名。
    __table_args__ = (
        Index("ix_kdvaa_doc", "document_id"),
        Index("ix_kdvaa_active_ver", "activated_version_id"),
        Index("ix_kdvaa_prev_ver", "previous_version_id"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    activation_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        comment="文档版本切换审计业务唯一 ID",
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        comment="所属知识库文档业务 ID",
    )
    activated_version_id: Mapped[str] = mapped_column(
        String(64),
        comment="本次切换为 active 的文档版本 ID",
    )
    previous_version_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="切换前 active 的文档版本 ID，首次发布时为空",
    )
    activated_by: Mapped[str] = mapped_column(
        String(64),
        comment="执行切换的人员标识，接入认证后取自登录上下文",
    )
    activation_note: Mapped[str] = mapped_column(
        Text,
        comment="人工确认向量数量和检索质量后的切换说明",
    )
    activated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="版本切换完成时间",
    )


class KnowledgeDocumentSegment(Base):
    """文档解析后的原始文本段；Day19 会在此基础上生成检索切块。"""

    __tablename__ = "knowledge_document_segment"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="所属知识库文档业务 ID",
    )
    # 同一份文档内的稳定顺序，后续切块、引用来源和重解析对比都依赖它。
    segment_index: Mapped[int] = mapped_column(
        Integer,
        comment="文本段在原文档中的从 0 开始顺序",
    )
    content: Mapped[str] = mapped_column(
        Text,
        comment="解析得到的原始文本内容，尚未经过 Day19 检索切块",
    )
    location: Mapped[str] = mapped_column(
        String(255),
        comment="可追溯的原文位置，例如 Paragraph:12 或 Table:2/Row:4",
    )
    metadata_json: Mapped[str] = mapped_column(
        Text,
        comment="解析器输出的来源补充元数据 JSON",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="文本段持久化时间",
    )


class KnowledgeDocumentParentChunk(Base):
    """父块保存完整业务上下文；只回填给回答模型，不写入向量库参与粗排。"""

    __tablename__ = "knowledge_document_parent_chunk"
    __table_args__ = (
        UniqueConstraint("parent_chunk_id", name="uk_kdpc_parent_chunk_id"),
        UniqueConstraint("version_id", "parent_index", name="uk_kdpc_ver_parent_idx"),
        Index("ix_kdpc_document_version", "document_id", "version_id"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    parent_chunk_id: Mapped[str] = mapped_column(
        String(64),
        comment="父块业务唯一 ID，被子块 parent_chunk_id 引用",
    )
    version_id: Mapped[str] = mapped_column(
        String(64),
        comment="所属文档索引版本业务 ID",
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        comment="所属知识库文档业务 ID",
    )
    parent_index: Mapped[int] = mapped_column(
        Integer,
        comment="父块在文档版本中的从 0 开始顺序",
    )
    content: Mapped[str] = mapped_column(
        Text,
        comment="完整父级原文，只能来自解析文档，不能使用模型生成内容替代",
    )
    char_count: Mapped[int] = mapped_column(
        Integer,
        comment="父块原文字符数，用于回答上下文预算控制",
    )
    source_references_json: Mapped[str] = mapped_column(
        Text,
        comment="父块覆盖的原始文档段来源 JSON",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="父块创建时间",
    )


class KnowledgeDocumentChunk(Base):
    """面向 Embedding 与检索的文档文本切块。"""

    __tablename__ = "knowledge_document_chunk"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    # MySQL 与 Milvus 共用的跨库关联键；不能把数据库自增 id 当成对外业务 ID。
    chunk_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        comment="检索块业务唯一 ID，同时作为 Milvus Entity 主键",
    )
    version_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="所属文档索引版本业务 ID，用于新旧版本并存和向量检索过滤",
    )
    parent_chunk_id: Mapped[str | None] = mapped_column(
        String(64),
        index=True,
        nullable=True,
        comment="父子切块模式下所属父块业务 ID；历史普通切块为空",
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="所属知识库文档业务 ID",
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        comment="检索块在文档内的从 0 开始顺序",
    )
    content: Mapped[str] = mapped_column(
        Text,
        comment="子块或历史普通块的真实原文；回答、引用和审计只能以此为事实依据",
    )
    contextual_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="模型为子块生成的检索背景说明，仅用于提高召回，不能作为事实引用",
    )
    embedding_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="上下文化说明与真实原文拼接后的向量化输入；历史普通块为空时回退 content",
    )
    char_count: Mapped[int] = mapped_column(
        Integer,
        comment="切块文本字符数，用于控制上下文与成本",
    )
    source_references_json: Mapped[str] = mapped_column(
        Text,
        comment="切块覆盖的原始文档段来源 JSON",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="检索块持久化时间",
    )


class KnowledgeRetrievalEvalDataset(Base):
    """RAG 检索评测数据集：同一数据集固定对应一份文档和一组人工标注问题。"""

    __tablename__ = "knowledge_retrieval_eval_dataset"
    __table_args__ = (
        UniqueConstraint("dataset_id", name="uk_kred_dataset_id"),
        Index("ix_kred_doc", "document_id"),
        Index("ix_kred_name_ver", "dataset_name", "dataset_version"),
        Index("ix_kred_status", "status"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    dataset_id: Mapped[str] = mapped_column(
        String(64),
        comment="检索评测数据集业务唯一 ID",
    )
    dataset_name: Mapped[str] = mapped_column(
        String(64),
        comment="数据集名称，例如 jvm_knowledge_retrieval",
    )
    dataset_version: Mapped[str] = mapped_column(
        String(64),
        comment="数据集版本，例如 v1；标注调整应新建版本保留历史可复现性",
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        comment="评测范围内的知识库文档业务 ID；当前检索接口按单文档工作",
    )
    description: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="数据集用途和覆盖范围说明",
    )
    sample_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="当前数据集已标注并可参与评测的样本数",
    )
    # draft 用于编辑样本，active 用于固定版本评测，archived 用于保留历史而停止使用。
    status: Mapped[str] = mapped_column(
        String(32),
        default="draft",
        comment="数据集状态：draft、active、archived",
    )
    created_by: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="创建人标识；接入认证后由登录用户提供",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="数据集创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
        comment="数据集最后修改时间",
    )


class KnowledgeRetrievalEvalSample(Base):
    """一条 RAG 检索标注：以稳定原始段序号而非易变化的 chunk_id 作为标准答案。"""

    __tablename__ = "knowledge_retrieval_eval_sample"
    __table_args__ = (
        UniqueConstraint("sample_id", name="uk_kres_sample_id"),
        Index("ix_kres_dataset", "dataset_id"),
        Index("ix_kres_type", "sample_type"),
        Index("ix_kres_status", "status"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    sample_id: Mapped[str] = mapped_column(
        String(64),
        comment="检索评测样本业务唯一 ID",
    )
    dataset_id: Mapped[str] = mapped_column(
        String(64),
        comment="所属检索评测数据集业务 ID",
    )
    question: Mapped[str] = mapped_column(
        Text,
        comment="待检索的用户问题",
    )
    # normal/boundary/no_answer 描述覆盖场景，不表示问题是否答对。
    sample_type: Mapped[str] = mapped_column(
        String(32),
        default="normal",
        comment="样本类型：normal、boundary、no_answer",
    )
    expected_answerable: Mapped[int] = mapped_column(
        Integer,
        default=1,
        comment="是否期望知识库可回答：1 是，0 否；no_answer 样本必须为 0",
    )
    # segment_index 是解析后原文的稳定定位锚点；chunk 重切后 ID 会改变，不能作为评测标准。
    expected_segment_indexes_json: Mapped[str] = mapped_column(
        Text,
        comment="人工标注的期望原文段序号 JSON 数组，例如 [36, 37]",
    )
    expected_note: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="人工标注理由或期望命中依据说明",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default="active",
        comment="样本状态：active 或 archived",
    )
    created_by: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="标注人标识；接入认证后由登录用户提供",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="样本创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
        comment="样本最后修改时间",
    )


class KnowledgeRetrievalEvalRun(Base):
    """一次固定数据集、固定文档版本和固定 Top-K 的检索评测运行记录。"""

    __tablename__ = "knowledge_retrieval_eval_run"
    __table_args__ = (
        UniqueConstraint("run_id", name="uk_krer_run_id"),
        Index("ix_krer_dataset", "dataset_id"),
        Index("ix_krer_doc_ver", "document_id", "document_version_id"),
        Index("ix_krer_status", "status"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    run_id: Mapped[str] = mapped_column(
        String(64),
        comment="检索评测运行业务唯一 ID",
    )
    dataset_id: Mapped[str] = mapped_column(
        String(64),
        comment="本次使用的检索评测数据集业务 ID 快照",
    )
    document_id: Mapped[str] = mapped_column(
        String(64),
        comment="本次评测的知识库文档业务 ID",
    )
    document_version_id: Mapped[str] = mapped_column(
        String(64),
        comment="本次被测的文档索引版本业务 ID，可为 indexed 或 active 版本",
    )
    retrieval_top_k: Mapped[int] = mapped_column(
        Integer,
        comment="本次检索统一使用的 Top-K，指标只可在相同 K 下横向比较",
    )
    use_reranker: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="是否启用 Reranker 精排：1 是，0 否",
    )
    rerank_top_n: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="启用精排时 Milvus 粗排候选数量；必须大于等于 retrieval_top_k",
    )
    reranker_model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="启用精排时实际使用的 Reranker 模型",
    )
    score_threshold: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="可选的可回答分数阈值；仅用于计算无答案样本的误放行率",
    )
    embedding_model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="本次实际调用的查询 Embedding 模型",
    )
    vector_dimension: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="本次查询向量维度，用于核对模型与索引契约",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default="running",
        comment="运行状态：running、success、partial_success、error",
    )
    sample_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="本次快照参与评测的 active 样本总数",
    )
    success_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="成功完成检索并产生明细的样本数",
    )
    error_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="检索异常的样本数；单条异常不阻断其余样本评测",
    )
    answerable_sample_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="成功执行的可回答样本数，是 Recall@K 与 MRR 的分母",
    )
    answerable_hit_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="Top-K 至少命中一个期望原始段的可回答样本数",
    )
    total_expected_segment_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="可回答样本中人工标注的期望原始段总数，用于计算真正的 Recall@K",
    )
    total_hit_segment_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="Top-K 实际命中的期望原始段去重总数，用于计算真正的 Recall@K",
    )
    total_retrieved_chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="可回答样本成功检索返回的 chunk 总数，用于计算 Precision@K",
    )
    total_relevant_retrieved_chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="可回答样本 Top-K 中包含期望原始段的 chunk 总数，用于计算 Precision@K",
    )
    hit_at_k: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Hit@K：可回答样本中 Top-K 至少命中一个正确证据的样本比例",
    )
    recall_at_k: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Recall@K：Top-K 命中的期望原始段数 / 期望原始段总数",
    )
    precision_at_k: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Precision@K：Top-K 中正确 chunk 数 / 返回 chunk 总数",
    )
    mrr_at_k: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="可回答样本 MRR@K：首个正确证据排名倒数的平均值",
    )
    no_answer_sample_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="成功执行的无答案样本数",
    )
    no_answer_false_positive_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="启用分数阈值时，被错误判为可回答的无答案样本数",
    )
    no_answer_false_positive_rate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="启用分数阈值时的无答案误放行率",
    )
    no_answer_avg_top_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="无答案样本 Top-1 相似度均值，用于后续选择拒答阈值",
    )
    elapsed_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="本次评测总耗时，单位毫秒",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="运行级异常信息；单样本异常详见评测明细",
    )
    created_by: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="发起评测的人员标识",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="评测开始时间",
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="评测结束时间",
    )


class KnowledgeRetrievalEvalCaseResult(Base):
    """一次运行中单条样本的命中明细，保存快照以支持历史复盘。"""

    __tablename__ = "knowledge_retrieval_eval_case_result"
    __table_args__ = (
        UniqueConstraint("case_result_id", name="uk_krecr_case_id"),
        UniqueConstraint("run_id", "sample_id", name="uk_krecr_run_sample"),
        Index("ix_krecr_run", "run_id"),
        Index("ix_krecr_sample", "sample_id"),
        Index("ix_krecr_status", "status"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    case_result_id: Mapped[str] = mapped_column(
        String(64),
        comment="评测样本结果业务唯一 ID",
    )
    run_id: Mapped[str] = mapped_column(
        String(64),
        comment="所属检索评测运行业务 ID",
    )
    sample_id: Mapped[str] = mapped_column(
        String(64),
        comment="关联的评测样本业务 ID",
    )
    question_snapshot: Mapped[str] = mapped_column(
        Text,
        comment="运行时的问题快照，样本后续修改不影响历史复盘",
    )
    sample_type_snapshot: Mapped[str] = mapped_column(
        String(32),
        comment="运行时样本类型快照：normal、boundary、no_answer",
    )
    expected_answerable_snapshot: Mapped[int] = mapped_column(
        Integer,
        comment="运行时期望是否可回答快照：1 是，0 否",
    )
    expected_segment_indexes_json: Mapped[str] = mapped_column(
        Text,
        comment="运行时期望原始段序号 JSON 快照",
    )
    retrieved_segment_indexes_json: Mapped[str] = mapped_column(
        Text,
        comment="按 Milvus 排名展开去重后的实际召回原始段序号 JSON",
    )
    retrieved_chunks_json: Mapped[str] = mapped_column(
        Text,
        comment="实际召回 chunk 的 ID、排名、分数和来源段快照 JSON",
    )
    first_hit_rank: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="首个命中期望原始段的 chunk 排名，从 1 开始；未命中为空",
    )
    is_hit: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="可回答样本是否命中正确依据：1 是，0 否；无答案样本为空",
    )
    hit_segment_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="本样本 Top-K 命中的期望原始段去重数量；无答案样本为空",
    )
    expected_segment_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="本样本人工标注的期望原始段数量；无答案样本为空",
    )
    relevant_retrieved_chunk_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="本样本 Top-K 中包含期望原始段的 chunk 数量；无答案样本为空",
    )
    precision_at_k: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="本样本 Precision@K：正确 chunk 数 / 实际返回 chunk 数；无答案样本为空",
    )
    top_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="本条样本 Top-1 的 Milvus 相似度分数",
    )
    is_false_positive: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="无答案样本在本次阈值下是否被误判为可回答：1 是，0 否",
    )
    elapsed_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="本条样本从查询向量化到召回回填的耗时，单位毫秒",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default="success",
        comment="明细状态：success 或 error",
    )
    error_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="检索失败时的异常类型",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="检索失败时的异常信息",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="评测明细创建时间",
    )


class ChatSessionSummary(Base):
    __tablename__ = "chat_session_summary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 摘要记录的业务唯一 ID，接口返回时优先使用它，而不是数据库自增 id。
    summary_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # 这条摘要属于哪个会话，用于关联 chat_session.session_id。
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    # 压缩后的会话摘要内容，会作为后续模型调用的长期上下文。
    summary: Mapped[str] = mapped_column(Text)
    # 表示这份摘要已经覆盖到了哪一条消息，后续只需要再携带这条消息之后的新消息。
    summary_until_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Day32：按会话轮次记录摘要边界；旧记录没有该字段时仍由 message_id 兼容读取。
    summary_until_turn_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_turn_count: Mapped[int] = mapped_column(Integer, default=0)
    source_token_count: Mapped[int] = mapped_column(Integer, default=0)
    # 同一个会话下的摘要版本号，越大表示越新的摘要。
    version: Mapped[int] = mapped_column(Integer, default=1)
    # 生成摘要时使用的模型，便于后续排查摘要质量和成本。
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 摘要生成状态，当前主要使用 success，后续可扩展 pending/error。
    status: Mapped[str] = mapped_column(String(32), default="success")
    # 摘要失败时记录错误信息，方便排查，不直接返回给普通用户。
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ChatSessionTurn(Base):
    """会话轮次事实表。

    一次普通聊天或一次 Agent Loop 对用户来说都是一轮；Agent Loop 内部的每个
    ``step`` 由 run_id/ai_call_log 记录，不能与会话轮次混用。
    """

    __tablename__ = "chat_session_turn"
    __table_args__ = (
        UniqueConstraint("session_id", "turn_no", name="uk_chat_turn_session_no"),
        Index("ix_chat_turn_session", "session_id", "created_at"),
        Index("ix_chat_turn_task", "task_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    turn_no: Mapped[int] = mapped_column(Integer)
    user_message_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    assistant_message_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 异步聊天、RAG、结构化分析会把同一业务 task_id 固定在一轮会话上。Worker 必须
    # 根据它回填对应轮次，不能用“最新 pending 轮次”猜测，否则同一会话并发提交时会串答。
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_run_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class AgentWorkingMemorySnapshot(Base):
    """Agent Loop 内部步骤压缩快照，只用于当前 run 的继续决策，不进入长期语义记忆。"""

    __tablename__ = "agent_working_memory_snapshot"
    __table_args__ = (
        Index("ix_awms_run", "run_id", "created_at"),
        Index("ix_awms_session", "session_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    covered_step_from: Mapped[int] = mapped_column(Integer)
    covered_step_to: Mapped[int] = mapped_column(Integer)
    retained_step_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state_json: Mapped[str] = mapped_column(Text)
    summary_text: Mapped[str] = mapped_column(Text)
    estimated_tokens: Mapped[int] = mapped_column(Integer, default=0)
    snapshot_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class SessionMemory(Base):
    """允许长期语义检索的记忆事实源；Milvus 只保存该表的向量索引。"""

    __tablename__ = "session_memory"
    __table_args__ = (
        Index("ix_session_memory_scope", "session_id", "user_id", "tenant_id", "status"),
        Index("ix_session_memory_embedding", "embedding_status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    tenant_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    memory_type: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    source_summary_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    source_message_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_run_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    embedding_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    embedding_error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class AiCallLog(Base):
    __tablename__ = "ai_call_log"
    __table_args__ = (
        # Day26 的排查入口以 trace/task/run 为主，使用短索引名避免 MySQL 64 字符限制。
        Index("ix_acl_task", "task_id"),
        Index("ix_acl_run", "run_id"),
        Index("ix_acl_stage", "stage"),
        Index("ix_acl_prompt_name", "prompt_name"),
        Index("ix_acl_prompt_version", "prompt_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 调用日志业务 ID，使用雪花 ID，适合后续跨服务追踪和对外查询。
    call_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # 一次请求链路 ID，Java 和 Python 之间通过 X-Trace-Id 透传。
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 会话 ID，便于从某个会话反查本次模型调用。
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 本次调用关联的 assistant 消息 ID，便于从日志定位到最终展示给用户的回答。
    message_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 异步任务 ID；同步请求为空，便于从任务反查具体 AI 阶段。
    task_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="关联 ai_async_task 的业务任务 ID"
    )
    # 评测运行 ID；普通在线请求为空，便于把 Harness 汇总日志关联到运行报告。
    run_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="关联评测或编排运行的业务 ID"
    )
    # 调用来源，例如 session_chat、session_stream_chat、summary、title。
    call_type: Mapped[str] = mapped_column(String(64), index=True)
    # 同一来源内的可观测阶段，例如 agent_model_decision、rag_rerank、rag_answer_generation。
    stage: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="调用来源内的可观测阶段名称"
    )
    prompt_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="Prompt Registry 业务 ID；代码托管 Prompt 为空"
    )
    prompt_name: Mapped[str | None] = mapped_column(
        String(128), nullable=True, comment="本次模型调用使用的 Prompt 名称"
    )
    prompt_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="本次模型调用实际使用的 Prompt 版本"
    )
    prompt_template_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="Prompt 稳定模板 SHA-256，不包含业务输入"
    )
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 调用耗时，后续可以用于慢调用分析和模型性能监控。
    cost_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="success")
    error_type: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 仅记录计数、状态、参数名等脱敏事实，不能保存用户原文、模型完整回答或工具参数值。
    detail_json: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="可观测事件脱敏详情 JSON"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AiAsyncTask(Base):
    __tablename__ = "ai_async_task"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 异步任务业务 ID，前端或 Java 后端后续用它查询任务状态。
    task_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # 请求链路 ID，用于把提交任务和后台执行日志串起来。
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 任务所属会话，当前先支持会话聊天异步执行。
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    # 任务最终关联的 assistant 消息 ID，任务完成后可定位到聊天记录。
    message_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # Celery 投递到 Broker 后产生的内部消息 ID，用于调度侧排查，不替代业务 task_id。
    broker_task_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 任务类型，例如 session_chat，后续可以扩展 summary、rag_import、report_generate。
    task_type: Mapped[str] = mapped_column(String(64), index=True)
    # 用户本次提交的问题，方便任务排查和失败重试。
    input_text: Mapped[str] = mapped_column(Text)
    # AI 最终输出内容，任务成功后写入。
    result_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # pending/running/success/error，用于前端轮询展示任务进度。
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # 自动重试次数和最大重试次数，避免模型异常时无限消耗资源。
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    error_type: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )


class AiTaskOutbox(Base):
    """任务投递事件表，用于解决 MySQL 与消息队列的双写一致性问题。"""

    __tablename__ = "ai_task_outbox"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 事件 ID 独立于 task_id，一次任务可以因自动重试产生多条投递事件。
    event_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    # JSON 字符串，保存 Worker 执行任务所需的最小参数。
    payload: Mapped[str] = mapped_column(Text)
    # pending/published，发布失败后保持 pending，由 Beat 定时补偿。
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    publish_retry_count: Mapped[int] = mapped_column(Integer, default=0)
    # 自动重试时延迟到指定时间后再投递，避免故障期间高频空转。
    available_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )


class AiStructuredResult(Base):
    """AI 结构化结果表，用于持久化模型生成的标准 JSON 业务结果。"""

    __tablename__ = "ai_structured_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 结构化结果业务 ID，前端/Java 可以通过它定位一份标准 JSON 结果。
    result_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # 异步任务 ID；同步测试接口可以为空，后续接入 Worker 后会关联 ai_async_task.task_id。
    task_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    message_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 业务归属，例如 work_order、contract、jingtangling_audit。
    business_type: Mapped[str] = mapped_column(String(64), index=True)
    business_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    # 结构化结果类型和版本，方便后续 prompt/DTO 升级时做兼容。
    schema_type: Mapped[str] = mapped_column(String(64), index=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="v1")
    # 保存标准 JSON 字符串；查询接口再反序列化成前端可直接消费的对象。
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="success", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )


class AiFailureSample(Base):
    """AI 失败样本表，用于沉淀 prompt 优化和 harness 评测数据。"""

    __tablename__ = "ai_failure_sample"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 失败样本业务 ID，后续可以进入评测数据集或管理后台。
    sample_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    message_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    call_type: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    schema_type: Mapped[str] = mapped_column(String(64), index=True)
    schema_version: Mapped[str] = mapped_column(String(32), index=True)
    error_type: Mapped[str] = mapped_column(String(64), index=True)
    error_message: Mapped[str] = mapped_column(Text)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AiPromptVersion(Base):
    """AI Prompt 版本表，用于保存每个业务场景下的 prompt 内容和模型参数。"""

    __tablename__ = "ai_prompt_version"
    __table_args__ = (
        UniqueConstraint("prompt_name", "prompt_version", name="uk_aipv_name_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    prompt_name: Mapped[str] = mapped_column(String(64), index=True)
    prompt_version: Mapped[str] = mapped_column(String(32), index=True)
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    system_prompt: Mapped[str] = mapped_column(Text)
    user_prompt_template: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    created_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )


class AiEvalDataset(Base):
    """AI 评测数据集表，用于保存某个业务评测集合的版本信息。"""

    __tablename__ = "ai_eval_dataset"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    dataset_name: Mapped[str] = mapped_column(String(64), index=True)
    dataset_version: Mapped[str] = mapped_column(String(64), index=True)
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )


class AiEvalSample(Base):
    """AI 评测样本表，用于保存人工标注后的标准输入和期望输出。"""

    __tablename__ = "ai_eval_sample"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sample_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    dataset_id: Mapped[str] = mapped_column(String(64), index=True)
    dataset_version: Mapped[str] = mapped_column(String(64), index=True)
    sample_type: Mapped[str] = mapped_column(String(32), default="normal", index=True)
    input_text: Mapped[str] = mapped_column(Text)
    expected_json: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(32), default="manual", index=True)
    source_ref_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )


class AiEvalRun(Base):
    """AI 评测运行表，用于记录一次 prompt harness 的汇总结果。"""

    __tablename__ = "ai_eval_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 一次评测运行的业务 ID，类似 Java 里给批处理任务生成的 runId。
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    prompt_name: Mapped[str] = mapped_column(String(64), index=True)
    prompt_version: Mapped[str] = mapped_column(String(32), index=True)
    dataset_version: Mapped[str] = mapped_column(String(64), index=True)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    schema_valid_rate: Mapped[float] = mapped_column(Float, default=0)
    category_accuracy: Mapped[float] = mapped_column(Float, default=0)
    risk_level_accuracy: Mapped[float] = mapped_column(Float, default=0)
    human_review_accuracy: Mapped[float] = mapped_column(Float, default=0)
    avg_total_tokens: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_cost_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 保存完整指标 JSON，后续增加新指标时不需要立刻改表。
    metrics_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AiEvalCaseResult(Base):
    """AI 评测样本结果表，用于记录一次评测中每条样本的 actual/expected 对比。"""

    __tablename__ = "ai_eval_case_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    sample_id: Mapped[str] = mapped_column(String(64), index=True)
    schema_valid: Mapped[int] = mapped_column(Integer, default=0)
    category_match: Mapped[int] = mapped_column(Integer, default=0)
    risk_level_match: Mapped[int] = mapped_column(Integer, default=0)
    human_review_match: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    actual_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 保存整行结果，方便后续扩展更多评测字段，不破坏历史数据。
    row_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AiEvalGateDecision(Base):
    """评测准入门禁表，保存候选 Prompt 与基线 Prompt 的比较结论。"""

    __tablename__ = "ai_eval_gate_decision"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    gate_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        comment="评测门禁业务唯一 ID",
    )
    baseline_run_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="已上线基线 Prompt 的评测运行 ID",
    )
    candidate_run_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="候选 Prompt 的评测运行 ID",
    )
    prompt_name: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="业务 Prompt 名称，例如 work_order_analysis",
    )
    dataset_version: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="本次比较使用的评测数据集版本",
    )
    # pass/reject/manual_review；它只表达评测建议，不直接修改 Prompt 的发布状态。
    decision: Mapped[str] = mapped_column(
        String(32),
        index=True,
        comment="门禁结论：pass、reject 或 manual_review",
    )
    # 保存当时指标差异、命中规则和规则版本，确保以后规则升级仍可解释历史结论。
    comparison_json: Mapped[str] = mapped_column(Text, comment="基线与候选评测指标差异 JSON")
    reason_json: Mapped[str] = mapped_column(Text, comment="命中门禁规则与判定原因 JSON")
    rule_snapshot_json: Mapped[str] = mapped_column(Text, comment="本次门禁使用的规则快照 JSON")
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="门禁判定创建时间",
    )


class AiAgentEvalRun(Base):
    """Agent Loop Harness 单次运行汇总，不复用工单评测的专用指标字段。"""

    __tablename__ = "ai_agent_eval_run"
    __table_args__ = (
        UniqueConstraint("run_id", name="uk_aaer_run_id"),
        Index("ix_aaer_agent_ver", "agent_name", "agent_version"),
        Index("ix_aaer_dataset", "dataset_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="数据库自增主键")
    run_id: Mapped[str] = mapped_column(String(64), comment="Agent Harness 单次运行业务唯一 ID")
    agent_name: Mapped[str] = mapped_column(String(64), comment="被评测 Agent 名称，例如 controlled_agent_loop")
    agent_version: Mapped[str] = mapped_column(String(64), comment="被评测 Agent 的实现或提示词版本标签")
    dataset_version: Mapped[str] = mapped_column(String(64), comment="本次评测使用的数据集版本")
    agent_snapshot_hash: Mapped[str] = mapped_column(String(64), comment="Agent 提示词和工具白名单快照 SHA-256")
    sample_count: Mapped[int] = mapped_column(Integer, default=0, comment="本次实际执行的评测样本数量")
    status_match_rate: Mapped[float] = mapped_column(Float, default=0, comment="最终状态命中率")
    step_sequence_match_rate: Mapped[float] = mapped_column(Float, default=0, comment="动作和工具调用顺序完整命中率")
    tool_call_accuracy: Mapped[float] = mapped_column(Float, default=0, comment="期望工具调用中工具名和参数均命中的比例")
    observation_status_accuracy: Mapped[float] = mapped_column(Float, default=0, comment="工具 observation 状态命中率")
    safety_case_pass_rate: Mapped[float] = mapped_column(Float, default=0, comment="安全样本完整通过率")
    full_pass_rate: Mapped[float] = mapped_column(Float, default=0, comment="所有必填断言同时通过的样本比例")
    avg_step_count: Mapped[float | None] = mapped_column(Float, nullable=True, comment="每条样本平均 Agent 循环步数")
    avg_total_tokens: Mapped[float | None] = mapped_column(Float, nullable=True, comment="每条样本平均总 Token 数")
    avg_cost_ms: Mapped[float | None] = mapped_column(Float, nullable=True, comment="每条样本平均总耗时，单位毫秒")
    metrics_json: Mapped[str] = mapped_column(Text, comment="完整评测指标、Agent 快照和失败样本 JSON")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment="评测运行创建时间")


class AiAgentEvalCaseResult(Base):
    """Agent Loop Harness 单样本结果，保存期望步骤与实际步骤的比对事实。"""

    __tablename__ = "ai_agent_eval_case_result"
    __table_args__ = (
        UniqueConstraint("run_id", "sample_id", name="uk_aaecr_run_sample"),
        Index("ix_aaecr_run", "run_id"),
        Index("ix_aaecr_sample", "sample_id"),
        Index("ix_aaecr_pass", "case_pass"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="数据库自增主键")
    run_id: Mapped[str] = mapped_column(String(64), comment="所属 Agent Harness 运行业务 ID")
    sample_id: Mapped[str] = mapped_column(String(64), comment="关联的通用 AI 评测样本业务 ID")
    sample_type: Mapped[str] = mapped_column(String(32), comment="样本类型，例如 normal、boundary、safety")
    status_match: Mapped[int] = mapped_column(Integer, default=0, comment="最终运行状态是否符合人工期望，使用 0 或 1 存储")
    step_sequence_match: Mapped[int] = mapped_column(Integer, default=0, comment="动作和工具调用顺序是否完整符合期望，使用 0 或 1 存储")
    tool_call_match: Mapped[int] = mapped_column(Integer, default=0, comment="本样本所有期望工具调用的名称和参数是否命中，使用 0 或 1 存储")
    observation_status_match: Mapped[int] = mapped_column(Integer, default=0, comment="本样本期望 observation 状态是否全部命中，使用 0 或 1 存储")
    answer_match: Mapped[int] = mapped_column(Integer, default=1, comment="回答关键字是否命中；未配置时默认通过，使用 0 或 1 存储")
    case_pass: Mapped[int] = mapped_column(Integer, default=0, comment="所有必填断言是否同时通过，使用 0 或 1 存储")
    actual_step_count: Mapped[int] = mapped_column(Integer, default=0, comment="实际 Agent 循环步骤数量")
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="本样本 Agent 消耗的总 Token 数")
    cost_ms: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="本样本 Agent 总耗时，单位毫秒")
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="执行异常类型")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="执行异常简要信息")
    expected_json: Mapped[str] = mapped_column(Text, comment="人工标注的预期最终状态、步骤和安全断言 JSON")
    actual_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="实际 Agent 响应快照 JSON")
    row_json: Mapped[str] = mapped_column(Text, comment="包含命中明细和运行快照的完整行 JSON")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment="样本评测结果创建时间")


class AiAgentEvalGateDecision(Base):
    """Agent Harness 准入结论；只比较已保存报告，不重复调用模型。"""

    __tablename__ = "ai_agent_eval_gate_decision"
    __table_args__ = (
        UniqueConstraint("gate_id", name="uk_aaegd_gate_id"),
        Index("ix_aaegd_baseline", "baseline_run_id"),
        Index("ix_aaegd_candidate", "candidate_run_id"),
        Index("ix_aaegd_agent", "agent_name"),
        Index("ix_aaegd_decision", "decision"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="数据库自增主键")
    gate_id: Mapped[str] = mapped_column(String(64), comment="Agent 评测门禁业务唯一 ID")
    baseline_run_id: Mapped[str] = mapped_column(String(64), comment="基线 Agent Harness 运行业务 ID")
    candidate_run_id: Mapped[str] = mapped_column(String(64), comment="候选 Agent Harness 运行业务 ID")
    agent_name: Mapped[str] = mapped_column(String(64), comment="被比较的 Agent 名称")
    dataset_version: Mapped[str] = mapped_column(String(64), comment="两次运行共用的数据集版本")
    decision: Mapped[str] = mapped_column(String(32), comment="准入结论：pass、reject 或 manual_review")
    comparison_json: Mapped[str] = mapped_column(Text, comment="基线和候选指标对比 JSON")
    reason_json: Mapped[str] = mapped_column(Text, comment="门禁结论原因列表 JSON")
    rule_snapshot_json: Mapped[str] = mapped_column(Text, comment="生成本次门禁结论时使用的规则快照 JSON")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment="Agent 门禁结论创建时间")


class AiPromptPublishAudit(Base):
    """Prompt 人工发布审计表，记录一次候选版本替换线上版本的审批事实。"""

    __tablename__ = "ai_prompt_publish_audit"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    publish_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        comment="Prompt 发布业务唯一 ID",
    )
    gate_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="本次发布依据的评测门禁 ID",
    )
    prompt_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="被发布的候选 Prompt ID",
    )
    prompt_name: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="业务 Prompt 名称，例如 work_order_analysis",
    )
    candidate_prompt_version: Mapped[str] = mapped_column(
        String(32),
        comment="本次发布的候选 Prompt 版本",
    )
    previous_prompt_version: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="发布前线上 active Prompt 版本；首次发布时为空",
    )
    gate_decision: Mapped[str] = mapped_column(
        String(32),
        comment="发布时 Gate 结论：pass 或 manual_review",
    )
    approval_note: Mapped[str] = mapped_column(
        Text,
        comment="人工批准说明，记录性能或业务权衡依据",
    )
    approved_by: Mapped[str] = mapped_column(
        String(64),
        comment="批准人标识；接入认证后应取自登录用户",
    )
    published_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="实际完成状态切换的时间",
    )


class AiPromptRollbackAudit(Base):
    """Prompt 人工回滚审计表，记录线上版本异常后的恢复操作。"""

    __tablename__ = "ai_prompt_rollback_audit"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="数据库自增主键",
    )
    rollback_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        comment="Prompt 回滚业务唯一 ID",
    )
    publish_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        comment="被回滚的原发布审计 ID；一条发布记录仅允许回滚一次",
    )
    prompt_name: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="业务 Prompt 名称，例如 work_order_analysis",
    )
    rolled_back_prompt_version: Mapped[str] = mapped_column(
        String(32),
        comment="被下线的当前 active Prompt 版本",
    )
    restored_prompt_version: Mapped[str] = mapped_column(
        String(32),
        comment="被恢复为 active 的历史 Prompt 版本",
    )
    rollback_reason: Mapped[str] = mapped_column(
        Text,
        comment="人工回滚原因，例如线上质量或延迟异常",
    )
    rolled_back_by: Mapped[str] = mapped_column(
        String(64),
        comment="执行回滚的人员标识；接入认证后应取自登录用户",
    )
    rolled_back_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="实际完成版本状态切换的时间",
    )


class AiSecurityAuditLog(Base):
    """Day27 授权决策审计，只记录脱敏身份与权限事实。"""

    __tablename__ = "ai_security_audit_log"
    __table_args__ = (
        Index("ix_asal_trace", "trace_id"),
        Index("ix_asal_actor", "actor_id"),
        Index("ix_asal_permission", "permission"),
        Index("ix_asal_decision", "decision"),
        Index("ix_asal_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment="数据库自增主键"
    )
    audit_id: Mapped[str] = mapped_column(
        String(64), unique=True, comment="授权审计业务唯一 ID"
    )
    trace_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="关联本次 HTTP 请求链路 ID"
    )
    actor_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="调用者身份 ID；认证失败时为空"
    )
    api_key_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="API Key 管理标识，不保存原始 Key 或 Key 哈希"
    )
    roles_json: Mapped[str] = mapped_column(
        Text, comment="调用者角色名称 JSON 快照"
    )
    permission: Mapped[str] = mapped_column(
        String(255), comment="本次请求要求的权限，多个权限以逗号分隔"
    )
    http_method: Mapped[str] = mapped_column(
        String(16), comment="HTTP 请求方法"
    )
    request_path: Mapped[str] = mapped_column(
        String(255), comment="请求路径，不包含查询参数和请求正文"
    )
    decision: Mapped[str] = mapped_column(
        String(16), comment="授权结论：allow 或 deny"
    )
    reason: Mapped[str] = mapped_column(
        String(64), comment="授权结论原因代码，不保存敏感业务内容"
    )
    resource_type: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="被访问资源类型，例如 prompt、task、trace"
    )
    resource_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, comment="来自受信任路径参数的资源业务 ID"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, comment="授权决策发生时间"
    )
