"""知识库文档解析阶段的 DTO。

无论原文件是 Word、PDF 还是 Excel，后续 RAG 流程都只消费这里定义的统一结构。
"""

from typing import Any

from pydantic import BaseModel, Field


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
