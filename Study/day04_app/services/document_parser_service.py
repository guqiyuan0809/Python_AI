"""知识库文档解析器注册中心。

类似 Java 中面向 DocumentParser 接口注入不同实现的策略模式。
本阶段只定义统一契约和路由选择，具体 Word/PDF/Excel 解析器将在后续小阶段接入。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from day04_app.common.exceptions import BusinessException
from day04_app.schemas.knowledge_schema import ParsedDocument
from day04_app.services.excel_document_parser import ExcelDocumentParser
from day04_app.services.pdf_document_parser import PdfDocumentParser
from day04_app.services.word_document_parser import WordDocumentParser


class DocumentParser(Protocol):
    """Python Protocol 类似 Java interface；满足同名属性和方法即可作为解析器使用。"""

    supported_file_types: frozenset[str]

    def parse(self, *, document_id: str, file_path: Path) -> ParsedDocument:
        """把已安全落盘的原文件转换为统一 ParsedDocument。"""


def normalize_file_type(file_path: Path) -> str:
    """将文件扩展名统一为小写类型名，例如 .PDF 转为 pdf。"""
    file_type = file_path.suffix.removeprefix(".").lower()
    if not file_type:
        raise BusinessException(code=40051, message="无法识别文件类型：文件没有扩展名")
    return file_type


class DocumentParserRegistry:
    """按文件类型选择解析策略；后续新增 PPT/HTML 时只需注册新解析器。"""

    def __init__(self, parsers: Iterable[DocumentParser] = ()) -> None:
        self._parser_by_file_type: dict[str, DocumentParser] = {}
        for parser in parsers:
            self.register(parser)

    def register(self, parser: DocumentParser) -> None:
        for file_type in parser.supported_file_types:
            normalized_type = file_type.lower().removeprefix(".")
            if normalized_type in self._parser_by_file_type:
                raise ValueError(f"文件类型 {normalized_type} 已注册解析器，不能重复覆盖")
            self._parser_by_file_type[normalized_type] = parser

    def get_parser(self, file_path: Path) -> DocumentParser:
        file_type = normalize_file_type(file_path)
        parser = self._parser_by_file_type.get(file_type)
        if parser is None:
            supported_types = ", ".join(sorted(self._parser_by_file_type)) or "暂未注册任何解析器"
            raise BusinessException(
                code=40052,
                message=f"暂不支持 {file_type} 文件解析，当前支持：{supported_types}",
            )
        return parser

    def parse(self, *, document_id: str, file_path: Path) -> ParsedDocument:
        """统一校验文件，再委派给对应格式的具体解析器。"""
        if not file_path.is_file():
            raise BusinessException(code=40053, message="待解析文件不存在或不是普通文件")
        return self.get_parser(file_path).parse(document_id=document_id, file_path=file_path)


# 类似 Spring 启动时完成策略 Bean 注册；调用方只依赖这个统一入口。
document_parser_registry = DocumentParserRegistry(
    parsers=[
        WordDocumentParser(),
        PdfDocumentParser(),
        ExcelDocumentParser(),
    ]
)
