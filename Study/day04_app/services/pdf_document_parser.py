"""PDF 文本层解析器。"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from day04_app.common.exceptions import BusinessException
from day04_app.schemas.knowledge_schema import ParsedDocument, ParsedDocumentSegment


class PdfDocumentParser:
    """按页提取 PDF 的可选中文本层，不承担扫描件 OCR。"""

    supported_file_types = frozenset({"pdf"})

    def parse(self, *, document_id: str, file_path: Path) -> ParsedDocument:
        try:
            pdf_reader = PdfReader(file_path)
        except (PdfReadError, OSError, ValueError) as exc:
            raise BusinessException(code=40061, message="PDF 文件无法读取或文件已损坏") from exc

        if pdf_reader.is_encrypted:
            # 密码不能作为接口参数透传或记录日志，企业中应走授权后的受控解密流程。
            raise BusinessException(code=40062, message="暂不支持解析加密 PDF，请上传已授权的非加密版本")

        segments: list[ParsedDocumentSegment] = []
        empty_page_count = 0
        try:
            for page_index, page in enumerate(pdf_reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    empty_page_count += 1
                    continue
                segments.append(
                    ParsedDocumentSegment(
                        segment_index=len(segments),
                        text=text,
                        location=f"Page:{page_index}",
                        metadata={"source_type": "page", "page_index": page_index},
                    )
                )
        except (PdfReadError, OSError, ValueError) as exc:
            raise BusinessException(code=40061, message="PDF 文本提取失败") from exc

        if not segments:
            # 扫描 PDF 往往只有页面图片，必须显式进入 OCR 流程，不能返回空成功结果。
            raise BusinessException(code=40063, message="PDF 不包含可提取文本，扫描件需要 OCR 后再导入")

        return ParsedDocument(
            document_id=document_id,
            file_name=file_path.name,
            file_type="pdf",
            parser_name=self.__class__.__name__,
            segments=segments,
            metadata={
                "page_count": len(pdf_reader.pages),
                "text_page_count": len(segments),
                "empty_page_count": empty_page_count,
                "segment_count": len(segments),
            },
        )
