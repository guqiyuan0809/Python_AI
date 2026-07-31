"""将 Day18 的原始文档段切成适合检索的文本块。"""

from __future__ import annotations

from dataclasses import dataclass

from day04_app.common.exceptions import BusinessException
from day04_app.schemas.knowledge_schema import (
    ChunkSourceReference,
    KnowledgeTextChunk,
    ParsedDocumentSegment,
)


@dataclass(frozen=True)
class ChunkingConfig:
    """切块参数；当前使用字符数，避免依赖特定模型的 tokenizer。"""

    max_characters: int = 500
    overlap_characters: int = 80
    boundary_search_characters: int = 120

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError("max_characters 必须大于 0")
        if not 0 <= self.overlap_characters < self.max_characters:
            raise ValueError("overlap_characters 必须大于等于 0 且小于 max_characters")
        if self.boundary_search_characters <= 0:
            raise ValueError("boundary_search_characters 必须大于 0")


@dataclass(frozen=True)
class _SourceRange:
    """原始段在拼接文本中的字符范围，结束位置遵循 Python 左闭右开规则。"""

    start: int
    end: int
    segment_index: int
    location: str


_BOUNDARY_CHARACTERS = frozenset("。！？；.!?;\n")


def _build_source_text(
    segments: list[ParsedDocumentSegment],
) -> tuple[str, list[_SourceRange]]:
    """拼接原始段，同时保存每段的字符范围，不能丢掉后续引用所需的来源。"""
    text_parts: list[str] = []
    source_ranges: list[_SourceRange] = []
    current_offset = 0

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if text_parts:
            # 段与段之间用换行保留自然语义边界，切块时也优先在此断开。
            text_parts.append("\n")
            current_offset += 1
        text_parts.append(text)
        source_ranges.append(
            _SourceRange(
                start=current_offset,
                end=current_offset + len(text),
                segment_index=segment.segment_index,
                location=segment.location,
            )
        )
        current_offset += len(text)

    return "".join(text_parts), source_ranges


def _choose_chunk_end(text: str, start: int, config: ChunkingConfig) -> int:
    """优先在句末或段落边界切开；找不到边界时才按最大字符数硬切。"""
    max_end = min(start + config.max_characters, len(text))
    if max_end == len(text):
        return max_end

    # 不在窗口开头寻找断点，避免某个很早的句号让切块过短。
    search_start = max(
        start + config.max_characters // 2,
        max_end - config.boundary_search_characters,
    )
    for position in range(max_end - 1, search_start - 1, -1):
        if text[position] in _BOUNDARY_CHARACTERS:
            return position + 1
    return max_end


def _trim_window(text: str, start: int, end: int) -> tuple[str, int, int]:
    """去除窗口边缘的空白，并同步修正字符范围，保证来源映射准确。"""
    raw_content = text[start:end]
    left_trimmed = raw_content.lstrip()
    leading_whitespace_count = len(raw_content) - len(left_trimmed)
    content = left_trimmed.rstrip()
    return content, start + leading_whitespace_count, start + leading_whitespace_count + len(content)


def _build_source_references(
    source_ranges: list[_SourceRange],
    content_start: int,
    content_end: int,
) -> list[ChunkSourceReference]:
    """只保留与当前切块字符范围有交集的原始文档段。"""
    return [
        ChunkSourceReference(
            segment_index=source_range.segment_index,
            location=source_range.location,
        )
        for source_range in source_ranges
        if source_range.start < content_end and source_range.end > content_start
    ]


def build_text_chunks(
    *,
    document_id: str,
    segments: list[ParsedDocumentSegment],
    config: ChunkingConfig | None = None,
) -> list[KnowledgeTextChunk]:
    """把原始文档段切为重叠文本块，输出仍可精确回溯至 Day18 的来源位置。"""
    if not document_id:
        raise ValueError("document_id 不能为空")

    text, source_ranges = _build_source_text(segments)
    if not text:
        raise BusinessException(code=40066, message="文档没有可用于切块的原始文本段")

    actual_config = config or ChunkingConfig()
    chunks: list[KnowledgeTextChunk] = []
    start = 0

    while start < len(text):
        end = _choose_chunk_end(text, start, actual_config)
        content, content_start, content_end = _trim_window(text, start, end)
        if content:
            chunks.append(
                KnowledgeTextChunk(
                    document_id=document_id,
                    chunk_index=len(chunks),
                    content=content,
                    char_count=len(content),
                    source_references=_build_source_references(
                        source_ranges,
                        content_start,
                        content_end,
                    ),
                )
            )

        if end >= len(text):
            break
        # 重叠窗口让“前后文跨切块”的语义在相邻 chunk 中各保留一部分。
        start = max(end - actual_config.overlap_characters, start + 1)

    return chunks
