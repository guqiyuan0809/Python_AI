"""将 Day18 的原始文档段切成适合检索的语义完整文本块。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from day04_app.common.exceptions import BusinessException
from day04_app.schemas.knowledge_schema import (
    ChunkSourceReference,
    KnowledgeParentTextChunk,
    KnowledgeTextChunk,
    ParentChildTextChunkBuildResult,
    ParsedDocumentSegment,
)


@dataclass(frozen=True)
class ChunkingConfig:
    """语义切块参数；字符数是预算，不再是普通段落的第一切分依据。"""

    max_characters: int = 500
    overlap_characters: int = 80
    boundary_search_characters: int = 120
    min_chunk_characters: int = 120
    semantic_overflow_characters: int = 80

    def __post_init__(self) -> None:
        if self.max_characters <= 0:
            raise ValueError("max_characters 必须大于 0")
        if not 0 <= self.overlap_characters < self.max_characters:
            raise ValueError("overlap_characters 必须大于等于 0 且小于 max_characters")
        if self.boundary_search_characters <= 0:
            raise ValueError("boundary_search_characters 必须大于 0")
        if not 1 <= self.min_chunk_characters <= self.max_characters:
            raise ValueError("min_chunk_characters 必须大于 0 且不大于 max_characters")
        if self.semantic_overflow_characters < 0:
            raise ValueError("semantic_overflow_characters 必须大于等于 0")


@dataclass(frozen=True)
class ParentChildChunkingConfig:
    """父块服务回答上下文，子块服务召回；两层都优先尊重段落与标题边界。"""

    parent_max_characters: int = 1800
    parent_min_characters: int = 600
    child_max_characters: int = 260
    child_overlap_characters: int = 40
    child_min_characters: int = 60
    child_semantic_overflow_characters: int = 60

    def __post_init__(self) -> None:
        if self.parent_min_characters > self.parent_max_characters:
            raise ValueError("parent_min_characters 必须不大于 parent_max_characters")
        if self.child_min_characters > self.child_max_characters:
            raise ValueError("child_min_characters 必须不大于 child_max_characters")
        if not 0 <= self.child_overlap_characters < self.child_max_characters:
            raise ValueError("child_overlap_characters 必须大于等于 0 且小于 child_max_characters")


@dataclass(frozen=True)
class _SemanticUnit:
    """最小语义单元：通常是段落，超长段落才退化为完整句子或硬切片。"""

    content: str
    source_reference: ChunkSourceReference
    is_heading: bool = False


_BOUNDARY_CHARACTERS = frozenset("。！？；.!?;\n")
_HEADING_NUMBER_PATTERN = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+[章节部分]|\d+(?:\.\d+){0,4}[、.．\s])")
_INLINE_SHORT_HEADING_PATTERN = re.compile(r"(\*\*[^*\n：:。！？；.!?;]{1,18}\*\*)")


def _is_heading(text: str) -> bool:
    """识别常见标题，优先把标题与其后的正文放在同一检索块中。"""
    normalized = text.strip()
    unwrapped = normalized.strip("*").strip()
    if not unwrapped or len(unwrapped) > 80 or "\n" in unwrapped:
        return False
    if normalized.startswith("#") or _HEADING_NUMBER_PATTERN.match(unwrapped):
        return True
    # Word 文档中的一级、小节标题常被解析成单独的加粗短段落。
    return normalized.startswith("**") and normalized.endswith("**") and not any(
        character in unwrapped for character in "。！？；.!?;"
    )


def _split_hard_text(text: str, config: ChunkingConfig) -> list[str]:
    """单句或单个表格单元格超过预算时的最后兜底，此时才应用 overlap。"""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        max_end = min(start + config.max_characters, len(text))
        if max_end == len(text):
            end = max_end
        else:
            search_start = max(
                start + config.max_characters // 2,
                max_end - config.boundary_search_characters,
            )
            end = max_end
            for position in range(max_end - 1, search_start - 1, -1):
                if text[position] in _BOUNDARY_CHARACTERS:
                    end = position + 1
                    break
        content = text[start:end].strip()
        if content:
            pieces.append(content)
        if end >= len(text):
            break
        # overlap 只服务无法按语义拆开的超长文本，不复制正常段落。
        start = max(end - config.overlap_characters, start + 1)
    return pieces


def _split_oversized_segment(text: str, config: ChunkingConfig) -> list[str]:
    """先按句末、换行拆分超长段落；只有超长句子才硬切。"""
    sentences: list[str] = []
    sentence_start = 0
    for position, character in enumerate(text):
        if character in _BOUNDARY_CHARACTERS:
            sentence = text[sentence_start : position + 1].strip()
            if sentence:
                sentences.append(sentence)
            sentence_start = position + 1
    tail = text[sentence_start:].strip()
    if tail:
        sentences.append(tail)

    units: list[str] = []
    pending_sentences: list[str] = []
    for sentence in sentences:
        if len(sentence) > config.max_characters:
            if pending_sentences:
                units.append("".join(pending_sentences))
                pending_sentences = []
            units.extend(_split_hard_text(sentence, config))
            continue
        candidate = "".join([*pending_sentences, sentence])
        if pending_sentences and len(candidate) > config.max_characters:
            units.append("".join(pending_sentences))
            pending_sentences = [sentence]
        else:
            pending_sentences.append(sentence)
    if pending_sentences:
        units.append("".join(pending_sentences))
    return units


def _split_inline_short_headings(text: str) -> list[str]:
    """把段内短标题提升为语义边界，例如“**堆** **作用：** ...”不应黏在上一主题后面。"""
    pieces: list[str] = []
    current_start = 0
    matches = list(_INLINE_SHORT_HEADING_PATTERN.finditer(text))
    for match in matches:
        if match.start() == 0:
            continue
        before_heading = text[match.start() - 1]
        after_heading = text[match.end() : match.end() + 1]
        if before_heading not in {" ", "\n", "。", "！", "？", "；", ".", "!", "?", ";"}:
            continue
        if after_heading and after_heading not in {" ", "\n", "。", "：", ":", "；", ";"}:
            continue
        previous = text[current_start : match.start()].strip()
        if previous:
            pieces.append(previous)
        current_start = match.start()
    tail = text[current_start:].strip()
    if tail:
        pieces.append(tail)
    return pieces or [text]


def _build_semantic_units(
    segments: list[ParsedDocumentSegment],
    config: ChunkingConfig,
) -> list[_SemanticUnit]:
    """解析器已有的段落/表格行就是首选语义边界，不先把全文拼成字符长串。"""
    units: list[_SemanticUnit] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        source_reference = ChunkSourceReference(
            segment_index=segment.segment_index,
            location=segment.location,
        )
        inline_heading_pieces = _split_inline_short_headings(text)
        if len(text) <= config.max_characters and len(inline_heading_pieces) == 1:
            units.append(
                _SemanticUnit(
                    content=text,
                    source_reference=source_reference,
                    is_heading=_is_heading(text),
                )
            )
            continue

        # 过长段落在自身内部降级为句子级单元，仍然继承同一个原文来源。
        for heading_piece in inline_heading_pieces:
            if len(heading_piece) <= config.max_characters:
                units.append(
                    _SemanticUnit(
                        content=heading_piece,
                        source_reference=source_reference,
                        is_heading=_is_heading(heading_piece),
                    )
                )
                continue
            units.extend(
                _SemanticUnit(content=piece, source_reference=source_reference)
                for piece in _split_oversized_segment(heading_piece, config)
            )
    return units


def _build_chunk(units: list[_SemanticUnit], document_id: str, chunk_index: int) -> KnowledgeTextChunk:
    """同一原始段因超长拆出多个单元时，来源引用去重但保持原文出现顺序。"""
    source_references = list(
        {
            (unit.source_reference.segment_index, unit.source_reference.location): unit.source_reference
            for unit in units
        }.values()
    )
    content = "\n".join(unit.content for unit in units)
    return KnowledgeTextChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        char_count=len(content),
        source_references=source_references,
    )


def _group_semantic_units(
    units: list[_SemanticUnit],
    *,
    max_characters: int,
    min_characters: int,
    semantic_overflow_characters: int,
    force_heading_boundary: bool = False,
) -> list[list[_SemanticUnit]]:
    """按标题和段落分组，供父块、子块共用，避免两套规则逐渐漂移。"""
    groups: list[list[_SemanticUnit]] = []
    current_units: list[_SemanticUnit] = []
    current_length = 0
    for unit in units:
        if unit.is_heading and current_units and (force_heading_boundary or current_length >= min_characters):
            groups.append(current_units)
            current_units = []
            current_length = 0

        separator_length = 1 if current_units else 0
        candidate_length = current_length + separator_length + len(unit.content)
        if current_units and candidate_length > max_characters:
            allow_semantic_overflow = (
                current_length < min_characters
                and candidate_length <= max_characters + semantic_overflow_characters
            )
            if not allow_semantic_overflow:
                groups.append(current_units)
                current_units = []
                current_length = 0
        if current_units:
            current_length += 1
        current_units.append(unit)
        current_length += len(unit.content)
    if current_units:
        groups.append(current_units)
    return groups


def _expand_units_for_child_chunks(
    parent_units: list[_SemanticUnit],
    child_config: ChunkingConfig,
) -> list[_SemanticUnit]:
    """父块保留完整段落，子块才按更小预算拆分超长段，且保持精确来源段。"""
    child_units: list[_SemanticUnit] = []
    for unit in parent_units:
        if len(unit.content) <= child_config.max_characters:
            child_units.append(unit)
            continue
        child_units.extend(
            _SemanticUnit(
                content=piece,
                source_reference=unit.source_reference,
                is_heading=False,
            )
            for piece in _split_oversized_segment(unit.content, child_config)
        )
    return child_units


def build_parent_child_text_chunks(
    *,
    document_id: str,
    segments: list[ParsedDocumentSegment],
    config: ParentChildChunkingConfig | None = None,
) -> ParentChildTextChunkBuildResult:
    """构建父子块：子块精确定位原文，父块提供完整可回答上下文。"""
    if not document_id:
        raise ValueError("document_id 不能为空")
    actual_config = config or ParentChildChunkingConfig()
    parent_split_config = ChunkingConfig(
        max_characters=actual_config.parent_max_characters,
        overlap_characters=0,
        boundary_search_characters=120,
        min_chunk_characters=actual_config.parent_min_characters,
        semantic_overflow_characters=200,
    )
    parent_units = _build_semantic_units(segments, parent_split_config)
    if not parent_units:
        raise BusinessException(code=40066, message="文档没有可用于父子切块的原始文本段")

    parent_groups = _group_semantic_units(
        parent_units,
        max_characters=actual_config.parent_max_characters,
        min_characters=actual_config.parent_min_characters,
        semantic_overflow_characters=200,
        force_heading_boundary=False,
    )
    child_split_config = ChunkingConfig(
        max_characters=actual_config.child_max_characters,
        overlap_characters=actual_config.child_overlap_characters,
        boundary_search_characters=120,
        min_chunk_characters=actual_config.child_min_characters,
        semantic_overflow_characters=actual_config.child_semantic_overflow_characters,
    )
    parent_chunks: list[KnowledgeParentTextChunk] = []
    child_chunks: list[KnowledgeTextChunk] = []
    for parent_index, parent_group in enumerate(parent_groups):
        parent = _build_chunk(parent_group, document_id, parent_index)
        parent_chunks.append(
            KnowledgeParentTextChunk(
                document_id=document_id,
                parent_index=parent_index,
                content=parent.content,
                char_count=parent.char_count,
                source_references=parent.source_references,
            )
        )
        child_groups = _group_semantic_units(
            _expand_units_for_child_chunks(parent_group, child_split_config),
            max_characters=actual_config.child_max_characters,
            min_characters=actual_config.child_min_characters,
            semantic_overflow_characters=actual_config.child_semantic_overflow_characters,
            force_heading_boundary=True,
        )
        for child_group in child_groups:
            child = _build_chunk(child_group, document_id, len(child_chunks))
            child_chunks.append(child.model_copy(update={"parent_index": parent_index}))
    return ParentChildTextChunkBuildResult(
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
    )


def build_text_chunks(
    *,
    document_id: str,
    segments: list[ParsedDocumentSegment],
    config: ChunkingConfig | None = None,
) -> list[KnowledgeTextChunk]:
    """语义完整性优先切块：按段落/标题/句子组织，长度超限才降级拆分。"""
    if not document_id:
        raise ValueError("document_id 不能为空")

    actual_config = config or ChunkingConfig()
    semantic_units = _build_semantic_units(segments, actual_config)
    if not semantic_units:
        raise BusinessException(code=40066, message="文档没有可用于切块的原始文本段")

    chunks: list[KnowledgeTextChunk] = []
    current_units: list[_SemanticUnit] = []
    current_length = 0

    for unit in semantic_units:
        # 已积累足够正文时，下一标题应开启新块，避免一个 chunk 混入两个主题。
        if unit.is_heading and current_units:
            chunks.append(_build_chunk(current_units, document_id, len(chunks)))
            current_units = []
            current_length = 0

        separator_length = 1 if current_units else 0
        candidate_length = current_length + separator_length + len(unit.content)
        if current_units and candidate_length > actual_config.max_characters:
            # 少量溢出换取“不留下过短尾块”或“标题不脱离正文”。
            allow_semantic_overflow = (
                current_length < actual_config.min_chunk_characters
                and candidate_length
                <= actual_config.max_characters + actual_config.semantic_overflow_characters
            )
            if not allow_semantic_overflow:
                chunks.append(_build_chunk(current_units, document_id, len(chunks)))
                current_units = []
                current_length = 0

        if current_units:
            current_length += 1
        current_units.append(unit)
        current_length += len(unit.content)

    if current_units:
        chunks.append(_build_chunk(current_units, document_id, len(chunks)))
    return chunks
