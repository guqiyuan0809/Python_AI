"""知识库原始文件的安全落盘服务。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from uuid import uuid4
import zipfile

from fastapi import UploadFile

from day04_app.common.exceptions import BusinessException
from day04_app.services.document_parser_service import normalize_file_type
from settings import settings


ALLOWED_FILE_TYPES = frozenset({"docx", "pdf", "xlsx"})
WRITE_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class StoredDocument:
    """服务层内部对象；物理路径不能直接返回给前端。"""

    document_id: str
    original_file_name: str
    file_type: str
    file_size: int
    content_sha256: str
    storage_key: str
    file_path: Path


def _get_safe_original_file_name(upload_file: UploadFile) -> str:
    """仅保留文件名部分，阻断 ../ 或磁盘绝对路径进入服务端目录。"""
    original_file_name = Path((upload_file.filename or "").replace("\\", "/")).name
    if not original_file_name:
        raise BusinessException(code=40054, message="上传文件必须包含文件名")
    if len(original_file_name) > 255:
        raise BusinessException(code=40054, message="上传文件名不能超过 255 个字符")
    return original_file_name


def _validate_file_content(file_path: Path, file_type: str) -> None:
    """扩展名可伪造，因此在不解析正文的前提下做最低限度的真实格式校验。"""
    if file_type == "pdf":
        with file_path.open("rb") as file:
            if file.read(5) != b"%PDF-":
                raise BusinessException(code=40057, message="文件内容不是合法的 PDF 格式")
        return

    try:
        with zipfile.ZipFile(file_path) as archive:
            entry_names = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise BusinessException(code=40057, message=f"文件内容不是合法的 {file_type.upper()} 格式") from exc

    required_entry = "word/document.xml" if file_type == "docx" else "xl/workbook.xml"
    if "[Content_Types].xml" not in entry_names or required_entry not in entry_names:
        raise BusinessException(code=40057, message=f"文件内容不是合法的 {file_type.upper()} 格式")


async def save_uploaded_document(upload_file: UploadFile) -> StoredDocument:
    """将上传流写入临时文件，校验通过后再原子移动到正式目录。"""
    original_file_name = _get_safe_original_file_name(upload_file)
    file_type = normalize_file_type(Path(original_file_name))
    if file_type not in ALLOWED_FILE_TYPES:
        supported_types = ", ".join(sorted(ALLOWED_FILE_TYPES))
        raise BusinessException(code=40055, message=f"暂只支持上传：{supported_types}")

    document_id = uuid4().hex
    upload_dir = settings.knowledge_upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    storage_key = f"{document_id}.{file_type}"
    temporary_path = upload_dir / f"{document_id}.uploading"
    target_path = upload_dir / storage_key
    file_size = 0
    sha256 = hashlib.sha256()

    try:
        with temporary_path.open("wb") as file:
            while chunk := await upload_file.read(WRITE_CHUNK_SIZE):
                file_size += len(chunk)
                if file_size > settings.knowledge_upload_max_bytes:
                    raise BusinessException(
                        code=40056,
                        message=f"文件不能超过 {settings.knowledge_upload_max_bytes // 1024 // 1024}MB",
                    )
                # 分块写入避免将整个上传文件一次性读进服务进程内存。
                file.write(chunk)
                sha256.update(chunk)

        if file_size == 0:
            raise BusinessException(code=40056, message="不允许上传空文件")

        _validate_file_content(temporary_path, file_type)
        # replace 在同一磁盘目录内完成原子改名，解析器只会看到已校验完成的正式文件。
        temporary_path.replace(target_path)
        return StoredDocument(
            document_id=document_id,
            original_file_name=original_file_name,
            file_type=file_type,
            file_size=file_size,
            content_sha256=sha256.hexdigest(),
            storage_key=storage_key,
            file_path=target_path,
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        await upload_file.close()


def delete_stored_document(stored_document: StoredDocument) -> None:
    """数据库元数据提交失败时删除刚写入的文件，避免产生可预见的孤儿文件。"""
    stored_document.file_path.unlink(missing_ok=True)
