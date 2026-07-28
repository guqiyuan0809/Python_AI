"""知识库文件上传接口。"""

from fastapi import APIRouter, Depends, File, Request, UploadFile
from sqlalchemy.orm import Session

from day04_app.common.response import ApiResponse, success
from day04_app.database import get_db
from day04_app.schemas.knowledge_schema import DocumentParseResponse, DocumentUploadResponse
from day04_app.services.knowledge_document_parse_service import parse_document_by_id
from day04_app.services.document_storage_service import delete_stored_document, save_uploaded_document
from day04_app.services.knowledge_document_service import create_uploaded_document


router = APIRouter(prefix="/api/knowledge", tags=["知识库"])


@router.post("/documents/upload", response_model=ApiResponse[DocumentUploadResponse], summary="安全上传知识库文件")
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="仅支持 docx、pdf、xlsx，最大 20MB"),
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentUploadResponse]:
    stored_document = await save_uploaded_document(file)
    try:
        document = create_uploaded_document(db, stored_document, request.state.trace_id)
    except Exception:
        # 磁盘和 MySQL 不能共享一个事务；数据库写入失败时主动补偿刚落盘的文件。
        delete_stored_document(stored_document)
        raise
    return success(
        DocumentUploadResponse(
            document_id=stored_document.document_id,
            original_file_name=stored_document.original_file_name,
            file_type=stored_document.file_type,
            file_size=stored_document.file_size,
            status=document.status,
        ),
        message="文件已安全落盘，等待后续解析",
        trace_id=request.state.trace_id,
    )


@router.post(
    "/documents/{document_id}/parse",
    response_model=ApiResponse[DocumentParseResponse],
    summary="按文档 ID 解析已上传文件",
)
def parse_document(
    document_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentParseResponse]:
    document, parsed_document = parse_document_by_id(db, document_id)
    return success(
        DocumentParseResponse(
            document_id=document.document_id,
            status=document.status,
            parser_name=document.parser_name or parsed_document.parser_name,
            parsed_segment_count=document.parsed_segment_count,
            # 仅返回少量预览，完整原文段已经落到数据库，不能随大文件一次性返回。
            preview_segments=parsed_document.segments[:5],
        ),
        message="文档解析完成",
        trace_id=request.state.trace_id,
    )
