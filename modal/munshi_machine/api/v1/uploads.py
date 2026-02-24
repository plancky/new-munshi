from typing import Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    responses,
    status,
)
from sqlmodel import Session

from ...core import config
from ...db.crud.transcript import create_transcript, find_transcript_by_file_hash
from ...db.database import engine
from ...lib.id_utils import get_hash_from_file_id
from ...lib.upload_utils import stream_upload
from ...models.private.transcript import Transcript

logger = config.get_logger("API.v1.uploads")
router = APIRouter()


async def valid_content_length(content_length: Optional[int] = Header(None)):
    if content_length is not None and content_length >= 524_288_000:
        raise HTTPException(status_code=413, detail="File too large (max 500MB)")
    return content_length


@router.post("/upload_file", dependencies=[Depends(valid_content_length)])
async def upload_file(file: UploadFile = File(...), fileName: str = Form(...)):
    try:
        logger.info(
            f"Upload request received - fileName: {fileName}, file size: {file.size if hasattr(file, 'size') else 'unknown'}"
        )
        logger.info(f"File content type: {file.content_type}")

        result = await stream_upload(file, fileName)
        uploaded, fileId, is_existing = result[:3]

        if uploaded:
            # Extract the raw hash from file_id (removes "local_" or "podcast_" prefix)
            file_hash = get_hash_from_file_id(fileId)
            
            with Session(engine, expire_on_commit=False) as session:
                # Check if transcript record already exists for this file hash
                transcript = find_transcript_by_file_hash(file_hash, session)
                
                if transcript:
                    # Transcript already exists
                    logger.info(f"Existing transcript found for file with uid: {transcript.uid}")
                    return responses.JSONResponse(
                        content={
                            "status": "transcript_exists",
                            "id": str(transcript.uid),
                            "message": "Transcript already exists for this file",
                        },
                        status_code=200,
                    )
                else:
                    # Create new transcript record
                    new_transcript = Transcript(
                        title=fileName,
                        file_hash=file_hash,
                    )
                    transcript = create_transcript(new_transcript, session)
                    logger.info(f"New file uploaded successfully with uid: {transcript.uid}")
                    return responses.JSONResponse(
                        content={"status": "file uploaded", "id": str(transcript.uid)}, 
                        status_code=200
                    )
        else:
            logger.error("Upload failed - stream_upload returned False")
            return responses.JSONResponse(content={"status": "file not uploaded"}, status_code=403)
    except HTTPException as e:
        logger.error(f"HTTP Exception in upload: {e.status_code} - {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected upload error: {e}", exc_info=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Upload failed: {str(e)}")


@router.post("/transcribe_local")
async def transcribe_local(request: Request):
    from munshi_machine.lib.processing_states.init_job import InitProcessingState

    logger.info(f"Received a request {request.client}")

    payload = await request.json()
    vid_str = payload.get("vid")
    
    if not vid_str:
        return responses.PlainTextResponse(content="bad request. vid missing", status_code=400)
    
    try:
        vid = UUID(vid_str)
    except (ValueError, AttributeError, TypeError):
        return responses.JSONResponse(
            content={"error": "Invalid vid format. Must be a valid UUID."},
            status_code=400
        )

    try:
        await InitProcessingState().run_job(vid, chained=False)
        from munshi_machine.functions.init_transcription import init_transcription

        call = init_transcription.spawn(vid)
        return responses.JSONResponse(content={"vid": str(vid), "call_id": call.object_id}, status_code=200)
    except RuntimeError:
        logger.error("transcribe_local failed", exc_info=True)
        return responses.JSONResponse(content={"error": "Server error occurred"}, status_code=500)