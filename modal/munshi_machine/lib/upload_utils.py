import hashlib
import logging
import os
import uuid
from typing import Tuple

from fastapi import UploadFile
from munshi_machine.db.crud.transcript import find_transcript_by_file_hash

from ..core.config import RAW_AUDIO_DIR
from ..core.volumes import audio_storage_vol
from .id_utils import generate_file_id_from_hash

logger = logging.getLogger("upload_utils")


def generate_upload_id() -> str:
    """Generate a unique upload ID for tracking."""
    return str(uuid.uuid4())


async def stream_upload(
    file: UploadFile,
    fileName: str,
) -> Tuple[bool, str, bool]:
    upload_id = generate_upload_id()
    temp_file_path = None
    file_id = None
    try:
        logger.info(
            f"Starting streaming upload for {fileName} (upload_id: {upload_id})"
        )

        if not fileName:
            raise ValueError("No fileName provided")
        if not hasattr(file, "read"):
            raise ValueError("Invalid file object - missing read method")

        logger.info(
            f"File info - name: {fileName}, content_type: {getattr(file, 'content_type', 'unknown')}"
        )

        if not fileName.lower().endswith(
            (".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi")
        ):
            raise ValueError(f"Unsupported file type: {fileName}")

        temp_file_path = os.path.join(RAW_AUDIO_DIR, f"{upload_id}.uploading")

        hasher = hashlib.sha256()

        total_bytes = 0
        logger.info(f"Streaming to temporary file: {temp_file_path}")

        with open(temp_file_path, "wb") as temp_file:
            try:
                while True:
                    chunk = await file.read(65536)
                    if not chunk:
                        break
                    temp_file.write(chunk)
                    hasher.update(chunk)
                    total_bytes += len(chunk)
            except Exception as read_error:
                logger.error(f"Error reading file data: {read_error}")
                raise ValueError(f"Error reading file data: {read_error}")

        logger.info(f"Upload completed: {total_bytes} bytes")

        if total_bytes == 0:
            raise ValueError("File is empty (0 bytes)")
        if total_bytes > 524_288_000:
            raise ValueError(f"File too large: {total_bytes} bytes (max 500MB)")

        content_hash = hasher.hexdigest()
        file_id = generate_file_id_from_hash(content_hash)
        logger.info(f"Generated file ID: {file_id} (from hash: {content_hash[:16]})")
        transcript = find_transcript_by_file_hash(content_hash)
        if transcript:
            logger.info(
                f"Completed transcript found for {file_id}! Redirecting to existing transcript."
            )
            return [True, file_id, True]

        file_extension = os.path.splitext(fileName)[1].lower()
        final_path = os.path.join(RAW_AUDIO_DIR, f"{file_id}{file_extension}")
        if os.path.exists(final_path):
            logger.info(f"File already exists at {final_path}, removing temporary file")
            os.remove(temp_file_path)
            return [True, file_id, False]

        os.rename(temp_file_path, final_path)
        audio_storage_vol.commit()
        logger.info(
            f"File successfully uploaded and stored as: {file_id}{file_extension}"
        )
        return [True, file_id, False]

    except (ValueError, TypeError) as validation_error:
        logger.error(f"Validation error: {validation_error}")
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup temporary file: {cleanup_error}")
        raise validation_error
    except Exception as e:
        logger.error(f"Unexpected upload error: {e}")
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup temporary file: {cleanup_error}")
        if file_id:
            file_extension = (
                os.path.splitext(fileName)[1].lower() if fileName else ".mp3"
            )
            final_path = os.path.join(RAW_AUDIO_DIR, f"{file_id}{file_extension}")
            if os.path.exists(final_path):
                try:
                    os.remove(final_path)
                    logger.info(f"Cleaned up partial final file: {final_path}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup final file: {cleanup_error}")
        return [False, None, False]
