from typing import List
from uuid import UUID

from munshi_machine.core import config
from munshi_machine.core.app import app, custom_secret
from munshi_machine.core.images import base_image
from munshi_machine.core.volumes import (
    audio_storage_vol,
    transcriptions_vol,
)
from munshi_machine.db.crud.transcript import find_transcript_by_uid
from munshi_machine.lib.processing_states import processingStateFactory
import modal

from munshi_machine.models.status import TranscriptStatus

logger = config.get_logger(__name__)


@app.function(
    image=base_image,
    volumes={
        str(config.RAW_AUDIO_DIR): audio_storage_vol,
        str(config.TRANSCRIPTIONS_DIR): transcriptions_vol,
    },
    secrets=[custom_secret],
    timeout=2000,
    max_containers=5,
)
@modal.concurrent(max_inputs=5, target_inputs=1)
async def init_transcription(uids: UUID | List[UUID] = None, restart: bool = False):
    if isinstance(uids, UUID):
        uids = [uids]

    if uids is None or not isinstance(uids, List):
        logger.error(f"Invalid uid value {uids}")
        return

    # refresh volumes
    audio_storage_vol.reload()

    for uid in uids:
        transcript = find_transcript_by_uid(uid)
        if transcript:
            status = transcript.status
            if status == TranscriptStatus.FAILED and restart:
                logger.info(f"Restarting transcription... {status}")
                await processingStateFactory(TranscriptStatus.PENDING).run_job(uid)
            elif status != TranscriptStatus.FAILED: 
                logger.info(f"Resuming transcription... {status}")
                await processingStateFactory(status).run_job(uid)
