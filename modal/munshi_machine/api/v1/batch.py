from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from munshi_machine.core import config
from munshi_machine.db.database import engine
from sqlmodel import Session
from munshi_machine.functions.fetch_podcasts import fetch_and_batch_transcribe_podcasts
from munshi_machine.models.private.batch import Batch

logger = config.get_logger("API.v1.podcasts")
router = APIRouter()


class PodcastBatch(BaseModel):
    podcast_guids: List[str]


@router.post("/batch_transcribe_podcasts")
def batch_podcasts(payload: PodcastBatch):
    with Session(engine) as session:
        # Create batch instance
        batch = Batch()
        
        # Generate batch.uid before spawning (needed for function call)
        session.add(batch)
        session.flush()  # Flush to generate UUID without committing
        
        # Capture batch_uid before session closes
        batch_uid = batch.uid
        
        logger.info(f"Starting batch transcription for {len(payload.podcast_guids)} podcasts")
        
        # Spawn Modal function with batch.uid
        fc_call = fetch_and_batch_transcribe_podcasts.spawn(payload.podcast_guids, batch_uid)
        
        # Set function_call_id and commit once
        batch.function_call_id = fc_call.object_id
        session.commit()
        
        # Capture call_id before session closes
        call_id = fc_call.object_id

    return {"message": "successful", "call_id": call_id, "batch_id": batch_uid}
