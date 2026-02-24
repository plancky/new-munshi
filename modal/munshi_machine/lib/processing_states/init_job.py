import asyncio
import time

from munshi_machine.db.crud.transcript import find_transcript_by_uid
from munshi_machine.db.database import engine
from sqlmodel import Session
from munshi_machine.lib.processing_states.base import ProcessingState
from munshi_machine.models.status import TranscriptStatus

from ...core import config

logger = config.get_logger(__name__)

class InitProcessingState(ProcessingState):
    def __init__(self) -> None:
        self.StateSymbol = TranscriptStatus.PENDING

    def _next_state(self, transcript=None):
        from .fetching_audio import FetchingAudioProcessingState
        from .transcribing import TranscribingProcessingState
        
        # If file_hash exists, audio is already uploaded - skip fetching
        if transcript and transcript.file_hash:
            logger.info(f"File hash exists ({transcript.file_hash[:16]}...), skipping audio fetch, going straight to transcription")
            return TranscribingProcessingState()
        
        # Otherwise, need to fetch audio (for podcast episodes)
        return FetchingAudioProcessingState()

    async def run_job(self, uid: str, chained=True):
        try:
            with Session(engine, expire_on_commit=False) as session:
                transcript = find_transcript_by_uid(uid, session)
                transcript.job_start_time = time.time()
                session.commit()

            await asyncio.sleep(1)
            if chained:
                # Pass transcript to determine next state
                next_state = self._next_state(transcript)
                await next_state.run_job(uid)
            return 0
        except Exception as err:
            self.on_error(uid)
            logger.error(f"[{self.StateSymbol}] Error: {err}", exc_info=True)
            return -1


