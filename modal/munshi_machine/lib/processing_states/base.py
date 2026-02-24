from munshi_machine.core import config
from munshi_machine.db.crud.transcript import find_transcript_by_uid
from munshi_machine.db.database import engine
from sqlmodel import Session
from munshi_machine.models.private.transcript import Transcript
from munshi_machine.models.status import TranscriptStatus

logger = config.get_logger(__name__)


class ProcessingState:
    def __init__(self) -> None:
        self.StateSymbol = TranscriptStatus.PENDING

    def _next_state(self):
        return None

    async def run_job(self, uid):
        return None

    def update_status_in_db(self, uid: str) -> Transcript:
        with Session(engine, expire_on_commit=False) as session:
            transcript = find_transcript_by_uid(uid, session)
            transcript.status = self.StateSymbol
            session.commit()
        return transcript
    
    async def on_error(self, uid: str) -> None:
        await FailedProcessingState().run_job(uid)


class FailedProcessingState(ProcessingState):
    def __init__(self) -> None:
        self.StateSymbol = TranscriptStatus.FAILED

    def _next_state(self):
        from munshi_machine.lib.processing_states.init_job import InitProcessingState
        return InitProcessingState()

    async def run_job(self, uid: str) -> int:
        try:
            self.update_status_in_db(uid)
            return 0
        except Exception as err:
            logger.error(f"[{self.StateSymbol}] Error: {err}", exc_info=True)
            return -1
