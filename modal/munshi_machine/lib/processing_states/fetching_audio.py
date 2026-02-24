import asyncio
from munshi_machine.db.crud.transcript import find_transcript_by_uid
from munshi_machine.db.database import engine
from sqlmodel import Session
from munshi_machine.lib.download_audio import download_podcast
from munshi_machine.lib.helpers import transcript_is_podcast_episode
from munshi_machine.lib.id_utils import generate_file_id_from_hash
from munshi_machine.lib.processing_states.base import ProcessingState
from munshi_machine.models.status import TranscriptStatus

from munshi_machine.core import config
from munshi_machine.lib.old_utils import (
    get_valid_audio_path,
)
from .transcribing import TranscribingProcessingState

logger = config.get_logger(__name__)


class FetchingAudioProcessingState(ProcessingState):
    def __init__(self) -> None:
        self.StateSymbol = TranscriptStatus.FETCHING_AUDIO

    def _next_state(self):
        return TranscribingProcessingState()

    async def run_job(self, uid: str):
        try:
            # update new state on the output json
            transcript = self.update_status_in_db(uid)
            if transcript is None:
                logger.error(f"Transcript record does not exist {uid}")
                raise ValueError("Transcript record does not exist")
            logger.info("Fetching Audio...")
            file_hash = transcript.file_hash
            # if file_hash is not found then download audio file and store the file_hash in db
            if file_hash is None:
                if transcript_is_podcast_episode(transcript):
                    if not transcript.audio_url:
                        raise ValueError("Audio URL is missing for podcast episode")
                    new_file_hash, _, audio_fileid = download_podcast(transcript.audio_url)
                    with Session(engine, expire_on_commit=False) as session:
                        transcript = find_transcript_by_uid(uid, session)
                        transcript.file_hash = new_file_hash
                        session.commit()
                    transcript = self.update_status_in_db(uid)
                    logger.info(f"New file hash {transcript.file_hash}")
                else:
                    raise ValueError("Unable to find audio source - file_hash is None and not a podcast episode")
            else:
                # Determine source type based on whether it's a podcast episode
                source_type = "podcast" if transcript_is_podcast_episode(transcript) else "local"
                audio_fileid = generate_file_id_from_hash(file_hash, source_type=source_type)
            # existence of file_hash tells us that the audio can be found on audio_storage volume
            await asyncio.sleep(1)  # Simulate async operation
            audiofile_path = get_valid_audio_path(audio_fileid)
            if audiofile_path:
                await asyncio.sleep(5)
                await self._next_state().run_job(uid)
                return 0
            raise RuntimeError(f"Could not find the necessary audio file for transcription {uid}")
        except Exception as err:
            await self.on_error(uid)
            logger.error(f"[{self.StateSymbol}] Error: {err}", exc_info=True)
            # Clean up files and set failed status
            return -1

