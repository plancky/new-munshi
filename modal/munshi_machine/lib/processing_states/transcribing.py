from pathlib import Path

from munshi_machine.core import config
from munshi_machine.core.volumes import audio_storage_vol
from munshi_machine.db.crud.transcript import update_or_create_transcripts
from munshi_machine.lib.helpers import (retry_with_db_refetch,
                                        transcript_is_podcast_episode)
from munshi_machine.lib.id_utils import generate_file_id_from_hash
from munshi_machine.lib.old_utils import get_valid_audio_path
from munshi_machine.lib.processing_states.base import ProcessingState
from munshi_machine.lib.processing_states.cleaning import \
    CleaningProcessingState
from munshi_machine.models.private.transcript import Transcript
from munshi_machine.models.status import TranscriptStatus

logger = config.get_logger(__name__)


class TranscribingProcessingState(ProcessingState):
    def __init__(self) -> None:
        self.StateSymbol = TranscriptStatus.TRANSCRIBING

    def _next_state(self):
        return CleaningProcessingState()

    async def run_job(self, uid: str) -> None:
        from munshi_machine.functions.embeddings import embed_transcript  # NEW
        from munshi_machine.functions.transcribe import Parakeet

        try:
            audio_storage_vol.reload()
            transcript = self.update_status_in_db(uid)
            
            # Define condition checker for audio file availability
            def check_audio_file(t: Transcript) -> Path | None:
                """Check if audio file exists for the given transcript."""
                if t.file_hash is None:
                    return None
                    
                source_type = "podcast" if transcript_is_podcast_episode(t) else "local"
                audiofile_path = get_valid_audio_path(
                    generate_file_id_from_hash(t.file_hash, source_type)
                )
                return audiofile_path
            
            # Retry logic to handle cases where file_hash might be updated asynchronously
            transcript, audiofile_path = retry_with_db_refetch(
                uid=uid,
                check_condition=check_audio_file,
                max_retries=5,
                initial_delay=1.0,
                logger=logger,
            )
            
            # Check audio file exists after retries
            if audiofile_path is None:
                source_type = "podcast" if transcript_is_podcast_episode(transcript) else "local"
                raise RuntimeError(
                    f"Audio file missing after retries -- "
                    f"source: {source_type}, file_hash: {transcript.file_hash}"
                )
            # Get speaker settings (ignored for Parakeet quick swap)
            # enable_speakers, num_speakers = get_speaker_settings(uid)
            # logger.info(f"Starting transcription for {uid} - Speakers: {enable_speakers}, Count: {num_speakers}")

            # Temporary: use Parakeet (no diarization)
            model = Parakeet()
            output_data, time_elapsed = model.transcribe.remote(
                str(uid),
                str(audiofile_path),
                enable_speakers=False,
                # num_speakers=num_speakers or 1,
            )
            transcript.transcript = output_data["text"]
            transcript.language = output_data["language"]
            update_or_create_transcripts(
                transcript,
                include_fields=["transcript", "language"],
                exclude_fields=["uid"],
                upsert=False,
            )

            logger.info(f"Transcription completed in {time_elapsed:.2f}s")

            # NEW: Spawn async embedding function (non-blocking)
            try:
                embedding_call = embed_transcript.spawn(str(uid))
                logger.info(
                    f"[TRANSCRIBING] Spawned async embedding job: {embedding_call.object_id}"
                )
            except Exception as embed_err:
                # Log error but don't fail the main pipeline
                logger.warning(
                    f"[TRANSCRIBING] Failed to spawn embedding job for {uid}: {embed_err}"
                )
            
            await self._next_state().run_job(uid)
            return 0

        except Exception as err:
            logger.error(f"[{self.StateSymbol}] Error: {err}", exc_info=True)
            # Clean up files and set failed status
            await self.on_error(uid)
            return -1
