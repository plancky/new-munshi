import asyncio
import logging

from munshi_machine.core.volumes import audio_storage_vol
from munshi_machine.lib.helpers import transcript_is_podcast_episode
from munshi_machine.lib.id_utils import generate_file_id_from_hash
from munshi_machine.lib.processing_states.base import ProcessingState
from munshi_machine.lib.old_utils import get_valid_audio_path
from munshi_machine.models.status import TranscriptStatus

logger = logging.getLogger(__name__)


class CompletedProcessingState(ProcessingState):
    def __init__(self) -> None:
        self.StateSymbol = TranscriptStatus.COMPLETED

    def _next_state(self):
        return None

    async def run_job(self, uid: str) -> None:
        import os
        import time

        try:
            await asyncio.sleep(1)  # Simulate async operation
            transcript = self.update_status_in_db(uid)
            # Compute and persist simple end-to-end duration
            # from ..utils import output_handler
            # oh = output_handler(uid)
            # start_ts = oh.output.get("job_started_at_unix")
            # if start_ts:
            #     duration_sec = max(0.0, time.time() - float(start_ts))
            #     oh.output["duration_sec"] = duration_sec
            #     oh.write_transcription_data()
            start_ts = transcript.job_start_time
            duration_sec = max(0.0, time.time() - float(start_ts))
            logger.info(
                f"[E2E] job_complete uid={uid} duration_sec={duration_sec:.2f}s"
            )

            # Delete the audio file after processing is complete
            source_type = (
                "podcast" if transcript_is_podcast_episode(transcript) else "local"
            )
            audiofile_path = get_valid_audio_path(generate_file_id_from_hash(transcript.file_hash, source_type))
            # Check audio file exists
            if audiofile_path:
                try:
                    os.remove(audiofile_path)
                    logger.info(
                        f"[CompletedProcessingState] Deleted audio file {audiofile_path}"
                    )
                    # Ensure deletion is persisted in volume
                    audio_storage_vol.commit()
                except Exception as e:
                    logger.error(
                        f"[CompletedProcessingState] Error deleting audio file {audiofile_path}: {e}",
                        exc_info=True,
                    )
            return 0
        except Exception as err:
            await self.on_error(uid)
            logger.error(f"[{self.StateSymbol}] Error: {err}", exc_info=True)
            return
