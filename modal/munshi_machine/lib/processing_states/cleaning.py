import time

from munshi_machine.core import config
from munshi_machine.db.crud.transcript import update_or_create_episodes, update_or_create_transcripts
from munshi_machine.lib.processing_states.base import ProcessingState
from munshi_machine.models.status import TranscriptStatus

from .summarizing import SummarizingGeminiProcessingState

logger = config.get_logger(__name__)

class CleaningProcessingState(ProcessingState):
    def __init__(self) -> None:
        self.StateSymbol = TranscriptStatus.CLEANING
        self._next_state_obj = SummarizingGeminiProcessingState()

    def _next_state(self):
        return self._next_state_obj

    async def run_job(self, uid: str) -> None:
        from ..gemini import get_cleaned_transcript

        try:
            transcript = self.update_status_in_db(uid)

            start = time.time()
            logger.info(f"Starting cleaning transcript {uid}")
            if transcript.transcript:
                metadata = {}
                if transcript.podcast is not None:
                    metadata["author"] = transcript.podcast.author
                    metadata["podcast"] = transcript.podcast.title
                    metadata["title"] = transcript.title

                cleaned = await get_cleaned_transcript(
                    transcript_text=transcript.transcript,
                    metadata=metadata
                )
                # cleaned can be dict with cleaned_text list or a string
                if isinstance(cleaned, dict) and "cleaned_text" in cleaned:
                    transcript.cleaned_transcript = cleaned["cleaned_text"]
                else:
                    transcript.cleaned_transcript = cleaned

            # Clean speaker transcript only if present
            # if oh.data and oh.data.get("speaker_transcript"):
            #     metadata = oh.get_metadata()
            #     cleaned_speaker = await get_cleaned_speaker_transcript(oh.data["speaker_transcript"], metadata)
            #     if isinstance(cleaned_speaker, dict):
            #         if cleaned_speaker.get("cleaned_transcript"):
            #             oh.data["speaker_transcript"] = cleaned_speaker["cleaned_transcript"]
            #         if cleaned_speaker.get("speaker_mappings"):
            #             oh.data["speaker_mappings"] = cleaned_speaker["speaker_mappings"]
            #         oh.write_transcription_data()

            update_or_create_transcripts(
                transcript, include_fields=["transcript", "cleaned_transcript"], exclude_fields=["uid", "file_hash"], upsert=False
            )
            # Keep legacy elapsed log only
            elapsed = time.time() - start
            logger.info(f"Cleaning finished for audio {uid} in {elapsed:.2f}s")

            await self._next_state().run_job(uid)
        except Exception as err:
            await self.on_error(uid)
            logger.error(f"[{self.StateSymbol}] Error: {err}", exc_info=True)
            return -1

        return 0
