import time

from munshi_machine.core import config
from munshi_machine.db.crud.transcript import update_or_create_episodes, update_or_create_transcripts
from munshi_machine.lib.processing_states.base import ProcessingState
from munshi_machine.models.status import TranscriptStatus

from .completed import CompletedProcessingState

logger = config.get_logger(__name__)


class SummarizingGeminiProcessingState(ProcessingState):
    def __init__(self) -> None:
        self.StateSymbol = TranscriptStatus.SUMMARIZING
        self._next_state_obj = CompletedProcessingState()

    def _next_state(self):
        return self._next_state_obj

    async def run_job(self, uid: str) -> None:
        from ..gemini import processor

        try:
            transcript = self.update_status_in_db(uid)

            summarize_start = time.time()
            # Load job start to compute end-to-end time if available
            e2e_start_unix = transcript.job_start_time
            try:
                logger.info(f"Starting summary generation for video {uid}")
                # Use the processor directly and await it properly
                metadata = {}
                if transcript.podcast is not None:
                    metadata["author"] = transcript.podcast.author
                    metadata["podcast"] = transcript.podcast.title
                    metadata["title"] = transcript.title

                summary_response = await processor.get_summary(
                    uid,
                    transcript.transcript,
                    metadata
                )
                
                # Temporarily store as JSON in summary field until DB migration
                # Format: {"summary": "<html>", "insights": [...], "tangents": [...]}
                import json
                summary_data = {
                    "summary": summary_response.summary,
                    "insights": [{"text": insight.text} for insight in summary_response.insights],
                    "tangents": [{"text": tangent.text} for tangent in summary_response.tangents]
                }
                transcript.summary = json.dumps(summary_data)
                
                update_or_create_transcripts(
                    transcript,
                    include_fields=["summary"],
                    exclude_fields=["uid"],
                    upsert=False,
                )
                logger.info(f"Summary generated successfully for audio {uid}")
                logger.info(f"Extracted {len(summary_response.insights)} insights and {len(summary_response.tangents)} tangents")
                
                # Spawn async job to embed insights (non-blocking, tangents not embedded)
                if len(summary_response.insights) > 0:
                    from munshi_machine.functions.embeddings import embed_insights
                    try:
                        insight_embedding_call = embed_insights.spawn(str(uid))
                        logger.info(f"Spawned insight embedding job for {uid} (call_id: {insight_embedding_call.object_id})")
                    except Exception as embed_err:
                        logger.error(f"Failed to spawn insight embedding job for {uid}: {embed_err}")
            except Exception as e:
                logger.error(f"Error generating summary for audio {uid}: {e}")
                # Continue to next state even if summary fails

            # Legacy elapsed log only
            summarize_elapsed = time.time() - summarize_start
            logger.info(f"Summarization finished in {summarize_elapsed:.2f}s")
            # End-to-end elapsed
            if e2e_start_unix:
                e2e_elapsed = time.time() - float(e2e_start_unix)
                logger.info(f"End-to-end processing time for {uid}: {e2e_elapsed:.2f}s")

            await self._next_state().run_job(uid)

        except Exception as err:
            logger.error(f"[{self.StateSymbol}] Error: {err}", exc_info=True)
            # Clean up files and set failed status
            await self.on_error(uid)
            return -1

        return 0
