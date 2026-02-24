from munshi_machine.models.status import TranscriptStatus

from .completed import CompletedProcessingState
from .cleaning import CleaningProcessingState
from .fetching_audio import FetchingAudioProcessingState
from .init_job import InitProcessingState
from .base import FailedProcessingState
from .summarizing import SummarizingGeminiProcessingState
from .transcribing import TranscribingProcessingState
from .embedding import EmbeddingProcessingState


def processingStateFactory(symbol: TranscriptStatus):

    STATE_MAP = {
        TranscriptStatus.PENDING: InitProcessingState(),
        TranscriptStatus.PRE_PROCESSING: InitProcessingState(),
        TranscriptStatus.FETCHING_AUDIO: FetchingAudioProcessingState(),
        TranscriptStatus.TRANSCRIBING: TranscribingProcessingState(),
        TranscriptStatus.CLEANING: CleaningProcessingState(),
        TranscriptStatus.EMBEDDING: EmbeddingProcessingState(),
        TranscriptStatus.SUMMARIZING: SummarizingGeminiProcessingState(),
        TranscriptStatus.COMPLETED: CompletedProcessingState(),
        TranscriptStatus.FAILED: FailedProcessingState(),
    }

    return STATE_MAP.get(symbol, FailedProcessingState())


__all__ = [
    "SummarizingGeminiProcessingState",
    "TranscribingProcessingState",
    "CompletedProcessingState",
    "FetchingAudioProcessingState",
    "InitProcessingState",
    "FailedProcessingState",
    "EmbeddingProcessingState"
]
