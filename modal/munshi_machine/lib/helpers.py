import logging
import time
from typing import Callable, TypeVar
from uuid import UUID

from munshi_machine.db.crud.transcript import (
    find_transcript_by_episode_guid,
    find_transcript_by_uid,
)
from munshi_machine.db.database import engine
from munshi_machine.models.private.transcript import Transcript
from sqlmodel import Session

T = TypeVar("T")


def check_if_episode_exists(transcript: Transcript) -> bool:
    transcript = find_transcript_by_episode_guid(transcript.episode_guid)
    if transcript:
        return transcript
    return False


def transcript_is_podcast_episode(transcript: Transcript) -> bool:
    return True if transcript.podcast_id is not None else False


def retry_with_db_refetch(
    uid: UUID,
    check_condition: Callable[[Transcript], T | None],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    logger: logging.Logger | None = None,
) -> tuple[Transcript, T | None]:
    """
    Retry a condition check with exponential backoff, refetching the transcript from DB on each retry.

    Args:
        uid: The transcript UID to refetch
        check_condition: A function that takes a Transcript and returns a result or None if condition not met
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds between retries (default: 1.0)
        logger: Optional logger for warning messages

    Returns:
        A tuple of (latest_transcript, condition_result)

    Raises:
        ValueError: If transcript cannot be found in database
    """
    retry_delay = initial_delay
    transcript = None
    result = None

    # Initial fetch
    with Session(engine, expire_on_commit=False) as session:
        transcript = find_transcript_by_uid(uid, session)
        if transcript is None:
            raise ValueError(f"Transcript not found for uid: {uid}")

    for attempt in range(max_retries):
        # Check condition
        result = check_condition(transcript)

        if result is not None:
            break

        # If condition not met and retries remaining, wait and refetch
        if attempt < max_retries - 1:
            if logger:
                logger.warning(
                    f"Condition not met (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {retry_delay}s..."
                )
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

            # Refetch transcript from database
            with Session(engine, expire_on_commit=False) as session:
                refetched_transcript = find_transcript_by_uid(uid, session)
                if refetched_transcript:
                    transcript = refetched_transcript

    return transcript, result
