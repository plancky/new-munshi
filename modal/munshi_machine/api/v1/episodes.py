from fastapi import APIRouter, Request, responses
from modal import FunctionCall
from uuid import UUID

from munshi_machine.core import config
from munshi_machine.db.crud.transcript import (
    create_transcript,
    find_transcript_by_episode_guid,
    find_transcript_by_uid,
    update_or_create_podcasts,
)
from munshi_machine.db.crud.vector_store import delete_vectors_by_transcript_id
from munshi_machine.db.database import engine
from sqlmodel import Session
from munshi_machine.functions.init_transcription import init_transcription
from munshi_machine.functions.embeddings import embed_transcript
from munshi_machine.lib.podcast_utils import fetch_podcast_by_guid
from munshi_machine.models.private.transcript import Transcript
from munshi_machine.models.status import TranscriptStatus

logger = config.get_logger("API.v1.episodes")
router = APIRouter()


@router.post("/process_episode")
async def process_episode(request: Request):
    payload = await request.json()
    episode_guid = payload.get("guid")
    podcast_guid = payload.get("podcast_guid")
    audio_url = payload.get("audio_url")
    episode_title = payload.get("episode_title")
    podcast_title = payload.get("podcast")
    author = payload.get("author")
    cover = payload.get("cover")
    feed_id = payload.get("feed_id")
    episode_id = payload.get("episode_id")
    duration_seconds = payload.get("duration")
    published_date = payload.get("published_date")
    reprocess = bool(payload.get("reprocess", False))
    if not audio_url or not isinstance(audio_url, str):
        return responses.PlainTextResponse(
            content="bad request. audio_url missing", status_code=400
        )

    try:
        with Session(engine, expire_on_commit=False) as session:
            # checks if a record with same episode_guid exists in the db
            transcript = find_transcript_by_episode_guid(episode_guid, session)

        # if the transcript record exists on the db
        # return if
        # there is no reprocess flag set and is in terminated state 
        # or
        # a modal function call is still processing
        if transcript:
            ongoing_modal_function_call = False
            has_completed = transcript.status == TranscriptStatus.COMPLETED
            has_failed = transcript.status == TranscriptStatus.FAILED

            in_terminated_state = has_failed or has_completed
            if not in_terminated_state:
                try:
                    existing_call = FunctionCall.from_id(
                        transcript.init_function_call_id
                    )
                    existing_call.get(timeout=0)
                except TimeoutError:
                    ongoing_modal_function_call = True

            # if the transcript has the status of completed
            if (in_terminated_state and not reprocess) or ongoing_modal_function_call:
                return responses.JSONResponse(
                    content={"status": "transcript_exists", "id": str(transcript.uid)},
                    status_code=200,
                )

        # fetch podcast metadata
        podcast_obj = await fetch_podcast_by_guid(podcast_guid)
        # update podcast object with upsert
        podcast, has_modified = update_or_create_podcasts(
            podcast_obj, exclude_fields=["status", "uid"], upsert=True
        )

        with Session(engine, expire_on_commit=False) as session:
            episode = Transcript(
                episode_guid=episode_guid,
                podcast_id=podcast.uid,
                title=episode_title,
                published_date=published_date,
                duration=duration_seconds,
                audio_url=audio_url,
                image=cover,
            )
            # otherwise create a new record and initate a the process from init.
            call = init_transcription.spawn(episode.uid)
            episode.init_function_call_id = call.object_id

            episode = create_transcript(episode, session)


        return responses.JSONResponse(
            content={
                "status": "processing",
                "id": str(episode.uid),
                "call_id": call.object_id,
            },
            status_code=200,
        )

    except Exception as e:
        logger.error(f"process_episode failed: {e}", exc_info=True)
        return responses.JSONResponse(
            content={"error": "Failed to process episode"}, status_code=500
        )


@router.post("/reembed_episode")
async def reembed_episode(request: Request):
    """
    Regenerate embeddings for a completed transcript.
    
    Expected payload:
        {
            "episode_uid": "uuid-string"
        }
    
    Returns:
        - 200: Embedding job spawned successfully
        - 400: Invalid request or episode not found
        - 500: Server error
    """
    try:
        payload = await request.json()
        episode_uid_str = payload.get("episode_uid")
        
        if not episode_uid_str:
            return responses.JSONResponse(
                content={"error": "episode_uid is required"},
                status_code=400
            )
        
        try:
            episode_uid = UUID(episode_uid_str)
        except (ValueError, AttributeError):
            return responses.JSONResponse(
                content={"error": "Invalid episode_uid format"},
                status_code=400
            )
        
        with Session(engine, expire_on_commit=False) as session:
            # Find transcript
            transcript = find_transcript_by_uid(episode_uid, session)
            
            if not transcript:
                return responses.JSONResponse(
                    content={"error": "Episode not found"},
                    status_code=404
                )
            
            # Check if transcript is completed
            if transcript.status != TranscriptStatus.COMPLETED:
                return responses.JSONResponse(
                    content={
                        "error": "Episode must be in COMPLETED status to regenerate embeddings",
                        "current_status": transcript.status
                    },
                    status_code=400
                )
            
            # Check if transcript exists
            if not transcript.transcript:
                return responses.JSONResponse(
                    content={"error": "No transcript text available for embedding"},
                    status_code=400
                )
            
            # Reset embeddings_generated flag
            transcript.embeddings_generated = False
            session.add(transcript)
            session.commit()
            session.refresh(transcript)
            
            # Delete existing vectors
            delete_vectors_by_transcript_id(episode_uid, session)
            
            logger.info(f"Reset embeddings for episode {episode_uid}, spawning new embedding job")
        
        # Spawn new embedding job (outside session to avoid blocking)
        call = embed_transcript.spawn(str(episode_uid))
        
        return responses.JSONResponse(
            content={
                "status": "embedding_job_spawned",
                "episode_uid": str(episode_uid),
                "call_id": call.object_id
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"reembed_episode failed: {e}", exc_info=True)
        return responses.JSONResponse(
            content={"error": "Failed to regenerate embeddings"},
            status_code=500
        )


@router.post("/retry_transcript")
async def retry_transcript(request: Request):
    """
    Retry a failed transcription job.
    """
    try:
        payload = await request.json()
        uid_str = payload.get("uid")

        if not uid_str:
            return responses.JSONResponse(
                content={"error": "uid is required"},
                status_code=400
            )

        try:
            uid = UUID(uid_str)
        except (ValueError, AttributeError):
            return responses.JSONResponse(
                content={"error": "Invalid uid format"},
                status_code=400
            )

        with Session(engine, expire_on_commit=False) as session:
            transcript = find_transcript_by_uid(uid, session)
            if not transcript:
                return responses.JSONResponse(
                    content={"error": "Transcript not found"},
                    status_code=404
                )

        # Spawn init_transcription with restart=True
        call = init_transcription.spawn([uid], restart=True)

        return responses.JSONResponse(
            content={
                "status": "retry_started",
                "id": str(uid),
                "call_id": call.object_id,
            },
            status_code=200,
        )

    except Exception as e:
        logger.error(f"retry_transcript failed: {e}", exc_info=True)
        return responses.JSONResponse(
            content={"error": "Failed to retry transcription"},
            status_code=500
        )
