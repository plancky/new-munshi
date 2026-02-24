from typing import List
import uuid
from fastapi import APIRouter, HTTPException, Request, responses

from munshi_machine.core import config

from munshi_machine.db.crud.transcript import (
    find_transcript_by_uid, 
    find_transcripts_by_podcast_uid, 
    count_transcripts_by_podcast_uid,
    get_all_batches, 
    get_all_podcasts,
    get_batch_by_uid,
    get_podcast_by_uid,
    get_podcasts_by_batch_id,
    get_transcript_status_minimal
)
from munshi_machine.db.database import engine_ro as engine
from sqlmodel import Session

from .serializers import (
    serialize_batch,
    serialize_batches,
    serialize_podcast,
    serialize_podcasts,
    serialize_transcript,
    serialize_transcripts,
)

logger = config.get_logger("API.v1.fetch")
router = APIRouter()


@router.post("/fetch_data")
async def fetch_output(request: Request):
    import json
    
    payload = await request.json()
    try:
        uid = payload["vid"]
    except KeyError:
        return responses.PlainTextResponse(
            content="bad request. vid missing", status_code=400
        )

    try:
        with Session(engine, expire_on_commit=False) as session:
            transcript = find_transcript_by_uid(uuid.UUID(uid), session)

        if not transcript:
            raise HTTPException(status_code=404, detail="Transcript not found")
        
        metadata = {}
        if transcript.podcast is not None:
            metadata["author"] = transcript.podcast.author
            metadata["podcast"] = transcript.podcast.title
            metadata["title"] = transcript.title

        # Parse summary field - could be JSON or plain HTML
        summary_html = transcript.summary
        insights = []
        tangents = []
        
        if transcript.summary:
            try:
                # Try to parse as JSON (new format)
                summary_data = json.loads(transcript.summary)
                if isinstance(summary_data, dict):
                    summary_html = summary_data.get("summary", transcript.summary)
                    insights = summary_data.get("insights", [])
                    tangents = summary_data.get("tangents", [])
            except (json.JSONDecodeError, TypeError):
                # Old format - plain HTML string
                summary_html = transcript.summary

        status, data, metadata = (
            transcript.status,
            {
                "text": transcript.cleaned_transcript or transcript.transcript, 
                "summary_gemini": summary_html,
                "insights": insights,
                "tangents": tangents
            },
            metadata
        )

    except RuntimeError:
        return responses.JSONResponse(
            content={
                "error": "Could not find output log, You must initiate the transcription first."
            },
            status_code=406,
        )
    return responses.JSONResponse(
        content={"status": status, "data": data, "metadata": metadata}, status_code=200
    )


@router.post("/fetch_status")
async def fetch_status(request: Request):
    payload = await request.json()
    try:
        uid = payload["vid"]
    except KeyError:
        return responses.PlainTextResponse(
            content="bad request. vid missing", status_code=400
        )

    try:
        with Session(engine, expire_on_commit=False) as session:
            # minimal fetch returns a tuple: (uid, status, title, podcast_id)
            minimal = get_transcript_status_minimal(uuid.UUID(uid), session)

        if not minimal:
            raise HTTPException(status_code=404, detail="Transcript not found")
        
        status = minimal[1]
        metadata = {"title": minimal[2]}
        
        # If we need podcast author/title, we'd need another join or separate query.
        # Given we want to keep it minimal, let's see if the frontend can live with just the transcript title 
        # during polling. LoadingUI.tsx usually shows the status.
        
    except RuntimeError:
        return responses.JSONResponse(
            content={"error": "Something went wrong"},
            status_code=500,
        )
    
    return responses.JSONResponse(
        content={"status": status, "data": None, "metadata": metadata}, status_code=200
    )


@router.get("/fetch_podcasts")
def fetch_podcasts(batch_id: str | None = None):
    """
    Fetch all podcasts, optionally filtered by batch_id.
    Query params:
        - batch_id (optional): UUID of the batch to filter by
    """
    try:
        batch_uuid = uuid.UUID(batch_id) if batch_id else None
        with Session(engine, expire_on_commit=False) as session:
            podcasts = get_all_podcasts(batch_uuid, session)
        
        if not podcasts:
            raise HTTPException(status_code=404, detail="Podcasts not found")

    except ValueError:
        return responses.JSONResponse(
            content={"error": "Invalid batch_id format. Must be a valid UUID."},
            status_code=400,
        )
    except RuntimeError:
        return responses.JSONResponse(
            content={
                "error": "Could not fetch podcasts. Please try again."
            },
            status_code=500,
        )
    
    # Serialize podcasts with transcripts included
    podcasts_dict = serialize_podcasts(podcasts, include_transcripts=True)
    return responses.JSONResponse(content={"podcasts": podcasts_dict})


@router.get("/fetch_episodes")
def fetch_episodes(podcast_uid: str, offset: int = 0, limit: int = 50):
    """
    Fetch all episodes/transcripts for a specific podcast.
    Query params:
        - podcast_uid (required): UUID of the podcast
        - offset (optional): Pagination offset (default 0)
        - limit (optional): Pagination limit (default 50)
    """
    try:
        podcast_uuid = uuid.UUID(podcast_uid)
    except (ValueError, AttributeError):
        return responses.PlainTextResponse(
            content="bad request. podcast_uid is missing or invalid", status_code=400
        )

    try:
        with Session(engine, expire_on_commit=False) as session:
            total_count = count_transcripts_by_podcast_uid(podcast_uuid, session)
            episodes = find_transcripts_by_podcast_uid(
                podcast_uuid, offset=offset, limit=limit, session=session
            ) or []
        
        # We don't raise 404 here if list is empty because it might just be a valid page with no results
        # checking total_count > 0 might be better but empty list is valid JSON response for pagination

    except RuntimeError:
        return responses.JSONResponse(
            content={
                "error": "Could not fetch episodes. Please try again."
            },
            status_code=500,
        )
    
    # Serialize transcripts (episodes)
    transcripts_dict = serialize_transcripts(episodes, include_podcast=False)
    
    return responses.JSONResponse(content={
        "transcripts": transcripts_dict,
        "total_count": total_count,
        "offset": offset,
        "limit": limit
    })


@router.get("/fetch_batches")
def fetch_batches():
    """
    Fetch all batches with their associated podcasts.
    """
    try:
        with Session(engine, expire_on_commit=False) as session:
            batches = get_all_batches(session)
        
        if not batches:
            raise HTTPException(status_code=404, detail="Batches not found")

    except RuntimeError:
        return responses.JSONResponse(
            content={
                "error": "Something went wrong"
            },
            status_code=500,
        )
    
    # Serialize batches with podcasts included
    batches_dict = serialize_batches(batches, include_podcasts=True)
    return responses.JSONResponse(content={"batches": batches_dict})


@router.get("/fetch_batch_by_id")
def fetch_batch_by_id(batch_uid: str):
    """
    Fetch a single batch by its UID with all associated podcasts.
    Query params:
        - batch_uid (required): UUID of the batch
    """
    try:
        batch_uuid = uuid.UUID(batch_uid)
    except (ValueError, AttributeError):
        return responses.PlainTextResponse(
            content="bad request. batch_uid is missing or invalid", status_code=400
        )

    try:
        with Session(engine, expire_on_commit=False) as session:
            batch = get_batch_by_uid(batch_uuid, session)
        
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")

    except RuntimeError:
        return responses.JSONResponse(
            content={
                "error": "Could not fetch batch. Please try again."
            },
            status_code=500,
        )
    
    # Serialize batch with podcasts included
    batch_dict = serialize_batch(batch, include_podcasts=True)
    return responses.JSONResponse(content=batch_dict)


@router.get("/fetch_podcast_by_id")
def fetch_podcast_by_id(podcast_uid: str):
    """
    Fetch a single podcast by its UID with all associated episodes/transcripts.
    Query params:
        - podcast_uid (required): UUID of the podcast
    """
    try:
        podcast_uuid = uuid.UUID(podcast_uid)
    except (ValueError, AttributeError):
        return responses.PlainTextResponse(
            content="bad request. podcast_uid is missing or invalid", status_code=400
        )

    try:
        with Session(engine, expire_on_commit=False) as session:
            podcast = get_podcast_by_uid(podcast_uuid, session)
        
        if not podcast:
            raise HTTPException(status_code=404, detail="Podcast not found")

    except RuntimeError:
        return responses.JSONResponse(
            content={
                "error": "Could not fetch podcast. Please try again."
            },
            status_code=500,
        )
    
    # Serialize podcast with transcripts included
    podcast_dict = serialize_podcast(podcast, include_transcripts=False)
    return responses.JSONResponse(content=podcast_dict)
