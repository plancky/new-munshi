from fastapi import APIRouter, Request, responses
from uuid import UUID

from munshi_machine.core import config
from munshi_machine.db.database import engine
from sqlmodel import Session
from munshi_machine.db.crud.collection import (
    create_collection,
    get_collection_by_uid,
    remove_transcripts_from_collection,
    add_transcripts_to_collection,
)
from munshi_machine.db.crud.transcript import (
    find_transcript_by_episode_guid,
    find_transcript_by_file_hash,
    create_transcript,
    update_or_create_podcasts,
)
from munshi_machine.functions.init_transcription import init_transcription
from munshi_machine.lib.podcast_utils import (
    fetch_podcast_by_guid,
    fetch_episode_by_guid,
)
from munshi_machine.lib.id_utils import get_hash_from_file_id
from munshi_machine.models.private.transcript import Transcript
from munshi_machine.api.v1.serializers import serialize_collection

logger = config.get_logger("API.v1.collections")
router = APIRouter()


async def _process_items(episode_items, audio_file_ids):
    transcript_uids = []
    with Session(engine, expire_on_commit=False) as session:
        # 1. Process episode_items
        for item in episode_items:
            if isinstance(item, str):
                guid = item
                feed_id = None
            else:
                guid = item.get("guid")
                feed_id = item.get("feed_id")

            transcript = find_transcript_by_episode_guid(guid, session)
            if not transcript:
                # Fetch from Podcast Index
                ep_obj, podcast_guid = await fetch_episode_by_guid(
                    guid, feed_id=feed_id
                )
                if not ep_obj or not podcast_guid:
                    logger.warning(
                        f"Episode or podcast not found in Podcast Index: {guid}"
                    )
                    continue

                # Ensure podcast exists
                podcast_meta = await fetch_podcast_by_guid(podcast_guid)
                podcast, _ = update_or_create_podcasts(
                    podcast_meta, exclude_fields=["status", "uid"], upsert=True
                )

                # Create Transcript record
                ep_obj.podcast_id = podcast.uid

                # Spawn transcription
                call = init_transcription.spawn(ep_obj.uid)
                ep_obj.init_function_call_id = call.object_id

                transcript = create_transcript(ep_obj, session)

            if transcript:
                transcript_uids.append(transcript.uid)

        # 2. Process audio_file_ids
        for file_id in audio_file_ids:
            with Session(engine, expire_on_commit=False) as session_audiofile:
                file_hash = get_hash_from_file_id(file_id)
                transcript = find_transcript_by_file_hash(file_hash, session_audiofile)
                if not transcript:
                    # Create a minimal transcript record for the uploaded file
                    new_transcript = Transcript(
                        title=f"Uploaded File {file_hash[:8]}",
                        file_hash=file_hash,
                    )
                    transcript = create_transcript(new_transcript, session_audiofile)

                # Spawn transcription
                call = init_transcription.spawn(transcript.uid)
                transcript.init_function_call_id = call.object_id
                session_audiofile.commit()
                transcript_uids.append(transcript.uid)
    return transcript_uids


@router.post("/create_collection")
async def api_create_collection(request: Request):
    try:
        payload = await request.json()
        title = payload.get("title")
        description = payload.get("description")
        episode_guids = payload.get("episode_guids", [])
        audio_file_ids = payload.get("audio_file_ids", [])

        if not title:
            return responses.JSONResponse(
                content={"error": "title is required"}, status_code=400
            )

        transcript_uids = await _process_items(episode_guids, audio_file_ids)

        if len(transcript_uids) > config.COLLECTION_MAX_ITEMS:
            return responses.JSONResponse(
                content={
                    "error": f"Collection cannot exceed {config.COLLECTION_MAX_ITEMS} items"
                },
                status_code=400,
            )

        # 3. Create the collection
        with Session(engine, expire_on_commit=False) as session:
            collection = create_collection(
                title=title,
                description=description,
                transcript_ids=transcript_uids,
                session=session,
            )
            _ = collection.transcripts
            session.commit()

            return responses.JSONResponse(
                content=serialize_collection(collection), status_code=200
            )

    except Exception as e:
        logger.error(f"create_collection failed: {e}", exc_info=True)
        return responses.JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/collection/{uid}/items")
async def api_add_to_collection(uid: str, request: Request):
    try:
        payload = await request.json()
        episode_guids = payload.get("episode_guids", [])
        audio_file_ids = payload.get("audio_file_ids", [])

        transcript_uids = await _process_items(episode_guids, audio_file_ids)

        with Session(engine, expire_on_commit=False) as session:
            collection = get_collection_by_uid(UUID(uid), session)
            if not collection:
                return responses.JSONResponse(
                    content={"error": "Collection not found"}, status_code=404
                )

            # Check limit
            current_count = len(collection.transcripts)
            new_items_count = len(transcript_uids)
            if current_count + new_items_count > config.COLLECTION_MAX_ITEMS:
                return responses.JSONResponse(
                    content={
                        "error": f"Adding these items would exceed the {config.COLLECTION_MAX_ITEMS} item limit"
                    },
                    status_code=400,
                )

            collection = add_transcripts_to_collection(
                UUID(uid), transcript_uids, session
            )

            return responses.JSONResponse(
                content=serialize_collection(collection), status_code=200
            )
    except Exception as e:
        logger.error(f"add_to_collection failed: {e}", exc_info=True)
        return responses.JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/collection/{uid}")
async def api_get_collection(uid: str):
    try:
        with Session(engine, expire_on_commit=False) as session:
            collection = get_collection_by_uid(UUID(uid), session)
            if not collection:
                return responses.JSONResponse(
                    content={"error": "Collection not found"}, status_code=404
                )

            return responses.JSONResponse(
                content=serialize_collection(collection), status_code=200
            )
    except Exception as e:
        logger.error(f"get_collection failed: {e}", exc_info=True)
        return responses.JSONResponse(content={"error": str(e)}, status_code=500)


@router.delete("/collection/{uid}/items/{transcript_uid}")
async def api_remove_from_collection(uid: str, transcript_uid: str):
    try:
        with Session(engine, expire_on_commit=False) as session:
            collection = remove_transcripts_from_collection(
                UUID(uid), [UUID(transcript_uid)], session
            )
            if not collection:
                return responses.JSONResponse(
                    content={"error": "Collection not found"}, status_code=404
                )

            return responses.JSONResponse(
                content=serialize_collection(collection), status_code=200
            )
    except Exception as e:
        logger.error(f"remove_from_collection failed: {e}", exc_info=True)
        return responses.JSONResponse(content={"error": str(e)}, status_code=500)
