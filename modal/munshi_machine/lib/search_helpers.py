from uuid import UUID
from sqlmodel import Session
from munshi_machine.db.database import engine
from munshi_machine.db.crud.transcript import get_podcast_by_uid
from munshi_machine.db.crud.collection import get_collection_by_uid
from munshi_machine.models.status import TranscriptStatus
from munshi_machine.core import config

logger = config.get_logger("search_helpers")

def get_eligible_transcripts_for_podcast(podcast_uid: str):
    """
    Fetches a podcast and filters its transcripts for those that are COMPLETED
    and have embeddings generated.
    
    Args:
        podcast_uid: UUID string of the podcast
        
    Returns:
        dict: Result dictionary with success status and either data or error message
    """
    try:
        podcast_uuid = UUID(podcast_uid)
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid podcast_uid format: {str(e)}"
        }

    with Session(engine, expire_on_commit=False) as session:
        podcast = get_podcast_by_uid(podcast_uuid, session)
        
        if not podcast:
            return {
                "success": False,
                "error": f"Podcast {podcast_uid} not found"
            }
        
        # Filter transcripts: COMPLETED + embeddings_generated = True
        eligible_transcripts = [
            t for t in podcast.transcripts
            if t.status == TranscriptStatus.COMPLETED 
            and t.embeddings_generated
        ]
        
        completed_count = sum(1 for t in podcast.transcripts if t.status == TranscriptStatus.COMPLETED)
        with_embeddings_count = sum(1 for t in podcast.transcripts if t.embeddings_generated)
        
        if len(eligible_transcripts) == 0:
            return {
                "success": False,
                "error": "No transcripts with embeddings found for this podcast",
                "stats": {
                    "total_transcripts": len(podcast.transcripts),
                    "completed_transcripts": completed_count,
                    "transcripts_with_embeddings": with_embeddings_count
                }
            }
        
        return {
            "success": True,
            "eligible_transcript_ids": [t.uid for t in eligible_transcripts],
            "podcast_title": podcast.title,
            "podcast_author": podcast.author,
            "stats": {
                "total_transcripts": len(podcast.transcripts),
                "completed_transcripts": completed_count,
                "transcripts_with_embeddings": with_embeddings_count,
                "eligible_count": len(eligible_transcripts)
            }
        }

def get_eligible_transcripts_for_collection(collection_uid: str):
    """
    Fetches a collection and filters its transcripts for those that are COMPLETED
    and have embeddings generated.
    
    Args:
        collection_uid: UUID string of the collection
        
    Returns:
        dict: Result dictionary with success status and either data or error message
    """
    try:
        collection_uuid = UUID(collection_uid)
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid collection_uid format: {str(e)}"
        }

    with Session(engine, expire_on_commit=False) as session:
        collection = get_collection_by_uid(collection_uuid, session)
        
        if not collection:
            return {
                "success": False,
                "error": f"Collection {collection_uid} not found"
            }
        
        # Filter transcripts: COMPLETED + embeddings_generated = True
        eligible_transcripts = [
            t for t in collection.transcripts
            if t.status == TranscriptStatus.COMPLETED 
            and t.embeddings_generated
        ]
        
        completed_count = sum(1 for t in collection.transcripts if t.status == TranscriptStatus.COMPLETED)
        with_embeddings_count = sum(1 for t in collection.transcripts if t.embeddings_generated)
        
        if len(eligible_transcripts) == 0:
            return {
                "success": False,
                "error": "No transcripts with embeddings found for this collection",
                "stats": {
                    "total_transcripts": len(collection.transcripts),
                    "completed_transcripts": completed_count,
                    "transcripts_with_embeddings": with_embeddings_count
                }
            }
        
        return {
            "success": True,
            "eligible_transcript_ids": [t.uid for t in eligible_transcripts],
            "collection_title": collection.title,
            "collection_author": collection.author,
            "stats": {
                "total_transcripts": len(collection.transcripts),
                "completed_transcripts": completed_count,
                "transcripts_with_embeddings": with_embeddings_count,
                "eligible_count": len(eligible_transcripts)
            }
        }
