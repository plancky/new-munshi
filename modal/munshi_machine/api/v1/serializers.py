"""
Serializer functions for converting SQLModel objects to dictionaries.
These ensure that SQLAlchemy relationships are properly included in API responses.
"""
from typing import List, Dict, Any, Optional
from munshi_machine.models.private.batch import Batch
from munshi_machine.models.private.podcast import Podcast
from munshi_machine.models.private.transcript import Transcript
from munshi_machine.models.private.collection import Collection


def serialize_collection(collection: Collection, include_transcripts: bool = True) -> Dict[str, Any]:
    """
    Convert a Collection SQLModel object to a dictionary.
    """
    result = {
        "uid": str(collection.uid),
        "title": collection.title,
        "description": collection.description,
        "artwork": collection.artwork,
        "author": collection.author,
        "ownerName": collection.ownerName,
        "language": collection.language,
        "download_link": collection.download_link,
    }

    if include_transcripts and hasattr(collection, 'transcripts') and collection.transcripts:
        result["transcripts"] = [
            serialize_transcript(transcript, include_podcast=True)
            for transcript in collection.transcripts
        ]
    
    return result


def serialize_transcript_minimal(transcript: Transcript) -> Dict[str, Any]:
    """
    Convert a Transcript to a minimal dictionary (for counting only).
    Only includes uid and status - enough for episode count without heavy data.
    
    Args:
        transcript: The Transcript object to serialize
        
    Returns:
        Minimal dictionary with just uid and status
    """
    return {
        "uid": str(transcript.uid),
        "status": transcript.status,
    }


def serialize_transcript(transcript: Transcript, include_podcast: bool = False) -> Dict[str, Any]:
    """
    Convert a Transcript SQLModel object to a dictionary.
    
    Args:
        transcript: The Transcript object to serialize
        include_podcast: Whether to include the related podcast object
        
    Returns:
        Dictionary representation of the transcript
    """
    import json
    
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
    
    result = {
        "uid": str(transcript.uid),
        "episode_guid": transcript.episode_guid,
        "title": transcript.title,
        "url": transcript.url,
        "description": transcript.description,
        "image": transcript.image,
        "artwork": transcript.artwork,
        "season": transcript.season,
        "episode": transcript.episode,
        "language": transcript.language,
        "download_link": transcript.download_link,
        "duration": transcript.duration,
        "date_published": transcript.date_published,
        "audio_url": transcript.audio_url,
        "status": transcript.status,
        "transcript": transcript.transcript,
        "cleaned_transcript": transcript.cleaned_transcript,
        "summary": summary_html,
        "insights": insights,
        "tangents": tangents,
        "podcast_id": str(transcript.podcast_id) if transcript.podcast_id else None,
    }
    
    if include_podcast and hasattr(transcript, 'podcast') and transcript.podcast:
        result["podcast"] = serialize_podcast(transcript.podcast, include_transcripts=False)
    
    return result


def serialize_podcast(podcast: Podcast, include_transcripts: bool = True, transcripts_minimal: bool = False, include_batch: bool = False) -> Dict[str, Any]:
    """
    Convert a Podcast SQLModel object to a dictionary.
    
    Args:
        podcast: The Podcast object to serialize
        include_transcripts: Whether to include the related transcripts array
        transcripts_minimal: If True, only include minimal transcript data (uid, status) for counting
        include_batch: Whether to include the related batch object
        
    Returns:
        Dictionary representation of the podcast
    """
    result = {
        "uid": str(podcast.uid),
        "pi_guid": str(podcast.pi_guid) if podcast.pi_guid else None,
        "title": podcast.title,
        "url": podcast.url,
        "description": podcast.description,
        "image": podcast.image,
        "artwork": podcast.artwork,
        "author": podcast.author,
        "ownerName": podcast.ownerName,
        "language": podcast.language,
        "download_link": podcast.download_link,
        "date_published": podcast.date_published,
        "status": podcast.status,
        "batch_id": str(podcast.batch_id) if podcast.batch_id else None,
    }
    
    if include_transcripts and hasattr(podcast, 'transcripts'):
        if transcripts_minimal:
            # Only include minimal data for counting
            result["transcripts"] = [
                serialize_transcript_minimal(transcript)
                for transcript in podcast.transcripts
            ]
        else:
            # Include full transcript data
            result["transcripts"] = [
                serialize_transcript(transcript, include_podcast=False)
                for transcript in podcast.transcripts
            ]
    
    if include_batch and hasattr(podcast, 'batch') and podcast.batch:
        result["batch"] = serialize_batch(podcast.batch, include_podcasts=False)
    
    return result


def serialize_batch(batch: Batch, include_podcasts: bool = True, include_podcast_transcripts: bool = True, transcripts_minimal: bool = True) -> Dict[str, Any]:
    """
    Convert a Batch SQLModel object to a dictionary.
    
    Args:
        batch: The Batch object to serialize
        include_podcasts: Whether to include the related podcasts array
        include_podcast_transcripts: Whether to include transcripts in each podcast (for episode count)
        transcripts_minimal: If True, only include minimal transcript data (uid, status) - keeps payload light
        
    Returns:
        Dictionary representation of the batch
    """
    result = {
        "uid": str(batch.uid),
        "function_call_id": batch.function_call_id,
    }
    
    if include_podcasts and hasattr(batch, 'podcasts'):
        result["podcasts"] = [
            serialize_podcast(
                podcast, 
                include_transcripts=include_podcast_transcripts, 
                transcripts_minimal=transcripts_minimal,
                include_batch=False
            )
            for podcast in batch.podcasts
        ]
    
    return result


def serialize_transcripts(transcripts: List[Transcript], include_podcast: bool = False) -> List[Dict[str, Any]]:
    """
    Serialize a list of transcripts.
    
    Args:
        transcripts: List of Transcript objects
        include_podcast: Whether to include related podcast for each transcript
        
    Returns:
        List of transcript dictionaries
    """
    return [serialize_transcript(t, include_podcast=include_podcast) for t in transcripts]


def serialize_podcasts(podcasts: List[Podcast], include_transcripts: bool = True, include_batch: bool = False) -> List[Dict[str, Any]]:
    """
    Serialize a list of podcasts.
    
    Args:
        podcasts: List of Podcast objects
        include_transcripts: Whether to include related transcripts for each podcast
        include_batch: Whether to include related batch for each podcast
        
    Returns:
        List of podcast dictionaries
    """
    return [serialize_podcast(p, include_transcripts=include_transcripts, include_batch=include_batch) for p in podcasts]


def serialize_batches(batches: List[Batch], include_podcasts: bool = True, include_podcast_transcripts: bool = True, transcripts_minimal: bool = True) -> List[Dict[str, Any]]:
    """
    Serialize a list of batches.
    
    Args:
        batches: List of Batch objects
        include_podcasts: Whether to include related podcasts for each batch
        include_podcast_transcripts: Whether to include transcripts in each podcast (for episode count)
        transcripts_minimal: If True, only include minimal transcript data (uid, status) - keeps payload light
        
    Returns:
        List of batch dictionaries
    """
    return [serialize_batch(b, include_podcasts=include_podcasts, include_podcast_transcripts=include_podcast_transcripts, transcripts_minimal=transcripts_minimal) for b in batches]
