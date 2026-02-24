import datetime
import json
import pathlib
from typing import Any, Dict, List, Literal

from munshi_machine.core import config
from munshi_machine.core.config import RAW_AUDIO_DIR, TRANSCRIPTIONS_DIR

logger = config.get_logger("UTILS")


def get_valid_audio_path(file_id: str) -> pathlib.Path | None:
    """Find audio file with any supported extension."""
    for ext in ["mp3", "wav", "m4a", "mp4", "mov", "avi"]:
        path = pathlib.Path(RAW_AUDIO_DIR, get_audiofile_name(file_id, ext))
        if path.exists():
            return path
    # path does not exist
    return None


def get_audiofile_name(
    name: str, ext: Literal["mp3", "wav", "m4a", "mp4", "mov", "avi"]
):
    return f"{name}.{ext}"


def updateOutputJson(vid: str, symbol: str = None, data=None):
    outputHandler = output_handler(vid)
    if symbol != None:
        outputHandler.update_field("status", symbol)
    if data != None:
        outputHandler.update_field("data", data)
    outputHandler.write_transcription_data()


def updateOutputJsonDict(vid: str, fieldsDict):
    outputHandler = output_handler(vid)
    for fieldname in fieldsDict.keys():
        outputHandler.update_field(fieldname, fieldsDict[fieldname])
    outputHandler.write_transcription_data()


MUNSHI_TRANSCRIPTION_STATUS = {
    "initiated": "initiated",
    "transcribing": "transcribing",
    "completed": "completed",
    "failed": "failed",
}


class output_handler:
    def __init__(self, vid):
        self.vid = vid
        self.out_path = f"{TRANSCRIPTIONS_DIR}/{vid}.json"
        self.audio_path = str(get_valid_audio_path(vid))
        # Initialize status to None, will be set by get_output()
        self.status = None
        self.data = None
        self.output = {}
        self.get_output()

    def write_output_data(self, data):
        self.data = data
        self.write_transcription_data()

    def update_field(self, fieldname, value):
        self.output[fieldname] = value

    def get_metadata(self):
        return {
            "title": self.output.get("title"),
            "author": self.output.get("author"),
            "podcast": self.output.get("podcast"),
            "duration": self.output.get("duration"),
            "cover": self.output.get("cover"),
            "feed_id": self.output.get("feed_id"),
            "episode_id": self.output.get("episode_id"),
            "published_date": self.output.get("published_date"),
        }

    def write_transcription_data(self):
        with open(self.out_path, "w+", encoding="utf-8") as output_file:
            json.dump(
                self.output,
                output_file,
                ensure_ascii=False,
                indent=4,
            )
        # Ensure changes are visible across containers immediately
        try:
            from ..core.volumes import transcriptions_vol

            transcriptions_vol.commit()
        except Exception:
            pass

    def get_output(self):
        if not pathlib.Path(self.out_path).exists():
            self.output = {}
            self.status = "Not Found"  # Set status for missing files
            self.data = None
            return -1

        try:
            with open(self.out_path, "r", encoding="utf-8") as output_file:
                _output = json.load(output_file)
                self.status = _output.get("status", "Unknown")
                self.data = _output.get("data", None)
                self.output = _output
            return 0
        except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
            logger.error(f"Error reading transcript file for {self.vid}: {e}")
            self.output = {}
            self.status = "Failed"  # Set status for corrupted/invalid files
            self.data = None
            return -1


def store_speaker_settings(vid: str, enable_speakers: bool, num_speakers: int):
    """Store speaker settings for a given video ID."""
    from ..core.volumes import transcriptions_vol

    # Reload volume to ensure we have latest data
    transcriptions_vol.reload()

    outputHandler = output_handler(vid)
    speaker_settings = {
        "enable_speakers": enable_speakers,
        "num_speakers": num_speakers,
    }
    outputHandler.update_field("speaker_settings", speaker_settings)
    outputHandler.write_transcription_data()

    # Commit changes to volume immediately
    transcriptions_vol.commit()

    logger.info(f"Stored speaker settings for {vid}: {speaker_settings}")


def cleanup_failed_transcription(vid: str):
    """Clean up files when transcription fails"""
    import os
    import pathlib

    logger.info(f"Cleaning up failed transcription for {vid}")

    # Delete the transcript JSON file
    transcript_path = pathlib.Path(f"{TRANSCRIPTIONS_DIR}/{vid}.json")
    if transcript_path.exists():
        try:
            os.remove(transcript_path)
            logger.info(f"Deleted failed transcript file: {transcript_path}")
        except Exception as e:
            logger.error(f"Error deleting transcript file {transcript_path}: {e}")
        # Commit transcript deletion
        try:
            from ..core.volumes import transcriptions_vol

            transcriptions_vol.commit()
        except Exception:
            pass

    # Delete the audio file if it exists
    audio_path_obj = get_valid_audio_path(vid)
    if audio_path_obj.exists():
        try:
            os.remove(audio_path_obj)
            logger.info(f"Deleted audio file: {audio_path_obj}")
        except Exception as e:
            logger.error(f"Error deleting audio file {audio_path_obj}: {e}")
        # Commit audio deletion
        try:
            from ..core.volumes import audio_storage_vol

            audio_storage_vol.commit()
        except Exception:
            pass


def get_speaker_settings(vid: str):
    """Retrieve speaker settings for a given video ID."""
    from ..core.volumes import transcriptions_vol

    # Reload volume to ensure we have latest data
    transcriptions_vol.reload()

    outputHandler = output_handler(vid)
    speaker_settings = outputHandler.output.get("speaker_settings", {})

    # Default values if not found - default to False for safety
    enable_speakers = speaker_settings.get("enable_speakers", False)
    num_speakers = speaker_settings.get("num_speakers", 2)

    logger.info(
        f"Retrieved speaker settings for {vid}: enable={enable_speakers}, num={num_speakers}"
    )
    return enable_speakers, num_speakers


def _index_path() -> pathlib.Path:
    return pathlib.Path(TRANSCRIPTIONS_DIR, "episodes_index.json")


def _load_index() -> List[Dict[str, Any]]:
    from ..core.volumes import transcriptions_vol

    transcriptions_vol.reload()
    index_path = _index_path()
    if not index_path.exists():
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_index(entries: List[Dict[str, Any]]):
    from ..core.volumes import transcriptions_vol

    index_path = _index_path()
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    transcriptions_vol.commit()


def upsert_episode_index_entry(vid: str, metadata: Dict[str, Any]):
    """Upsert a compact record for this episode to power trending/recents.

    Stored fields: vid, podcast, title, duration, cover, author, feed_id, episode_id, published_date, last_seen
    """
    entries = _load_index()

    now_iso = datetime.datetime.utcnow().isoformat() + "Z"
    record = {
        "vid": vid,
        "podcast": metadata.get("podcast"),
        "title": metadata.get("title"),
        "duration": metadata.get("duration"),
        "cover": metadata.get("cover"),
        "author": metadata.get("author"),
        "feed_id": metadata.get("feed_id"),
        "episode_id": metadata.get("episode_id"),
        "published_date": metadata.get("published_date"),
        "last_seen": now_iso,
    }

    # Update if exists, else append
    updated = False
    for idx, existing in enumerate(entries):
        if existing.get("vid") == vid:
            entries[idx] = {**existing, **record}
            updated = True
            break
    if not updated:
        entries.append(record)

    # Keep index from growing unbounded
    # Sort by last_seen desc and trim to 1000 entries
    def _key(e):
        return e.get("last_seen", "")

    entries.sort(key=_key, reverse=True)
    entries = entries[:1000]

    _save_index(entries)


def get_recent_episodes(limit: int = 20) -> List[Dict[str, Any]]:
    entries = _load_index()
    # Already sorted by last_seen desc in upsert; sort again defensively
    entries.sort(key=lambda e: e.get("last_seen", ""), reverse=True)
    return entries[: max(1, min(limit, 100))]


def get_trending_podcasts(limit: int = 10) -> List[Dict[str, Any]]:
    """Aggregate recent index into unique podcasts ordered by recency/frequency.

    Strategy: count episodes per podcast and record most recent last_seen.
    Sort by count desc, then last_seen desc.
    """
    entries = _load_index()
    agg: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        podcast = e.get("podcast")
        if not podcast:
            continue
        item = agg.get(podcast)
        if not item:
            agg[podcast] = {
                "title": podcast,
                "author": e.get("author"),
                "cover": e.get("cover"),
                "feed_id": e.get("feed_id"),
                "count": 1,
                "last_seen": e.get("last_seen"),
            }
        else:
            item["count"] = item.get("count", 0) + 1
            # Update recency and backfill missing attributes if necessary
            if (e.get("last_seen") or "") > (item.get("last_seen") or ""):
                item["last_seen"] = e.get("last_seen")
                for k in ["author", "cover", "feed_id"]:
                    if e.get(k):
                        item[k] = e.get(k)

    sorted_items = sorted(
        agg.values(),
        key=lambda x: (x.get("count", 0), x.get("last_seen", "")),
        reverse=True,
    )
    # Map to stable shape
    result = [
        {
            "title": i.get("title"),
            "author": i.get("author"),
            "cover": i.get("cover"),
            "feed_id": i.get("feed_id"),
            "count": i.get("count"),
        }
        for i in sorted_items[: max(1, min(limit, 50))]
    ]
    return result


def get_embedding_coverage_stats(session=None) -> dict:
    """
    Get statistics on embedding coverage using embeddings_generated flag.
    
    Returns:
        dict with:
          - total_completed: Total completed transcripts
          - with_embeddings: Transcripts with embeddings_generated=true
          - coverage_percent: Percentage with embeddings
          - total_chunks: Total vector chunks in DB
          - pending_embedding: Transcripts waiting for embeddings
    """
    from uuid import UUID
    from sqlmodel import Session, select, func
    from munshi_machine.models.private.transcript import Transcript
    from munshi_machine.models.private.vector_store import VectorStore
    from munshi_machine.db.database import engine
    
    def execute(session):
        # Count completed transcripts
        total_stmt = (
            select(func.count(Transcript.uid))
            .where(Transcript.status == "COMPLETED")
        )
        total_completed = session.exec(total_stmt).one()
        
        # Count transcripts with embeddings_generated=true
        embedded_stmt = (
            select(func.count(Transcript.uid))
            .where(Transcript.status == "COMPLETED")
            .where(Transcript.embeddings_generated == True)
        )
        with_embeddings = session.exec(embedded_stmt).one()
        
        # Count pending (completed but no embeddings yet)
        pending_stmt = (
            select(func.count(Transcript.uid))
            .where(Transcript.status == "COMPLETED")
            .where(Transcript.embeddings_generated == False)
        )
        pending = session.exec(pending_stmt).one()
        
        # Count total vector chunks
        chunks_stmt = select(func.count(VectorStore.uid))
        total_chunks = session.exec(chunks_stmt).one()
        
        # Calculate average chunks per embedded transcript
        avg_chunks = (
            round(total_chunks / with_embeddings, 2) 
            if with_embeddings > 0 
            else 0
        )
        
        return {
            "total_completed": total_completed,
            "with_embeddings": with_embeddings,
            "pending_embedding": pending,
            "coverage_percent": round(with_embeddings / total_completed * 100, 2) if total_completed > 0 else 0,
            "total_chunks": total_chunks,
            "avg_chunks_per_transcript": avg_chunks
        }
    
    if session is not None:
        return execute(session)
    
    with Session(engine, expire_on_commit=False) as session:
        return execute(session)
