import asyncio
import time
import uuid

from modal import current_function_call_id

from munshi_machine.functions.init_transcription import init_transcription

from ..core import config
from ..core.app import app, custom_secret
from ..core.images import base_image
from ..core.volumes import (
    audio_storage_vol,
    transcriptions_vol,
)

logger = config.get_logger(__name__)


@app.function(
    image=base_image,
    volumes={
        str(config.RAW_AUDIO_DIR): audio_storage_vol,
        str(config.TRANSCRIPTIONS_DIR): transcriptions_vol,
    },
    secrets=[custom_secret],
    timeout=2000,
)
async def fetch_and_batch_transcribe_podcasts(guid_list: list[str], batch_uid: uuid.UUID):
    from typing import List

    from pydantic import BaseModel

    from munshi_machine.db.crud.transcript import (
        update_or_create_episodes,
        update_or_create_podcasts,
    )
    from munshi_machine.lib.podcast_utils import (
        fetch_episodes_by_podcast_guid,
        fetch_podcast_by_guid,
    )
    from munshi_machine.models.private.podcast import Podcast
    from munshi_machine.models.private.transcript import Transcript

    """
    Fetch podcasts by a list of GUIDs concurrently and return results in order.

    Args:
        guid_list (list): List of podcast GUIDs (strings)
    Returns:
        list: Ordered list of httpx.Response objects, aligned with input order
    """
    # fetch podcasts from podcast index
    tasks = [fetch_podcast_by_guid(guid) for guid in guid_list]
    podcasts: List[Podcast] = await asyncio.gather(*tasks)
    function_call_id = current_function_call_id()
    for podcast in podcasts:
        podcast.batch_id = batch_uid
    # write podcasts to db
    t1 = time.perf_counter()
    podcasts, _modfied = update_or_create_podcasts(podcasts, exclude_fields=["status", "uid"], upsert=True)
    podcasts = podcasts if isinstance(podcasts,list) else [podcasts]
    t2 = time.perf_counter()
    print("DB write time:", t2 - t1)

    tasks = [fetch_episodes_by_podcast_guid(podcast.pi_guid, podcast.uid, limit=100) for podcast in podcasts]
    transcripts: List[List[Transcript]] = await asyncio.gather(*tasks)
    transcripts = transcripts if isinstance(transcripts,list) else [transcripts]
    flattened_transcripts = [ep for sublist in transcripts for ep in sublist]

    # write episodes to db
    t1 = time.perf_counter()
    episodes, _modified = update_or_create_episodes(
        flattened_transcripts,
        exclude_fields=["status", "transcript", "cleaned_transcript", "summary", "uid", "file_hash"],
        upsert=True,
    )
    t2 = time.perf_counter()
    print("DB transcripts time:", t2 - t1)
    
    await init_transcription.spawn_map.aio([transcript.uid for transcript in episodes], [True for transcript in episodes])

    return [
        result.model_dump() if isinstance(result, BaseModel) else result
        for result in flattened_transcripts
    ]