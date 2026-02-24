import time

from munshi_machine.lib.utils import output_handler
from munshi_machine.models.private.transcript import Transcript

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
async def migrate_transcriptions_to_db():

    import os

    from munshi_machine.db.crud.transcript import (
        update_or_create_episodes,
    )

    # loop through all files in the transcription volume
    transcripts = []
    for entry in os.scandir(config.TRANSCRIPTIONS_DIR):
        print(entry.path, entry.name)
        if "podcast" in entry.name:
            oh = output_handler(entry.name)
            oh.out_path = f"{config.TRANSCRIPTIONS_DIR}/{entry.name}"
            oh.get_output()
            try:
                transcript = get_transcript_from_json_file(oh.output)
                transcripts.append(transcript)
            except Exception:
                continue

    # write podcasts to db
    t1 = time.perf_counter()
    update_or_create_episodes(transcripts, upsert=True)
    t2 = time.perf_counter()
    print("DB write time:", t2 - t1)
    return t2 - t1


def get_transcript_from_json_file(output: dict) -> Transcript:
    import uuid
    from collections import defaultdict
    from operator import itemgetter

    (
        guid,
        title,
        url,
        description,
        author,
        ownerName,
        image,
        artwork,
        language,
        duration,
        published_date,
        data,
    ) = itemgetter(
        "podcastGuid",
        "title",
        "url",
        "description",
        "author",
        "ownerName",
        "cover",
        "artwork",
        "language",
        "duration",
        "published_date",
        "data",
    )(
        defaultdict(lambda: None, output)
    )

    t = Transcript(
        pi_guid=uuid.UUID(guid),
        title=title,
        url=url,
        image=image,
        language=language,
        artwork=artwork,
        duration=duration,
        date_published=published_date,
        transcript=data["text"],
        summary=data["summary_gemini"],
    )
    return t
