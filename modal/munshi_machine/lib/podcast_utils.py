import hashlib
import os
import time
from typing import List
import uuid


from munshi_machine.models.private.podcast import Podcast
from munshi_machine.models.private.transcript import Transcript

API_BASE_URL = "https://api.podcastindex.org/api/1.0"

PODCAST_INDEX_API_KEY = os.environ.get("PODCAST_INDEX_API_KEY")
PODCAST_INDEX_API_SECRET = os.environ.get("PODCAST_INDEX_API_SECRET")


def generate_auth_headers(key: str, secret: str) -> dict:
    """
    Generate PodcastIndex-compatible auth headers.

    Args:
        key (str): The API key.
        secret (str): The API secret.
    Returns:
        dict: Auth header dictionary.
    """
    now = str(int(time.time()))
    hash_string = f"{key}{secret}{now}"
    hash_value = hashlib.sha1(hash_string.encode("utf-8")).hexdigest()
    return {
        "User-Agent": "Munshi",
        "X-Auth-Key": key,
        "X-Auth-Date": now,
        "Authorization": hash_value,
    }


async def fetch_podcast_by_guid(guid: str) -> Podcast:
    from httpx import AsyncClient

    """
    Fetch podcast metadata from PodcastIndex.org by GUID.

    Args:
        guid (str): The podcast GUID.
    Returns:
        httpx.Response: The full response object.
    """
    url = f"{API_BASE_URL}/podcasts/byguid"
    headers = generate_auth_headers(PODCAST_INDEX_API_KEY, PODCAST_INDEX_API_SECRET)
    params = {"guid": guid}
    async with AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        try:
            body: dict = response.json()
            return get_podcast_from_pi_feed(body.get("feed", None))
        except Exception:
            body = response.text
        return body


def get_podcast_from_pi_feed(feed: dict) -> Podcast:
    import uuid
    from collections import defaultdict
    from operator import itemgetter

    guid, title, url, description, author, ownerName, image, artwork, language = (
        itemgetter(
            "podcastGuid",
            "title",
            "url",
            "description",
            "author",
            "ownerName",
            "image",
            "artwork",
            "language",
        )(defaultdict(lambda: None, feed))
    )

    return Podcast(
        pi_guid=uuid.UUID(guid),
        title=title,
        url=url,
        description=description,
        author=author,
        ownerName=ownerName,
        image=image,
        language=language,
        artwork=artwork,
    )



async def fetch_episodes_by_podcast_guid(guid: str, uid: uuid.UUID | None, limit: int = 30, offset: int = 0) -> List[Transcript]:
    from httpx import AsyncClient

    """
    Fetch episodes for a podcast from PodcastIndex.org by GUID.

    Args:
        guid (str): The podcast GUID.
    Returns:
        Response: The full response object.
    """
    url = f"{API_BASE_URL}/episodes/bypodcastguid"
    headers = generate_auth_headers(PODCAST_INDEX_API_KEY, PODCAST_INDEX_API_SECRET)
    params = {"guid": guid, "max": 200}
    async with AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        try:
            body = response.json()
            items = body.get("items", [])
            return [get_episode_from_pi_feed(ep, podcast_id=uid) for ep in items[offset : offset + limit]]
        except Exception as e:
            body = response.text
        return body


async def fetch_episode_by_guid(
    guid: str,
    podcast_id: uuid.UUID | None = None,
    feed_id: str | int | None = None,
    feed_guid: str | None = None,
) -> tuple[Transcript | None, str | None]:
    from httpx import AsyncClient

    """
    Fetch a single episode from PodcastIndex.org by GUID.
    Returns (Transcript object, podcast_guid_string)
    """
    url = f"{API_BASE_URL}/episodes/byguid"
    headers = generate_auth_headers(PODCAST_INDEX_API_KEY, PODCAST_INDEX_API_SECRET)
    params = {"guid": guid}
    if feed_id:
        params["feedid"] = str(feed_id)
    if feed_guid:
        params["feedguid"] = feed_guid

    async with AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        body = response.json()
        ep_data = body.get("episode")
        if not ep_data:
            return None, None

        podcast_guid = ep_data.get("podcastGuid")
        return get_episode_from_pi_feed(ep_data, podcast_id=podcast_id), podcast_guid


def get_episode_from_pi_feed(feed: dict, podcast_id: uuid.UUID) -> Transcript:
    from collections import defaultdict
    from operator import itemgetter

    (
        guid,
        podcast_guid,
        title,
        url,
        description,
        image,
        artwork,
        language,
        date_published,
        season,
        episode,
        duration,
    ) = itemgetter(
        "guid",
        "podcastGuid",
        "title",
        "enclosureUrl",
        "description",
        "image",
        "artwork",
        "language",
        "datePublished",
        "season",
        "episode",
        "duration",
    )(
        defaultdict(lambda: None, feed)
    )

    return Transcript(
        episode_guid=guid,
        podcast_id=podcast_id,
        title=title,
        url=url,
        audio_url=url,
        description=description,
        image=image,
        language=language,
        artwork=artwork,
        season=season,
        episode=episode,
        duration=duration,
        date_published=date_published,
    )
