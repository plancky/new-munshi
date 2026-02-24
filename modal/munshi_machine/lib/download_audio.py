import hashlib
import ipaddress
import os
import pathlib
import time
from urllib.parse import urlparse

import blake3

from munshi_machine.core.config import RAW_AUDIO_DIR, get_logger
from munshi_machine.core.volumes import audio_storage_vol
from munshi_machine.lib.id_utils import (
    generate_file_id_from_hash,
)
from munshi_machine.lib.old_utils import get_audiofile_name

logger = get_logger("DOWNLOAD_AUDIO", 10)


def get_stored_audio(path: str):
    with open(path, "rb") as f:
        audio = f.read()
    return audio


def get_metadata(vid):
    try:
        from mutagen.easyid3 import EasyID3

        from .old_utils import get_valid_audio_path

        path = str(get_valid_audio_path(vid))
        audio = EasyID3(path)
        logger.info(f"Found ID3 tags: {audio}")

        # Safely extract metadata with fallbacks
        title = audio.get("title", [None])[0] if audio.get("title") else None
        artist = audio.get("artist", [None])[0] if audio.get("artist") else None

        return {"title": title, "author": artist}
    except Exception as e:
        logger.warning(f"Could not extract metadata from {vid}: {e}")
        return {"title": None, "author": None}


def download_podcast(audio_url: str):
    """
    Download audio from a URL into RAW_AUDIO_DIR.

    Returns (file_id, transcript_exists, saved_path_or_none)
    """
    import requests  # Import inside method per runtime constraints

    logger.info(f"Downloading audio: {audio_url}")

    # Stream download and compute hash
    hasher = blake3.blake3()
    try:
        # Basic SSRF guard and scheme validation
        if not _is_public_http_url(audio_url):
            raise RuntimeError("Invalid or non-public audio URL")

        attempts = 0
        last_exc = None
        while attempts < 3:
            try:
                with requests.get(audio_url, stream=True, timeout=30) as resp:
                    resp.raise_for_status()
                    ext = _infer_extension(audio_url, resp.headers.get("Content-Type")).strip(".")

                    temp_name = (
                        f"download_{hashlib.md5(audio_url.encode()).hexdigest()}"
                    )
                    temp_path = pathlib.Path(RAW_AUDIO_DIR, f"{temp_name}.downloading")

                    total_bytes = 0
                    with open(temp_path, "wb") as out:
                        for chunk in resp.iter_content(chunk_size=1 << 16):
                            if not chunk:
                                continue
                            out.write(chunk)
                            hasher.update(chunk)
                            total_bytes += len(chunk)
                break
            except Exception as e:
                last_exc = e
                attempts += 1
                if attempts < 3:
                    time.sleep(0.5 * attempts)
        if last_exc and attempts >= 3:
            raise last_exc

        if total_bytes == 0:
            raise RuntimeError("Downloaded file is empty")

        content_hash = hasher.hexdigest()
        file_id = generate_file_id_from_hash(content_hash, source_type="podcast")

        final_filename = get_audiofile_name(file_id, ext)
        final_path = pathlib.Path(RAW_AUDIO_DIR, final_filename) 

        logger.info("Renaming temp to final_path" f"{temp_path}" f"{final_path}")
        os.rename(temp_path, final_path)
        audio_storage_vol.commit()
        logger.info(f"Saved audio: {final_path}")

        return (content_hash, False, str(file_id))

    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        # Best-effort cleanup
        try:
            if "temp_path" in locals() and temp_path.exists():
                os.remove(temp_path)
        except Exception:
            pass
        raise


def _is_public_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname
        if not host:
            return False
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            # Not an IP, allow domain
            return True
        return not (
            ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
        )
    except Exception:
        return False


def _infer_extension(url: str, content_type: str | None) -> str:
    if content_type:
        ct = content_type.lower()
        if "audio/mpeg" in ct or "audio/mp3" in ct:
            return ".mp3"
        if "audio/mp4" in ct or "audio/x-m4a" in ct or "audio/aac" in ct:
            return ".m4a"
        if "audio/wav" in ct or "audio/x-wav" in ct:
            return ".wav"
        if "audio/ogg" in ct:
            return ".ogg"

    path = urlparse(url).path.lower()
    for ext in (".mp3", ".m4a", ".wav", ".mp4", ".mov", ".avi", ".ogg"):
        if path.endswith(ext):
            return ext
    return ".mp3"
