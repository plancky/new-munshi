from urllib.parse import urlparse

from modal import method

from munshi_machine.lib.download_audio import _is_public_http_url, download_podcast

from ..core.app import app
from ..core.config import RAW_AUDIO_DIR, TRANSCRIPTIONS_DIR, get_logger
from ..core.images import base_image
from ..core.volumes import audio_storage_vol, transcriptions_vol

logger = get_logger(__name__)


@app.cls(
    image=base_image,
    volumes={
        str(RAW_AUDIO_DIR): audio_storage_vol,
        str(TRANSCRIPTIONS_DIR): transcriptions_vol,
    },
    timeout=600,
)
class Downloader:
    @method()
    def download_podcast(self, audio_url):
        return download_podcast(audio_url)

    def _is_public_http_url(self, url: str) -> bool:
        return _is_public_http_url(url)
