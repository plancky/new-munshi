from typing import List

from fastapi import APIRouter, responses
from pydantic import BaseModel

from munshi_machine.core import config

logger = config.get_logger("API.v1.podcasts")
router = APIRouter()


class PodcastBatch(BaseModel):
    podcast_guids: List[str]

@router.get("/podcasts/trending")
async def trending_podcasts(limit: int = 10):
    from ...core.volumes import transcriptions_vol
    from ...lib.utils import get_trending_podcasts

    try:
        transcriptions_vol.reload()
        items = get_trending_podcasts(limit=limit)
        return responses.JSONResponse(content={"podcasts": items}, status_code=200)
    except Exception as e:
        logger.error(f"trending_podcasts failed: {e}", exc_info=True)
        return responses.JSONResponse(
            content={"error": "Failed to fetch trending"}, status_code=500
        )
