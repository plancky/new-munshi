from fastapi import APIRouter, responses
from ...core import config
from ...core.volumes import transcriptions_vol
from ...lib.old_utils import get_embedding_coverage_stats

logger = config.get_logger("API.v1.stats")
router = APIRouter()


@router.get("/health")
async def health():
    return True


@router.get("/stats/transcripts")
async def transcripts_stats():
    try:
        import json as _json
        import pathlib

        transcriptions_vol.reload()
        base_dir = pathlib.Path(config.TRANSCRIPTIONS_DIR)
        count_completed = 0
        total_json = 0

        if base_dir.exists():
            for path in base_dir.iterdir():
                if not path.is_file():
                    continue
                if path.suffix != ".json":
                    continue
                if path.name == "episodes_index.json":
                    continue
                total_json += 1
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = _json.load(f)
                        status = data.get("status") or data.get("Status")
                        if status in ("Completed", "completed"):
                            count_completed += 1
                except Exception:
                    continue

        return responses.JSONResponse(content={"completed": count_completed, "total": total_json}, status_code=200)
    except Exception as e:
        logger.error(f"/stats/transcripts failed: {e}", exc_info=True)
        return responses.JSONResponse(content={"error": "Failed to compute stats"}, status_code=500)


@router.get("/stats/embedding_coverage")
async def embedding_coverage_stats():
    """
    Get statistics on embedding generation coverage.
    
    Returns:
        - total_completed: Total completed transcripts
        - with_embeddings: Transcripts with embeddings generated
        - pending_embedding: Transcripts waiting for embeddings
        - coverage_percent: Percentage with embeddings
        - total_chunks: Total vector chunks stored
        - avg_chunks_per_transcript: Average chunks per embedded transcript
    """
    try:
        stats = get_embedding_coverage_stats()
        return responses.JSONResponse(content=stats, status_code=200)
    except Exception as e:
        logger.error(f"/stats/embedding_coverage failed: {e}", exc_info=True)
        return responses.JSONResponse(
            content={"error": "Failed to compute embedding coverage stats"},
            status_code=500
        )
