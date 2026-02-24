import uuid
from fastapi import APIRouter, Request, responses, HTTPException
from munshi_machine.core import config
from munshi_machine.functions.embeddings import search_over_transcripts, TextEmbeddingsInference
from munshi_machine.db.crud.search import create_search, get_search_by_uid

logger = config.get_logger("API.v1.search")
router = APIRouter()


@router.post("/share_search")
async def share_search_endpoint(request: Request):
    """
    Save search results to be shared via a permanent link.
    """
    try:
        payload = await request.json()
    except Exception:
        return responses.JSONResponse(
            content={"error": "Invalid JSON payload"},
            status_code=400
        )
    
    query = payload.get("query")
    results = payload.get("results")
    answer = payload.get("answer")
    timing = payload.get("timing")
    podcast_id = payload.get("podcast_id")
    collection_id = payload.get("collection_id")
    
    if not query or not results:
        return responses.JSONResponse(
            content={"error": "Missing required fields: query and results"},
            status_code=400
        )
    
    try:
        # Step 1: Generate embedding for the query (backend-only)
        logger.info(f"[SHARE_API] Generating embedding for query: '{query[:50]}...' ")
        embedding_input = [("query", "", "", query)]
        model = TextEmbeddingsInference()
        _, embeddings = model.embed.remote(embedding_input)
        query_embedding = embeddings[0].tolist()
        
        # Step 2: Convert string IDs to UUIDs if provided
        p_uid = uuid.UUID(podcast_id) if podcast_id else None
        c_uid = uuid.UUID(collection_id) if collection_id else None
        
        # Step 3: Create search record
        shared_search = create_search(
            query=query,
            results=results,
            timing=timing,
            query_embedding=query_embedding,
            answer=answer,
            podcast_id=p_uid,
            collection_id=c_uid,
        )
        
        return responses.JSONResponse(
            content={
                "success": True,
                "uid": str(shared_search.uid)
            },
            status_code=201
        )
    except Exception as e:
        logger.error(f"[SHARE_API] Error creating shared search: {str(e)}")
        return responses.JSONResponse(
            content={"error": "Failed to create shared search", "detail": str(e)},
            status_code=500
        )


@router.get("/shared_search/{search_uid}")
async def get_shared_search_endpoint(search_uid: str):
    """
    Retrieve a shared search by its UUID.
    """
    try:
        uid = uuid.UUID(search_uid)
        shared_search = get_search_by_uid(uid)
        
        if not shared_search:
            return responses.JSONResponse(
                content={"error": "Shared search not found"},
                status_code=404
            )
        
        return responses.JSONResponse(
            content={
                "success": True,
                "query": shared_search.query,
                "answer": shared_search.answer,
                "results": shared_search.results,
                "timing": shared_search.timing,
                "podcast_id": str(shared_search.podcast_id) if shared_search.podcast_id else None,
                "collection_id": str(shared_search.collection_id) if shared_search.collection_id else None,
                "created_on": shared_search.created_on.isoformat(),
                "created_at": shared_search.created_at
            },
            status_code=200
        )
    except ValueError:
        return responses.JSONResponse(
            content={"error": "Invalid search_uid format"},
            status_code=400
        )
    except Exception as e:
        logger.error(f"[SHARE_API] Error retrieving shared search: {str(e)}")
        return responses.JSONResponse(
            content={"error": "Failed to retrieve shared search", "detail": str(e)},
            status_code=500
        )


@router.post("/search_podcast")
async def search_podcast_endpoint(request: Request):
    """
    Semantic search within a specific podcast's or collection's episodes.
    
    Searches across all COMPLETED transcripts with embeddings in the target podcast or collection.
    Uses vector similarity search to find the most relevant text chunks.
    
    Request body:
        {
            "query": "search query text",
            "podcast_uid": "uuid-of-podcast",  // Optional if collection_uid provided
            "collection_uid": "uuid-of-collection", // Optional if podcast_uid provided
            "limit": 10  // optional, defaults to 10
        }
    
    Response (Success):
        {
            "success": true,
            "query": "search query text",
            "podcast_uid": "uuid" (or null),
            "collection_uid": "uuid" (or null),
            "title": "Podcast/Collection Name",
            "author": "Author Name",
            "results": [...],
            ...
        }
    """
    logger.info("[SEARCH_API] Received search request")
    
    try:
        payload = await request.json()
        logger.info(f"[SEARCH_API] Payload received: query length={len(payload.get('query', ''))}, podcast_uid={payload.get('podcast_uid')}, collection_uid={payload.get('collection_uid')}, limit={payload.get('limit', 10)}")
    except Exception as e:
        logger.warning(f"[SEARCH_API] Invalid JSON payload: {str(e)}")
        return responses.JSONResponse(
            content={"error": "Invalid JSON payload"},
            status_code=400
        )
    
    # Validate required fields
    query = payload.get("query")
    podcast_uid = payload.get("podcast_uid")
    collection_uid = payload.get("collection_uid")
    limit = payload.get("limit", 10)
    
    if not query:
        logger.warning("[SEARCH_API] Missing required field: query")
        return responses.JSONResponse(
            content={"error": "Missing required field: query"},
            status_code=400
        )
    
    if not podcast_uid and not collection_uid:
        logger.warning("[SEARCH_API] Missing required field: podcast_uid or collection_uid")
        return responses.JSONResponse(
            content={"error": "Missing required field: podcast_uid or collection_uid"},
            status_code=400
        )
    
    if podcast_uid and collection_uid:
        logger.warning("[SEARCH_API] Conflicting fields: provided both podcast_uid and collection_uid")
        return responses.JSONResponse(
            content={"error": "Cannot provide both podcast_uid and collection_uid. Please provide only one."},
            status_code=400
        )
    
    # Validate limit
    try:
        limit = int(limit)
        if limit < 1 or limit > 100:
            logger.warning(f"[SEARCH_API] Invalid limit value: {limit} (must be 1-100)")
            return responses.JSONResponse(
                content={"error": "limit must be between 1 and 100"},
                status_code=400
            )
    except (ValueError, TypeError):
        logger.warning(f"[SEARCH_API] Invalid limit type: {limit}")
        return responses.JSONResponse(
            content={"error": "limit must be a valid integer"},
            status_code=400
        )
    
    # Validate UIDs format
    try:
        if podcast_uid:
            uuid.UUID(podcast_uid)
        if collection_uid:
            uuid.UUID(collection_uid)
    except (ValueError, AttributeError) as e:
        logger.warning(f"[SEARCH_API] Invalid UID format: {str(e)}")
        return responses.JSONResponse(
            content={"error": "podcast_uid and collection_uid must be valid UUIDs"},
            status_code=400
        )
    
    logger.info(
        f"[SEARCH_API] Validation passed | "
        f"Query: '{query[:100]}...' | "
        f"Podcast: {podcast_uid} | "
        f"Collection: {collection_uid} | "
        f"Limit: {limit}"
    )
    
    try:
        logger.info("[SEARCH_API] Invoking Modal search_over_transcripts function")
        
        # Call Modal function
        result = search_over_transcripts.remote(
            query=query,
            podcast_uid=podcast_uid,
            collection_uid=collection_uid,
            limit=limit
        )
        
        logger.info(
            f"[SEARCH_API] Modal function returned | "
            f"Success: {result.get('success')} | "
            f"Results: {result.get('result_count', 0)}"
        )
        
        # Check if search was successful
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            
            # Determine appropriate status code based on error message
            if "not found" in error_msg.lower():
                status_code = 404
                logger.warning(f"[SEARCH_API] Target not found")
            elif "no transcripts with embeddings" in error_msg.lower():
                status_code = 422
                logger.warning(
                    f"[SEARCH_API] No eligible transcripts | "
                    f"Stats: {result.get('stats', {})}"
                )
            elif "invalid" in error_msg.lower():
                status_code = 400
                logger.warning(f"[SEARCH_API] Invalid input: {error_msg}")
            else:
                status_code = 500
                logger.error(f"[SEARCH_API] Search error: {error_msg}")
            
            return responses.JSONResponse(
                content={
                    "error": error_msg,
                    "hint": result.get("hint"),
                    "stats": result.get("stats")
                },
                status_code=status_code
            )
        
        # Log success metrics
        target_name = result.get('podcast_title') or result.get('collection_title') or result.get('title')
        logger.info(
            f"[SEARCH_API] Search successful | "
            f"Target: {target_name} | "
            f"Results: {result.get('result_count')} | "
            f"Eligible transcripts: {result.get('eligible_transcript_count')} | "
            f"Duration: {result.get('search_duration_sec', 0):.3f}s"
        )
        
        # Return successful results
        return responses.JSONResponse(
            content=result,
            status_code=200
        )
        
    except Exception as e:
        logger.error(
            f"[SEARCH_API] Unexpected error during search | "
            f"Query: '{query[:50]}...' | "
            f"Error: {str(e)}",
            exc_info=True
        )
        return responses.JSONResponse(
            content={
                "error": "Search failed due to internal error",
                "detail": str(e)
            },
            status_code=500
        )
