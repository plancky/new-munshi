"""
Whispers API endpoints - Insight cards and pattern discovery
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class GenerateCardsRequest(BaseModel):
    """Request to generate insight cards"""
    user_id: str | None = None


class GenerateCardsResponse(BaseModel):
    """Response from card generation"""
    success: bool
    cards_generated: int
    insights_processed: int
    total_insights: int
    duration_sec: float
    message: str | None = None


@router.post("/generate_cards", response_model=GenerateCardsResponse)
async def generate_cards_endpoint(request: GenerateCardsRequest):
    """
    Generate insight cards by analyzing patterns across all insights.
    
    This endpoint:
    1. Loads all insight vectors
    2. Processes them sequentially
    3. Finds similar insights
    4. Generates cards (Echo, Bridge, Fracture)
    5. Stores cards for retrieval
    
    Note: This is an expensive operation and should be run periodically,
    not on every request.
    """
    from munshi_machine.functions.generate_cards import generate_all_cards
    
    try:
        result = await generate_all_cards.remote.aio(user_id=request.user_id)
        
        return GenerateCardsResponse(
            success=result["success"],
            cards_generated=result.get("cards_generated", 0),
            insights_processed=result.get("insights_processed", 0),
            total_insights=result.get("total_insights", 0),
            duration_sec=result.get("duration_sec", 0),
            message=result.get("message")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cards")
async def get_cards_endpoint(
    card_type: str | None = None,
    limit: int = 100
):
    """
    Fetch generated insight cards.
    
    Args:
        card_type: Filter by type ("echo", "bridge", "fracture", or None for all)
        limit: Maximum number of cards to return
    
    Returns:
        dict: Cards with metadata, sorted by quality score
    """
    from munshi_machine.core import config
    from munshi_machine.db.database import engine
    from munshi_machine.models.private.insight_card import InsightCard, CardType
    from sqlmodel import Session, select
    
    logger = config.get_logger(__name__)
    
    try:
        logger.info(f"[WHISPERS] Fetching cards | Type: {card_type} | Limit: {limit}")
        
        with Session(engine, expire_on_commit=False) as session:
            query = (
                select(InsightCard)
                .order_by(InsightCard.quality_score.desc(), InsightCard.created_at.desc())
            )
            
            # Filter by type if specified
            if card_type:
                try:
                    type_enum = CardType(card_type)
                    query = query.where(InsightCard.type == type_enum)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid card type: {card_type}. Must be echo, bridge, or fracture"
                    )
            
            query = query.limit(limit)
            cards = session.exec(query).all()
            
            # Format response
            results = [
                {
                    "uid": str(card.uid),
                    "type": card.type,
                    "title": card.title,
                    "content": card.content,
                    "source_count": card.source_count,
                    "quality_score": card.quality_score,
                    "created_at": card.created_at
                }
                for card in cards
            ]
            
            logger.info(f"[WHISPERS] Fetched {len(results)} cards")
            
            return {
                "success": True,
                "cards": results,
                "count": len(results)
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[WHISPERS] Error fetching cards: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_whispers_stats():
    """
    Get statistics about insights and cards.
    
    Returns counts of:
    - Total insights
    - Total transcripts with insights
    - Total cards generated
    - Average quality score
    """
    from munshi_machine.core import config
    from munshi_machine.db.database import engine
    from munshi_machine.models.private.insight_vector import InsightVector
    from munshi_machine.models.private.insight_card import InsightCard
    from munshi_machine.models.private.transcript import Transcript
    from sqlmodel import Session, select, func
    
    logger = config.get_logger(__name__)
    
    try:
        with Session(engine, expire_on_commit=False) as session:
            total_insights = session.exec(
                select(func.count(InsightVector.uid))
            ).one()
            
            transcripts_with_insights = session.exec(
                select(func.count(func.distinct(InsightVector.transcript_id)))
            ).one()
            
            total_transcripts = session.exec(
                select(func.count(Transcript.uid))
            ).one()
            
            total_cards = session.exec(
                select(func.count(InsightCard.uid))
            ).one()
            
            avg_quality = session.exec(
                select(func.avg(InsightCard.quality_score))
            ).one()
            
            logger.info(
                f"[WHISPERS] Stats: {total_insights} insights, {total_cards} cards from "
                f"{transcripts_with_insights} transcripts"
            )
            
            return {
                "success": True,
                "total_insights": total_insights,
                "transcripts_with_insights": transcripts_with_insights,
                "total_transcripts": total_transcripts,
                "total_cards": total_cards,
                "avg_quality": round(avg_quality or 0.0, 2),
                "connections": total_cards
            }
            
    except Exception as e:
        logger.error(f"[WHISPERS] Error fetching stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights")
async def get_all_insights(limit: int = 100):
    """
    Get all insights for debugging/inspection.
    
    Args:
        limit: Maximum number of insights to return
    
    Returns:
        dict: List of insights with their embeddings and metadata
    """
    from munshi_machine.core import config
    from munshi_machine.db.database import engine
    from munshi_machine.models.private.insight_vector import InsightVector
    from sqlmodel import Session, select
    
    logger = config.get_logger(__name__)
    
    try:
        with Session(engine, expire_on_commit=False) as session:
            query = (
                select(InsightVector)
                .order_by(InsightVector.created_at.desc())
                .limit(limit)
            )
            
            insights = session.exec(query).all()
            
            results = [
                {
                    "uid": str(insight.uid),
                    "transcript_id": str(insight.transcript_id),
                    "text": insight.text,
                    "type": insight.type,
                    "index": insight.index,
                    "has_embedding": len(insight.embedding) == 384,
                    "created_at": insight.created_at
                }
                for insight in insights
            ]
            
            logger.info(f"[WHISPERS] Fetched {len(results)} insights")
            
            return {
                "success": True,
                "insights": results,
                "count": len(results)
            }
            
    except Exception as e:
        logger.error(f"[WHISPERS] Error fetching insights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tangents")
async def get_tangents_endpoint():
    """
    Fetch all tangents for the Tangent Cloud.
    
    Returns:
        dict: Tangents with their source transcripts
    """
    from munshi_machine.core import config
    from munshi_machine.db.database import engine
    from munshi_machine.models.private import Transcript
    from sqlmodel import Session, select
    import json
    
    logger = config.get_logger(__name__)
    
    try:
        logger.info(f"[WHISPERS] Fetching tangents")
        
        tangents = []
        
        with Session(engine, expire_on_commit=False) as session:
            query = (
                select(Transcript)
                .where(Transcript.summary.isnot(None))
            )
            transcripts = session.exec(query).all()
            
            for t in transcripts:
                try:
                    summary_data = json.loads(t.summary)
                    transcript_tangents = summary_data.get("tangents", [])
                    
                    for tangent in transcript_tangents:
                        tangents.append({
                            "text": tangent,
                            "transcript_id": str(t.uid),
                            "transcript_title": t.title or "Untitled"
                        })
                except (json.JSONDecodeError, KeyError):
                    continue
        
        logger.info(f"[WHISPERS] Fetched {len(tangents)} tangents")
        
        return {
            "success": True,
            "tangents": tangents,
            "count": len(tangents)
        }
        
    except Exception as e:
        logger.error(f"[WHISPERS] Error fetching tangents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cards/{card_uid}")
async def get_card_detail_endpoint(card_uid: str):
    """
    Fetch full details for a specific card including related insights.
    
    Args:
        card_uid: Card UUID
    
    Returns:
        dict: Card with full context and related insights
    """
    from munshi_machine.core import config
    from munshi_machine.db.database import engine
    from munshi_machine.models.private.insight_card import InsightCard
    from munshi_machine.models.private.insight_vector import InsightVector
    from munshi_machine.models.private import Transcript
    from sqlmodel import Session, select
    from uuid import UUID
    
    logger = config.get_logger(__name__)
    
    try:
        card_id = UUID(card_uid)
        logger.info(f"[WHISPERS] Fetching card detail: {card_uid}")
        
        with Session(engine, expire_on_commit=False) as session:
            card = session.get(InsightCard, card_id)
            if not card:
                raise HTTPException(status_code=404, detail="Card not found")
            
            insights_query = (
                select(InsightVector)
                .where(InsightVector.uid.in_(card.involved_insight_uids))
            )
            insights = session.exec(insights_query).all()
            
            transcript_ids = list(set([iv.transcript_id for iv in insights]))
            transcripts_query = (
                select(Transcript)
                .where(Transcript.uid.in_(transcript_ids))
            )
            transcripts = session.exec(transcripts_query).all()
            transcript_map = {str(t.uid): t for t in transcripts}
            
            related_insights = [
                {
                    "uid": str(iv.uid),
                    "text": iv.text,
                    "transcript_id": str(iv.transcript_id),
                    "transcript_title": transcript_map.get(str(iv.transcript_id), {}).title or "Unknown"
                }
                for iv in insights
            ]
            
            return {
                "success": True,
                "card": {
                    "uid": str(card.uid),
                    "type": card.type,
                    "title": card.title,
                    "content": card.content,
                    "source_count": card.source_count,
                    "quality_score": card.quality_score,
                    "created_at": card.created_at,
                    "related_insights": related_insights
                }
            }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid card UID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[WHISPERS] Error fetching card detail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

