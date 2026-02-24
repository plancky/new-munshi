"""
Card generation engine for Whispers.

This module contains the logic to:
1. Process insights one by one
2. Find similar insights across transcripts
3. Aggregate context from source transcripts
4. Generate insight cards using LLM
"""

import json
import time
from uuid import UUID

import modal

from munshi_machine.core.app import app, custom_secret
from munshi_machine.core.images import base_image


@app.function(
    image=base_image,
    timeout=3600,  # 1 hour max
    secrets=[custom_secret],
    retries=2,
)
async def generate_all_cards(user_id: str | None = None) -> dict:
    """
    Generate all insight cards by processing insights sequentially.
    
    Algorithm:
    1. Fetch all insights that haven't been processed
    2. For each insight:
       - Search for similar insights (excluding already processed)
       - Determine card type (Echo, Bridge, Fracture)
       - Fetch transcript context for all involved insights
       - Generate card using LLM
       - Mark all involved insights as processed
    3. Return generated cards
    
    Args:
        user_id: Optional filter by user (future feature)
    
    Returns:
        dict: Generated cards and stats
    """
    from munshi_machine.core import config
    from munshi_machine.db.database import engine
    from munshi_machine.models.private.insight_vector import InsightVector
    from sqlmodel import Session, select
    
    logger = config.get_logger(__name__)
    start_time = time.perf_counter()
    
    try:
        # Step 1: Load all insights
        logger.info("[CARD_GEN] Loading all insight vectors from database")
        
        with Session(engine, expire_on_commit=False) as session:
            query = select(InsightVector).order_by(InsightVector.created_at.desc())
            all_insights = session.exec(query).all()
            
            total_count = len(all_insights)
            logger.info(f"[CARD_GEN] Loaded {total_count} insights")
            
            if total_count == 0:
                return {
                    "success": True,
                    "cards_generated": 0,
                    "insights_processed": 0,
                    "message": "No insights found"
                }
            
            # Convert to list of dicts for easier processing
            insights_data = [
                {
                    "uid": str(insight.uid),
                    "transcript_id": str(insight.transcript_id),
                    "text": insight.text,
                    "embedding": insight.embedding,
                    "index": insight.index,
                }
                for insight in all_insights
            ]
        
        # Step 2: Process insights sequentially
        logger.info("[CARD_GEN] Starting sequential processing")
        
        processed_uids = set()  # Track which insights have been used in cards
        cards = []
        
        for idx, current_insight in enumerate(insights_data):
            # Skip if already processed
            if current_insight["uid"] in processed_uids:
                continue
            
            logger.info(
                f"[CARD_GEN] Processing insight {idx + 1}/{total_count}: "
                f"{current_insight['text'][:80]}..."
            )
            
            # Step 3: Find similar insights (excluding processed ones)
            similar_insights = await find_similar_insights(
                current_insight,
                insights_data,
                processed_uids,
                threshold=0.60  # Minimum similarity score (lowered from 0.75)
            )
            
            logger.info(
                f"[CARD_GEN] Found {len(similar_insights)} similar insights "
                f"(threshold: 0.60)"
            )
            
            # Step 4: Generate card if we have enough matches
            if len(similar_insights) >= 2:  # Need at least 2 matches
                card = await generate_card(
                    current_insight,
                    similar_insights
                )
                
                if card:
                    # Save card to database
                    saved_card = await save_card_to_db(card)
                    
                    if saved_card:
                        cards.append({
                            "uid": str(saved_card.uid),
                            "type": saved_card.type,
                            "title": saved_card.title,
                            "source_count": saved_card.source_count
                        })
                        
                        # Mark all involved insights as processed
                        processed_uids.add(current_insight["uid"])
                        for sim in similar_insights:
                            processed_uids.add(sim["uid"])
                        
                        logger.info(
                            f"[CARD_GEN] Generated {card['type']} card: {card['title'][:60]}..."
                        )
        
        elapsed = time.perf_counter() - start_time
        
        logger.info(
            f"[CARD_GEN] Completed: {len(cards)} cards generated, "
            f"{len(processed_uids)}/{total_count} insights processed in {elapsed:.2f}s"
        )
        
        return {
            "success": True,
            "cards": cards,
            "cards_generated": len(cards),
            "insights_processed": len(processed_uids),
            "total_insights": total_count,
            "duration_sec": elapsed
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"[CARD_GEN] Failed after {elapsed:.2f}s: {e}", exc_info=True)
        
        return {
            "success": False,
            "error": str(e),
            "duration_sec": elapsed
        }


async def save_card_to_db(card_data: dict) -> any:
    """
    Save generated card to database.
    
    Args:
        card_data: Card dict from LLM with type, title, content, metadata
    
    Returns:
        InsightCard: Saved card record, or None if save fails
    """
    from munshi_machine.core import config
    from munshi_machine.db.database import engine
    from munshi_machine.models.private.insight_card import InsightCard, CardType
    from sqlmodel import Session
    from uuid import UUID
    
    logger = config.get_logger(__name__)
    
    try:
        logger.info(f"[CARD_GEN] Attempting to save card: {card_data.get('title', 'Unknown')[:50]}")
        logger.debug(f"[CARD_GEN] Card data keys: {list(card_data.keys())}")
        
        # Extract card type (normalize to lowercase)
        card_type_str = card_data.get("type", "echo").lower()
        logger.debug(f"[CARD_GEN] Card type: {card_type_str}")
        card_type = CardType(card_type_str)
        
        # Build content dict (all card-specific fields)
        content = {}
        
        if card_type == CardType.ECHO:
            content = {
                "core_fact": card_data.get("core_fact"),
                "significance": card_data.get("significance"),
                "convergence_score": card_data.get("convergence_score"),
                "instances": []  # Populated from similar_insights
            }
        elif card_type == CardType.BRIDGE:
            content = {
                "connecting_insight": card_data.get("connecting_insight"),
                "cluster_a": card_data.get("cluster_a"),
                "cluster_b": card_data.get("cluster_b"),
                "pattern_explanation": card_data.get("pattern_explanation"),
                "insight_counts": card_data.get("insight_counts", {})
            }
        elif card_type == CardType.FRACTURE:
            content = {
                "issue": card_data.get("issue"),
                "severity": card_data.get("severity", "medium"),
                "top_patterns": card_data.get("top_patterns", []),
                "recommendation": card_data.get("recommendation")
            }
        
        # Calculate quality score
        quality_score = calculate_quality_score(card_data)
        logger.debug(f"[CARD_GEN] Quality score: {quality_score}")
        
        # Create card record
        logger.debug(f"[CARD_GEN] Creating InsightCard object...")
        # Convert UUIDs to strings for JSON storage
        involved_uids_str = [str(uid) if not isinstance(uid, str) else uid for uid in card_data["involved_insight_uids"]]
        
        card = InsightCard(
            type=card_type,
            title=card_data.get("title", "Untitled Card"),
            content=content,
            seed_insight_uid=UUID(card_data["seed_insight_uid"]),
            involved_insight_uids=involved_uids_str,
            source_count=card_data["source_count"],
            quality_score=quality_score
        )
        
        logger.debug(f"[CARD_GEN] Saving to database...")
        with Session(engine, expire_on_commit=False) as session:
            session.add(card)
            session.commit()
            session.refresh(card)
        
        logger.info(f"[CARD_GEN] ✅ Saved {card_type} card to DB: {card.uid}")
        
        return card
        
    except Exception as e:
        logger.error(f"[CARD_GEN] Failed to save card to DB: {e}", exc_info=True)
        return None


def calculate_quality_score(card_data: dict) -> float:
    """
    Calculate quality score for ranking cards.
    
    Factors:
    - Source count (more sources = higher quality)
    - Similarity scores (higher convergence = higher quality)
    - Card type (Echoes > Bridges > Fractures in terms of reliability)
    
    Returns:
        float: Score between 0-1
    """
    source_count = card_data.get("source_count", 1)
    card_type = card_data.get("type", "echo")
    
    # Base score from source count (diminishing returns)
    import math
    source_score = min(1.0, math.log(source_count + 1) / math.log(10))
    
    # Type multiplier
    type_weights = {
        "echo": 1.0,      # Highest quality (independent confirmation)
        "bridge": 0.9,    # High quality (connecting patterns)
        "fracture": 0.85  # Good quality (identifying issues)
    }
    type_multiplier = type_weights.get(card_type, 0.8)
    
    # Convergence bonus for echoes
    convergence_bonus = 0
    if card_type == "echo" and "convergence_score" in card_data:
        convergence_bonus = card_data["convergence_score"] * 0.2
    
    quality = (source_score * type_multiplier) + convergence_bonus
    
    return min(1.0, quality)


async def find_similar_insights(
    current_insight: dict,
    all_insights: list[dict],
    processed_uids: set,
    threshold: float = 0.60
) -> list[dict]:
    """
    Find insights similar to the current one using cosine similarity.
    
    Args:
        current_insight: The insight to compare against
        all_insights: All available insights
        processed_uids: Set of insight UIDs already used in cards
        threshold: Minimum similarity score (0-1)
    
    Returns:
        list: Similar insights with similarity scores, sorted by score (highest first)
    """
    import numpy as np
    from numpy.linalg import norm
    from munshi_machine.core import config
    
    logger = config.get_logger(__name__)
    
    current_embedding = np.array(current_insight["embedding"])
    similar = []
    all_scores = []  # Track all similarity scores for debugging
    
    for insight in all_insights:
        # Skip self, already processed, or same transcript
        if (
            insight["uid"] == current_insight["uid"]
            or insight["uid"] in processed_uids
            or insight["transcript_id"] == current_insight["transcript_id"]
        ):
            continue
        
        # Compute cosine similarity
        other_embedding = np.array(insight["embedding"])
        similarity = np.dot(current_embedding, other_embedding) / (
            norm(current_embedding) * norm(other_embedding)
        )
        
        all_scores.append(float(similarity))
        
        if similarity >= threshold:
            similar.append({
                **insight,
                "similarity": float(similarity)
            })
    
    # Log top similarities for debugging (even if below threshold)
    if all_scores:
        top_3 = sorted(all_scores, reverse=True)[:3]
        logger.debug(f"[SIMILARITY] Top 3 scores: {[f'{s:.3f}' for s in top_3]}, threshold: {threshold}")
    
    # Sort by similarity (highest first)
    similar.sort(key=lambda x: x["similarity"], reverse=True)
    
    return similar


async def generate_card(
    current_insight: dict,
    similar_insights: list[dict]
) -> dict | None:
    """
    Generate an insight card using LLM.
    
    Steps:
    1. Fetch transcript context for all involved insights
    2. LLM analyzes patterns and determines card type
    3. LLM synthesizes card content
    
    Args:
        current_insight: The seed insight
        similar_insights: List of similar insights with scores
    
    Returns:
        dict: Generated card with LLM-determined type, or None if generation fails
    """
    from munshi_machine.core import config
    from munshi_machine.db.crud.transcript import find_transcript_by_uid
    from munshi_machine.db.database import engine
    from munshi_machine.lib.gemini.processor import generate_card_content
    from sqlmodel import Session
    
    logger = config.get_logger(__name__)
    
    try:
        # Step 1: Fetch transcript context
        all_involved = [current_insight] + similar_insights
        transcript_ids = list(set([ins["transcript_id"] for ins in all_involved]))
        
        logger.info(f"[CARD_GEN] Fetching context from {len(transcript_ids)} transcripts")
        
        with Session(engine, expire_on_commit=False) as session:
            transcript_contexts = []
            
            for tid in transcript_ids:
                transcript = find_transcript_by_uid(UUID(tid), session)
                if transcript:
                    # Parse summary to get context
                    try:
                        summary_data = json.loads(transcript.summary)
                        summary_html = summary_data.get("summary", "")
                    except:
                        summary_html = transcript.summary or ""
                    
                    transcript_contexts.append({
                        "uid": str(transcript.uid),
                        "title": transcript.title,
                        "summary": summary_html,
                        "podcast_title": transcript.podcast.title if transcript.podcast else None,
                    })
        
        # Step 2: Generate card using LLM (LLM determines card type)
        card = await generate_card_content(
            seed_insight=current_insight,
            similar_insights=similar_insights[:10],  # Top 10 matches
            transcript_contexts=transcript_contexts
        )
        
        if card:
            logger.info(f"[CARD_GEN] LLM determined card type: {card.get('type', 'unknown')}")
        
        if not card:
            return None
        
        return card
        
    except Exception as e:
        logger.error(f"[CARD_GEN] Failed to generate card: {e}", exc_info=True)
        return None


@app.local_entrypoint()
def generate_cards_cli():
    """
    CLI entrypoint to generate cards.
    
    Usage:
        modal run munshi_machine/functions/generate_cards.py
    """
    result = generate_all_cards.remote()
    print(json.dumps(result, indent=2))
