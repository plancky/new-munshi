import uuid
import time
from sqlmodel import Field, Column, Text, JSON
from sqlalchemy import Index, Enum
from .base import PrivateSchemaBase
import enum


class CardType(str, enum.Enum):
    """Type of insight card"""
    ECHO = "echo"
    BRIDGE = "bridge"
    FRACTURE = "fracture"


class InsightCard(PrivateSchemaBase, table=True):
    """
    Store generated insight cards.
    
    Each card represents a pattern found across multiple insights:
    - Echo: Independent sources confirming same fact
    - Bridge: Insights connecting different domains
    - Fracture: Multiple signals of a problem/concern
    """
    __tablename__: str = "insight_cards"
    __table_args__ = (
        Index("ix_insight_cards_type", "type"),
        Index("ix_insight_cards_created_at", "created_at"),
        {"schema": "private"}
    )

    uid: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        unique=True,
        nullable=False,
    )

    # Card type determined by LLM
    type: CardType = Field(
        sa_column=Column(
            Enum(CardType, native_enum=False, length=20),
            nullable=False
        )
    )
    
    # Core content
    title: str = Field(max_length=500)
    
    # Type-specific content stored as JSON
    # For Echo: {core_fact, significance, convergence_score, instances: [...]}
    # For Bridge: {connecting_insight, cluster_a, cluster_b, pattern_explanation}
    # For Fracture: {issue, severity, top_patterns: [...], recommendation}
    content: dict = Field(sa_column=Column(JSON, nullable=False))
    
    # Metadata
    seed_insight_uid: uuid.UUID = Field(nullable=False)
    involved_insight_uids: list[uuid.UUID] = Field(sa_column=Column(JSON, nullable=False))
    source_count: int = Field(nullable=False)  # Number of insights in this card
    
    created_at: float = Field(default_factory=time.time, nullable=False)
    
    # Quality/ranking score (computed from similarity, source count, etc.)
    quality_score: float = Field(default=0.0, nullable=False)
