import uuid
import time
from sqlmodel import Field, Relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Text, Index, Enum
from .base import PrivateSchemaBase
from typing import TYPE_CHECKING
import enum

if TYPE_CHECKING:
    from .transcript import Transcript


class InsightType(str, enum.Enum):
    """Type of insight extracted from transcript"""
    INSIGHT = "insight"
    TANGENT = "tangent"


class InsightVector(PrivateSchemaBase, table=True):
    """
    Store individual insight embeddings for cross-transcript pattern detection.
    
    Each insight from a transcript gets its own embedding vector.
    This enables finding Echoes, Bridges, and Fractures across all transcripts.
    
    Note: Tangents are not embedded, only insights.
    """
    __tablename__: str = "insight_vectors"
    __table_args__ = (
        Index("ix_insight_vectors_transcript_id", "transcript_id"),
        Index("ix_insight_vectors_type", "type"),
        {"schema": "private"}
    )

    uid: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        unique=True,
        nullable=False,
    )

    transcript_id: uuid.UUID = Field(
        foreign_key="private.transcripts.uid",
        nullable=False,
        index=True,
    )

    # The actual insight text
    text: str = Field(sa_column=Column(Text, nullable=False))
    
    # Type: insight or tangent
    type: InsightType = Field(
        sa_column=Column(
            Enum(InsightType, native_enum=False, length=20),
            nullable=False
        )
    )
    
    # Position in original list (for ordering)
    index: int = Field(nullable=False)

    # 384-dimensional embedding vector
    embedding: list[float] = Field(
        sa_column=Column(Vector(384), nullable=False)
    )

    created_at: float = Field(default_factory=time.time, nullable=False)

    # Relationship
    transcript: "Transcript" = Relationship(back_populates="insight_vectors")
