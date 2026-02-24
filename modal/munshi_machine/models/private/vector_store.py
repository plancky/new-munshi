import uuid
import time
from sqlmodel import Field, Relationship
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, Text, Index
from .base import PrivateSchemaBase
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .transcript import Transcript


class VectorStore(PrivateSchemaBase, table=True):
    """
    Store chunked text embeddings for semantic search.
    
    Each transcript is split into chunks, and each chunk gets its own
    embedding vector stored here. This enables fine-grained semantic search.
    """
    __tablename__: str = "vector_store"
    __table_args__ = (
        Index("ix_vector_store_transcript_id", "transcript_id"),
        # IVFFlat index for similarity search (created after data population)
        # Index('ix_vector_store_embedding', 'embedding', postgresql_using='ivfflat'),
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

    chunk_text: str = Field(sa_column=Column(Text, nullable=False))
    chunk_index: int = Field(sa_column=Column(Integer, nullable=False))

    embedding: list[float] = Field(
        sa_column=Column(Vector(384), nullable=False)  # pgvector type, 384 dimensions
    )

    created_at: float = Field(default_factory=time.time, nullable=False)

    # Relationship
    transcript: "Transcript" = Relationship(back_populates="vector_chunks")
