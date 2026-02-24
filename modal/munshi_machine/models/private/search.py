import uuid
import time
from datetime import datetime, timezone
from typing import Optional, List, Any, Dict
from sqlmodel import Field, Relationship, Column, JSON, DateTime
from pgvector.sqlalchemy import Vector
from .base import PrivateSchemaBase
from pydantic import ConfigDict

class Search(PrivateSchemaBase, table=True):
    """
    Store shared search results.
    
    A search can be performed over a single podcast or a batch (collection).
    We store the query, the generated answer, and the matching chunks
    so that the search can be shared via a permanent link.
    """
    __tablename__: str = "searches"
    __table_args__ = {"schema": "private"}

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    uid: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        unique=True,
        nullable=False,
    )

    query: str = Field(nullable=False)
    answer: Optional[str] = Field(default=None)
    
    # Store the search results (SearchResult objects) as JSON
    results: List[dict] = Field(default_factory=list, sa_column=Column(JSON))
    
    # Performance metrics associated with the search
    timing: Optional[Dict[str, float]] = Field(default=None, sa_column=Column(JSON))
    
    # Query embedding for future similarity searches or analysis
    query_embedding: Optional[List[float]] = Field(
        sa_column=Column(Vector(384), nullable=True)
    )
    
    # Metadata about the search scope
    podcast_id: Optional[uuid.UUID] = Field(
        default=None,
        foreign_key="private.podcasts.uid",
        nullable=True,
    )
    collection_id: Optional[uuid.UUID] = Field(
        default=None,
        foreign_key="private.collections.uid",
        nullable=True,
    )
    
    # When the search was created/shared
    created_on: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    
    # Timestamp for convenience (matches existing models)
    created_at: float = Field(default_factory=time.time, nullable=False)
