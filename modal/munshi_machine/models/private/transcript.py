import time
import uuid

from sqlalchemy import text, Index
from sqlmodel import Field, Relationship
from typing import List, TYPE_CHECKING
from pydantic import ConfigDict

from .collection import Collection, CollectionTranscriptLink

if TYPE_CHECKING:
    from .vector_store import VectorStore
    from .insight_vector import InsightVector


from munshi_machine.models.status import TranscriptStatus

from .base import PrivateSchemaBase
from .podcast import Podcast


class Transcript(PrivateSchemaBase, table=True):
    __tablename__: str = "transcripts"
    __table_args__ = (
        Index("ix_transcript_podcast_published", "podcast_id", "date_published"),
    )
    
    # Configure Pydantic to properly serialize SQLModel relationships
    model_config = ConfigDict(
        from_attributes=True,  # Allow ORM mode (read from attributes)
        arbitrary_types_allowed=True,  # Allow SQLModel types
    )

    uid: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        index=True,
        unique=True,
        nullable=False,
        primary_key=True,
    )

    # podcast index episode metadata
    episode_guid: str | None = Field(default=None, unique=True, index=True)
    title: str
    url: str | None = None
    description: str | None = None
    image: str | None = None
    artwork: str | None = None
    season: int | None = None
    episode: int | None = None
    language: str | None = None
    download_link: str | None = None
    duration: int | None = None
    date_published: int | None = None
    audio_url: str | None = None
    job_start_time: float | None = time.time()
    init_function_call_id: str | None = None
    # meta: dict | None = Field(
    #     default=None, sa_type=JSON, sa_column_kwargs={"name": "metadata"}
    # )
    # processed output texts
    status: TranscriptStatus = Field(
        default=TranscriptStatus.PENDING,
        sa_column_kwargs={
            "nullable": False,
            "server_default": text("'PENDING'"),  # <-- server-side enum default
        },
    )
    transcript: str | None = None
    cleaned_transcript: str | None = None
    summary: str | None = None
    file_hash: str | None = Field(default=None, index=True)
    
    # Track embedding generation status
    embeddings_generated: bool = Field(
        default=False,
        nullable=False,
        sa_column_kwargs={
            "server_default": text("false")  # Default to false for existing records
        }
    )

    # relationship
    podcast: Podcast = Relationship(back_populates="transcripts")
    # refers to parent podcast
    podcast_id: uuid.UUID | None = Field(
        foreign_key="private.podcasts.uid", nullable=True, ondelete="CASCADE"
    )

    vector_chunks: List["VectorStore"] = Relationship(
        back_populates="transcript",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    insight_vectors: List["InsightVector"] = Relationship(
        back_populates="transcript",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    collections: List[Collection] = Relationship(
        back_populates="transcripts",
        link_model=CollectionTranscriptLink
    )
