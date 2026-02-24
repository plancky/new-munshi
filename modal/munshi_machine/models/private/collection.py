from .base import PrivateSchemaBase
from sqlmodel import Field, Relationship
from typing import List, TYPE_CHECKING
from pydantic import ConfigDict
import uuid

if TYPE_CHECKING:
    from .transcript import Transcript


class CollectionTranscriptLink(PrivateSchemaBase, table=True):

    __tablename__: str = "collection_transcript_links"
    
    collection_uid: uuid.UUID = Field(
        foreign_key="private.collections.uid", primary_key=True, ondelete="CASCADE"
    )
    transcript_uid: uuid.UUID = Field(
        foreign_key="private.transcripts.uid", primary_key=True, ondelete="CASCADE"
    )


class Collection(PrivateSchemaBase, table=True):
    __tablename__: str = "collections"
    
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
    title: str | None = None
    description: str | None = None
    artwork: str | None = None
    author: str | None = None
    ownerName: str | None = None

    language: str | None = None
    download_link: str | None = None

    # Relationship to transcripts via the link model
    transcripts: List["Transcript"] = Relationship(
        back_populates="collections",
        link_model=CollectionTranscriptLink
    )
