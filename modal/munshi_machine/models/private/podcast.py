from munshi_machine.models.private.batch import Batch
from .base import PrivateSchemaBase
from sqlmodel import Field, Relationship
from typing import List, TYPE_CHECKING
from munshi_machine.models.status import TranscriptStatus
from pydantic import ConfigDict
import uuid

if TYPE_CHECKING:
    from .transcript import Transcript


class Podcast(PrivateSchemaBase, table=True):
    __tablename__: str = "podcasts"
    # __table_args__ = {"schema": "private"}
    
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
    # podcast index metadata
    pi_guid: uuid.UUID | None = Field(default=None, unique=True, index=True)
    title: str | None = None
    url: str | None = None
    description: str | None = None
    image: str | None = None
    artwork: str | None = None
    author: str | None = None
    ownerName: str | None = None
    language: str | None = None
    download_link: str | None = None
    date_published: int | None = None
    # meta: dict | None = Field(
    #     default=None, sa_type=JSON, sa_column_kwargs={"name": "metadata"}
    # )

    # back reference
    transcripts: List["Transcript"] = Relationship(
        back_populates="podcast",
        sa_relationship_kwargs={
            "cascade": "all, delete",
            "order_by": "desc(Transcript.date_published)",
        },
    )
    status: TranscriptStatus = Field(
        default=TranscriptStatus.PENDING, sa_column_kwargs={"nullable": False}
    )

    # relationship
    batch: Batch = Relationship(back_populates="podcasts")
    # refers to parent podcast
    batch_id: uuid.UUID | None = Field( 
        foreign_key="private.batches.uid", nullable=True
    )