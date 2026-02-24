from .base import PrivateSchemaBase
from datetime import datetime, timezone
from sqlmodel import Field, Relationship, Column, DateTime
from typing import List, TYPE_CHECKING
from pydantic import ConfigDict
import uuid

if TYPE_CHECKING:
    from .podcast import Podcast


class Batch(PrivateSchemaBase, table=True):
    __tablename__: str = "batches"
    # __table_args__ = {"schema": "private"}
    # When the search was created/shared
    created_on: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=True)
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
    # podcast index metadata
    function_call_id: str | None = Field(
        default=None,
        index=True,
        unique=True,
        nullable=True,
    )
    # back reference
    podcasts: List["Podcast"] = Relationship(
        back_populates="batch"
    )
