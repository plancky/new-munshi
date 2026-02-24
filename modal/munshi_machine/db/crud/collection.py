from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from munshi_machine.db.database import engine
from munshi_machine.models.private.collection import Collection, CollectionTranscriptLink
from munshi_machine.models.private.transcript import Transcript


def get_collection_by_uid(
    uid: UUID, session: Session | None = None
) -> Optional[Collection]:
    def execute(session: Session):
        statement = (
            select(Collection)
            .where(Collection.uid == uid)
            .options(selectinload(Collection.transcripts))
        )
        result = session.exec(statement)
        collection = result.first()

        if collection:
            # Explicitly access transcripts to ensure they are loaded
            _ = collection.transcripts

        return collection

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def get_all_collections(session: Session | None = None) -> List[Collection]:
    def execute(session: Session):
        statement = select(Collection).options(selectinload(Collection.transcripts))
        result = session.exec(statement)
        collections = result.all()

        for collection in collections:
            _ = collection.transcripts

        return collections

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def create_collection(
    title: str,
    description: str | None = None,
    artwork: str | None = None,
    author: str | None = None,
    owner_name: str | None = None,
    transcript_ids: List[UUID] | None = None,
    session: Session | None = None,
) -> Collection:
    def execute(session: Session):
        collection = Collection(
            title=title,
            description=description,
            artwork=artwork,
            author=author,
            ownerName=owner_name,
        )
        session.add(collection)
        session.flush()  # Get the UID for the collection

        if transcript_ids:
            # Fetch transcripts to verify they exist and link them
            statement = select(Transcript).where(Transcript.uid.in_(transcript_ids))
            transcripts = session.exec(statement).all()
            collection.transcripts = transcripts

        session.commit()
        session.refresh(collection)
        return collection

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def add_transcripts_to_collection(
    collection_uid: UUID,
    transcript_ids: List[UUID],
    session: Session | None = None,
) -> Optional[Collection]:
    def execute(session: Session):
        collection = session.get(Collection, collection_uid)
        if not collection:
            return None

        # Fetch transcripts that aren't already in the collection
        existing_ids = {t.uid for t in collection.transcripts}
        new_ids = [tid for tid in transcript_ids if tid not in existing_ids]

        if new_ids:
            statement = select(Transcript).where(Transcript.uid.in_(new_ids))
            new_transcripts = session.exec(statement).all()
            collection.transcripts.extend(new_transcripts)
            session.add(collection)
            session.commit()
            session.refresh(collection)

        return collection

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def remove_transcripts_from_collection(
    collection_uid: UUID,
    transcript_ids: List[UUID],
    session: Session | None = None,
) -> Optional[Collection]:
    def execute(session: Session):
        collection = session.get(Collection, collection_uid)
        if not collection:
            return None

        # Filter out the transcripts to remove
        collection.transcripts = [
            t for t in collection.transcripts if t.uid not in transcript_ids
        ]
        
        session.add(collection)
        session.commit()
        session.refresh(collection)
        return collection

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def delete_collection(uid: UUID, session: Session | None = None) -> bool:
    def execute(session: Session):
        collection = session.get(Collection, uid)
        if not collection:
            return False
        
        session.delete(collection)
        session.commit()
        return True

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)
