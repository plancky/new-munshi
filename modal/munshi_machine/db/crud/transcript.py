from typing import List
from uuid import UUID

from sqlalchemy.orm import selectinload, sessionmaker
from sqlmodel import Session, select, func

from munshi_machine.db.database import engine
from munshi_machine.models.private.batch import Batch
from munshi_machine.models.private.podcast import Podcast
from munshi_machine.models.private.transcript import Transcript


def find_transcripts_by_podcast_uid(
    uid: UUID, offset: int = 0, limit: int = 50, session: Session | None = None
) -> List[Transcript] | None:
    def execute(session: Session):
        statement = (
            select(Transcript)
            .where(Transcript.podcast_id == uid)
            .options(selectinload(Transcript.podcast))
            .order_by(Transcript.date_published.desc())
            .offset(offset)
            .limit(limit)
        )
        result = session.exec(statement)
        transcripts = result.all()
        
        # Explicitly access the podcast relationship to ensure it's loaded
        # This forces SQLAlchemy to populate the relationship before serialization
        for transcript in transcripts:
            _ = transcript.podcast  # Access to trigger loading
        
        return transcripts

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def count_transcripts_by_podcast_uid(
    uid: UUID, session: Session | None = None
) -> int:
    def execute(session: Session):
        statement = select(func.count()).where(Transcript.podcast_id == uid)
        return session.exec(statement).one()

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def get_all_podcasts(batch_id: UUID | None = None, session: Session | None = None):
    def execute(session: Session):
        # Select only the columns needed for listing podcasts to keep the response light
        statement = select(
            Podcast.uid,
            Podcast.pi_guid,
            Podcast.title,
            Podcast.url,
            Podcast.description,
            Podcast.image,
            Podcast.artwork,
            Podcast.author,
            Podcast.ownerName,
            Podcast.language,
            Podcast.download_link,
            Podcast.date_published,
            Podcast.status,
            Podcast.batch_id,
        )
        if batch_id:
            statement = statement.where(Podcast.batch_id == batch_id)
        
        result = session.exec(statement)
        return result.all()

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def get_all_batches(session: Session | None = None):
    def execute(session: Session):
        statement = select(Batch).options(
            selectinload(Batch.podcasts).selectinload(Podcast.transcripts)
        )
        result = session.exec(statement)
        batches = result.all()
        
        # Explicitly access the relationships to ensure they're loaded
        # This forces SQLAlchemy to populate the relationships before serialization
        for batch in batches:
            _ = batch.podcasts  # Access to trigger loading
            for podcast in batch.podcasts:
                _ = podcast.transcripts  # Access to trigger loading
        
        return batches

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def get_batch_by_uid(
    batch_uid: UUID, session: Session | None = None
) -> Batch | None:
    def execute(session: Session):
        statement = (
            select(Batch)
            .where(Batch.uid == batch_uid)
            .options(selectinload(Batch.podcasts).selectinload(Podcast.transcripts))
        )
        result = session.exec(statement)
        batch = result.first()
        
        # Explicitly access the relationships to ensure they're loaded
        # This forces SQLAlchemy to populate the relationships before serialization
        if batch:
            _ = batch.podcasts  # Access to trigger loading
            for podcast in batch.podcasts:
                _ = podcast.transcripts  # Access to trigger loading
        
        return batch

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def get_podcast_by_uid(
    podcast_uid: UUID, session: Session | None = None
) -> Podcast | None:
    def execute(session: Session):
        statement = (
            select(Podcast)
            .where(Podcast.uid == podcast_uid)
            .options(selectinload(Podcast.transcripts))
        )
        result = session.exec(statement)
        podcast = result.first()
        
        # Explicitly access the transcripts relationship to ensure it's loaded
        # This forces SQLAlchemy to populate the relationship before serialization
        if podcast:
            _ = podcast.transcripts  # Access to trigger loading
        
        return podcast

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def get_podcasts_by_batch_id(
    batch_id: UUID, session: Session | None = None
) -> List[Podcast]:
    def execute(session: Session):
        statement = (
            select(Podcast)
            .where(Podcast.batch_id == batch_id)
            .options(selectinload(Podcast.transcripts))
        )
        result = session.exec(statement)
        podcasts = result.all()
        
        # Explicitly access the transcripts relationship to ensure it's loaded
        # This forces SQLAlchemy to populate the relationship before serialization
        for podcast in podcasts:
            _ = podcast.transcripts  # Access to trigger loading
        
        return podcasts

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def find_transcript_by_uid(
    uid: UUID, session: Session | None = None
) -> Transcript | None:
    def execute(session: Session):
        statement = (
            select(Transcript)
            .where(Transcript.uid == uid)
            .options(selectinload(Transcript.podcast))
        )
        result = session.exec(statement)
        return result.first()

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def get_transcript_status_minimal(
    uid: UUID, session: Session | None = None
):
    def execute(session: Session):
        # Only select the columns we need for status polling
        statement = (
            select(
                Transcript.uid, 
                Transcript.status, 
                Transcript.title, 
                Transcript.podcast_id
            )
            .where(Transcript.uid == uid)
        )
        result = session.exec(statement)
        return result.first()

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def find_transcript_by_episode_guid(
    guid: str, session: Session | sessionmaker[Session] | None = None
) -> Transcript | None:
    def execute(session: Session):
        statement = (
            select(Transcript)
            .where(Transcript.episode_guid == guid)
            .options(selectinload(Transcript.podcast))
        )
        result = session.exec(statement)
        return result.first()

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def find_transcript_by_file_hash(
    file_hash: str, session: Session | None = None
) -> Transcript | None:
    def execute(session: Session):
        statement = (
            select(Transcript)
            .where(Transcript.file_hash == file_hash)
            .options(selectinload(Transcript.podcast))
        )
        result = session.exec(statement)
        return result.first()

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def create_transcript(
    transcript: Transcript, session: Session | sessionmaker[Session] | None
) -> Transcript:
    def execute(session: Session):
        session.add(transcript)
        session.commit()
        session.refresh(transcript)
        return transcript

    if session is not None:
        return execute(session)

    with Session(engine, expire_on_commit=False) as session:
        return execute(session)

def update_or_create_transcripts(
    transcripts: List[Transcript] | Transcript,
    *,
    include_fields: List | None = None,
    exclude_fields: List | None = ["uid"],
    upsert=True
) -> list[Transcript] | Transcript:
    if isinstance(transcripts, Transcript):
        transcripts: List[Transcript] = [transcripts]
    if isinstance(transcripts, List) and len(transcripts) == 0:
        return []

    with Session(engine, expire_on_commit=False) as session:
        results, modified = Transcript.fast_batch_find_replace_sync(
            session,
            operations=list(
                map(
                    lambda _t: (
                        {
                            "uid": _t.uid,
                        },
                        _t.model_dump(
                            include=set(include_fields) if include_fields else None,
                            exclude=set(exclude_fields) if exclude_fields else None,
                        ),
                    ),
                    transcripts,
                )
            ),
            upsert=upsert,
        )

    return results if len(results) > 1 else results[0], modified

def update_or_create_episodes(
    transcripts: List[Transcript] | Transcript,
    *,
    include_fields: List | None = None,
    exclude_fields: List | None = ["uid"],
    upsert=True
) -> list[Transcript] | Transcript:
    if isinstance(transcripts, Transcript):
        transcripts: List[Transcript] = [transcripts]
    if isinstance(transcripts, List) and len(transcripts) == 0:
        return []

    with Session(engine, expire_on_commit=False) as session:
        results, modified = Transcript.fast_batch_find_replace_sync(
            session,
            operations=list(
                map(
                    lambda _t: (
                        {
                            "episode_guid": _t.episode_guid,
                        },
                        _t.model_dump(
                            include=set(include_fields) if include_fields else None,
                            exclude=set(exclude_fields) if exclude_fields else None,
                        ),
                    ),
                    transcripts,
                )
            ),
            upsert=upsert,
        )

    return results if len(results) > 1 else results[0], modified


def update_or_create_podcasts(
    podcasts: list[Podcast] | Podcast,
    *,
    include_fields: List | None = None,
    exclude_fields: List | None = ["uid"],
    upsert=True
) -> list[Podcast] | Podcast:
    if isinstance(podcasts, Podcast):
        podcasts = [podcasts]

    if isinstance(podcasts, List) and len(podcasts) == 0:
        return []

    with Session(engine, expire_on_commit=False) as session:
        results, modified = Podcast.fast_batch_find_replace_sync(
            session,
            operations=list(
                map(
                    lambda _t: (
                        {
                            "pi_guid": _t.pi_guid,
                        },
                        _t.model_dump(
                            include=set(include_fields) if include_fields else None,
                            exclude=set(exclude_fields) if exclude_fields else None,
                        ),
                    ),
                    podcasts,
                )
            ),
            upsert=upsert,
        )

    return results if len(results) > 1 else results[0], modified
