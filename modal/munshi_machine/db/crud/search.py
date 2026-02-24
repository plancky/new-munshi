from uuid import UUID
from sqlmodel import Session, select
from munshi_machine.db.database import engine
from munshi_machine.models.private import Search
from typing import List, Optional, Dict

def create_search(
    query: str,
    results: List[dict],
    timing: Optional[Dict[str, float]] = None,
    query_embedding: Optional[List[float]] = None,
    answer: Optional[str] = None,
    podcast_id: Optional[UUID] = None,
    collection_id: Optional[UUID] = None,
    session: Optional[Session] = None
) -> Search:
    """
    Create a new shared search record.
    """
    def execute(session: Session):
        db_search = Search(
            query=query,
            results=results,
            timing=timing,
            query_embedding=query_embedding,
            answer=answer,
            podcast_id=podcast_id,
            collection_id=collection_id
        )
        session.add(db_search)
        session.commit()
        session.refresh(db_search)
        return db_search
    
    if session is not None:
        return execute(session)
    
    with Session(engine, expire_on_commit=False) as session:
        return execute(session)

def get_search_by_uid(
    uid: UUID,
    session: Optional[Session] = None
) -> Optional[Search]:
    """
    Retrieve a shared search record by its UUID.
    """
    def execute(session: Session):
        statement = select(Search).where(Search.uid == uid)
        result = session.exec(statement)
        return result.first()
    
    if session is not None:
        return execute(session)
    
    with Session(engine, expire_on_commit=False) as session:
        return execute(session)
