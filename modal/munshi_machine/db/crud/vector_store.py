from uuid import UUID
from sqlmodel import Session, select, func
from munshi_machine.db.database import engine
from munshi_machine.models.private import VectorStore
from typing import List

def find_vectors_by_transcript_id(
    transcript_id: UUID,
    session: Session | None = None
) -> List[VectorStore]:
    """
    Retrieve all vector chunks for a given transcript.
    
    Args:
        transcript_id: UUID of the transcript
        session: Optional existing session
    
    Returns
        List of VectorStore records, ordered by chunk_index
    """
    def execute(session: Session):
        statement = (
            select(VectorStore)
            .where(VectorStore.transcript_id == transcript_id)
            .order_by(VectorStore.chunk_index)
        )
        result = session.exec(statement)
        return result.all()
    
    if session is not None:
        return execute(session)
    
    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def delete_vectors_by_transcript_id(
    transcript_id: UUID,
    session: Session | None = None
) -> int:
    """
    Delete all vector chunks for a given transcript.
    Useful for re-processing or cleanup.
    
    Args:
        transcript_id: UUID of the transcript
        session: Optional existing session
    
    Returns:
        Number of records deleted
    """
    def execute(session: Session):
        statement = select(VectorStore).where(
            VectorStore.transcript_id == transcript_id
        )
        vectors = session.exec(statement).all()
        count = len(vectors)
        
        for vector in vectors:
            session.delete(vector)
        
        session.commit()
        return count
    
    if session is not None:
        return execute(session)
    
    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def similarity_search(
    query_embedding: List[float],
    limit: int = 10,
    transcript_id: UUID | None = None,
    transcript_ids: List[UUID] | None = None,
    session: Session | None = None
) -> List[tuple[VectorStore, float]]:
    """
    Perform similarity search using cosine distance.
    
    Args:
        query_embedding: The embedding vector to search for (384 dims)
        limit: Maximum number of results to return
        transcript_id: Optional filter by specific transcript
        transcript_ids: Optional filter by list of transcript IDs (for podcast search)
        session: Optional existing session
    
    Returns:
        List of (VectorStore, distance) tuples, ordered by similarity
    """
    def execute(session: Session):
        # Using pgvector's cosine distance operator <=>
        # Lower distance = more similar
        statement = (
            select(
                VectorStore,
                VectorStore.embedding.cosine_distance(query_embedding).label('distance')
            )
        )
        
        # Apply filters (mutually exclusive)
        if transcript_id:
            statement = statement.where(VectorStore.transcript_id == transcript_id)
        elif transcript_ids:
            statement = statement.where(VectorStore.transcript_id.in_(transcript_ids))
        
        statement = statement.order_by('distance').limit(limit)
        
        result = session.exec(statement)
        return result.all()
    
    if session is not None:
        return execute(session)
    
    with Session(engine, expire_on_commit=False) as session:
        return execute(session)


def count_vectors_by_transcript_id(
    transcript_id: UUID,
    session: Session | None = None
) -> int:
    """Count number of vector chunks for a transcript."""
    def execute(session: Session):
        statement = (
            select(func.count(VectorStore.uid))
            .where(VectorStore.transcript_id == transcript_id)
        )
        result = session.exec(statement).one()
        return result
    
    if session is not None:
        return execute(session)
    
    with Session(engine, expire_on_commit=False) as session:
        return execute(session)
