import time
from munshi_machine.core import config
from munshi_machine.lib.processing_states.base import ProcessingState
from munshi_machine.lib.processing_states.cleaning import CleaningProcessingState
from munshi_machine.lib.chunking_utils import chunk_transcript
from munshi_machine.models.status import TranscriptStatus
from munshi_machine.models.private.vector_store import VectorStore
from munshi_machine.db.database import engine
from sqlmodel import Session
logger = config.get_logger(__name__)


class EmbeddingProcessingState(ProcessingState):
    def __init__(self) -> None:
        self.StateSymbol = TranscriptStatus.EMBEDDING
    
    def _next_state(self):
        return CleaningProcessingState()
    
    async def run_job(self, uid: str) -> int:
        """
        Generate embeddings for transcript chunks and store in vector_store table.
        
        Process:
        1. Load transcript text from database
        2. Split into overlapping chunks
        3. Generate embeddings for each chunk using TextEmbeddingsInference
        4. Store chunks and embeddings in vector_store table
        5. Transition to next state (CLEANING)
        """
        from munshi_machine.functions.embeddings import TextEmbeddingsInference
        
        try:
            start_time = time.time()
            
            # Update status in DB
            transcript = self.update_status_in_db(uid)
            logger.info(f"[EMBEDDING] Starting embedding generation for {uid}")
            
            # Validate transcript exists and has content
            if not transcript.transcript or len(transcript.transcript.strip()) == 0:
                raise ValueError(f"Transcript {uid} has no content to embed")
            if transcript.embeddings_generated:
                logger.info(f"[EMBEDDING] Embeddings already generated for {uid}, skipping...")
                await self._next_state().run_job(uid)
                return 0
            # Step 1: Chunk the transcript
            logger.info(f"[EMBEDDING] Chunking transcript (length: {len(transcript.transcript)} chars)")
            chunks = chunk_transcript(
                text=transcript.transcript,
                chunk_size=1200,
                overlap=200
            )
            logger.info(f"[EMBEDDING] Generated {len(chunks)} chunks")
            
            if len(chunks) == 0:
                raise ValueError(f"Chunking produced 0 chunks for transcript {uid}")
            
            # Step 2: Prepare chunks for embedding model
            # Format: List of (id, url, title, text) tuples
            embedding_inputs = [
                (f"{uid}_{chunk_idx}", "", transcript.title or "", chunk_text)
                for chunk_idx, chunk_text in chunks
            ]
            
            # Step 3: Generate embeddings using TextEmbeddingsInference
            logger.info(f"[EMBEDDING] Generating embeddings for {len(chunks)} chunks")
            model = TextEmbeddingsInference()
            
            # Call embed method - returns (chunks, embeddings)
            returned_chunks, embeddings = await model.embed.remote(embedding_inputs)
            
            # Validate embedding dimensions
            if embeddings.shape[1] != 384:
                raise ValueError(
                    f"Expected 384 dimensions, got {embeddings.shape[1]}"
                )
            
            logger.info(
                f"[EMBEDDING] Generated embeddings with shape {embeddings.shape}"
            )
            
            # Step 4: Store chunks and embeddings in database
            vector_records = []
            for (chunk_idx, chunk_text), embedding_vector in zip(chunks, embeddings):
                vector_record = VectorStore(
                    transcript_id=transcript.uid,
                    chunk_text=chunk_text,
                    chunk_index=chunk_idx,
                    embedding=embedding_vector.tolist(),  # Convert numpy array to list
                    created_at=time.time()
                )
                vector_records.append(vector_record)
            
            # Batch insert all vector records
            with Session(engine, expire_on_commit=False) as session:
                session.add_all(vector_records)
                session.commit()
            
            elapsed = time.time() - start_time
            logger.info(
                f"[EMBEDDING] Completed for {uid}: "
                f"{len(chunks)} chunks embedded and stored in {elapsed:.2f}s"
            )
            
            # Step 5: Transition to next state
            await self._next_state().run_job(uid)
            return 0
            
        except Exception as err:
            logger.error(
                f"[{self.StateSymbol}] Error processing {uid}: {err}",
                exc_info=True
            )
            await self.on_error(uid)
            return -1