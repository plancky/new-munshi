import asyncio
import json
import subprocess

from munshi_machine.core.app import app, custom_secret
from munshi_machine.core.config import EMBEDDING_MODEL_ID, MODEL_DIR
from munshi_machine.core.images import base_image, tei_image

import modal

# We first set out configuration variables for our script.
## Embedding Containers Configuration
GPU_CONCURRENCY = 100
GPU_CONFIG = "A10G"
MODEL_SLUG = EMBEDDING_MODEL_ID.split("/")[-1]
BATCH_SIZE = 512

## Dataset-Specific Configuration
MODEL_CACHE_VOLUME = modal.Volume.from_name(
    "embedding-model-cache", create_if_missing=True
)
DATASET_NAME = "wikipedia"
DATASET_READ_VOLUME = modal.Volume.from_name(
    "embedding-wikipedia", create_if_missing=True
)
EMBEDDING_CHECKPOINT_VOLUME = modal.Volume.from_name(
    "checkpoint", create_if_missing=True
)
DATASET_DIR = "/data"
CHECKPOINT_DIR = "/checkpoint"
SAVE_TO_DISK = False

## HF Text-Embedding Inference specific Configuration
LAUNCH_FLAGS = [
    "--model-id",
    EMBEDDING_MODEL_ID,
    "--port",
    "8000",
    "--max-client-batch-size",
    str(BATCH_SIZE),
    "--max-batch-tokens",
    str(BATCH_SIZE * 512),
    "--huggingface-hub-cache",
    MODEL_DIR,
]



def spawn_server() -> subprocess.Popen:
    import socket

    process = subprocess.Popen(["text-embeddings-router"] + LAUNCH_FLAGS)
    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")


with tei_image.imports():
    import numpy as np


def generate_chunks_from_dataset(xs, chunk_size: int):
    """
    Generate chunks from a dataset.

    Args:
        xs (list): The dataset containing dictionaries with "id", "url", "title", and "text" keys.
        chunk_size (int): The size of each chunk.

    Yields:
        tuple: A tuple containing the id, url, title, and a chunk of text.
    """
    for data in xs:
        id_ = data["id"]
        url = data["url"]
        title = data["title"]
        text = data["text"]
        for chunk_start in range(0, len(text), chunk_size):
            yield (
                id_,
                url,
                title,
                text[chunk_start : chunk_start + chunk_size],
            )


def generate_batches(xs, batch_size):
    batch = []
    for x in xs:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@app.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    max_containers=GPU_CONCURRENCY,
    retries=3,
)
@modal.concurrent(max_inputs=10)
class TextEmbeddingsInference:
    @modal.enter()
    def open_connection(self):
        # If the process is running for a long time, the client does not seem to close the connections, results in a pool timeout
        from httpx import AsyncClient

        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=30)

    @modal.exit()
    def terminate_connection(self):
        self.process.terminate()

    async def _embed(self, chunk_batch):
        texts = [chunk[3] for chunk in chunk_batch]
        res = await self.client.post("/embed", json={"inputs": texts})
        return np.array(res.json())

    @modal.method()
    async def embed(self, chunks):
        """Embeds a list of texts.  id, url, title, text = chunks[0]"""
        coros = [
            self._embed(chunk_batch)
            for chunk_batch in generate_batches(chunks, batch_size=BATCH_SIZE)
        ]

        embeddings = np.vstack(await asyncio.gather(*coros))
        return chunks, embeddings


def load_dataset_from_disk(down_scale: float = 0.01):
    """
    Load a dataset from disk and return a subset of the training data.

    Args:
        down_scale (float): The fraction of the training data to select. Defaults to 0.01.

    Returns:
        Dataset: A subset of the training data.
    """
    import time

    from datasets import load_from_disk

    start = time.perf_counter()
    # Load the dataset as a Hugging Face dataset
    print(f"Loading dataset from {DATASET_DIR}/wikipedia")
    dataset = load_from_disk(f"{DATASET_DIR}/wikipedia")
    print(f"Dataset loaded in {time.perf_counter() - start:.2f} seconds")

    # Extract the total size of the dataset
    ttl_size = len(dataset["train"])

    sample_size = int(ttl_size * down_scale)

    return dataset["train"].select(range(sample_size))


def save_dataset_to_intermediate_checkpoint(acc_chunks, embeddings, batch_size):
    """Saves the dataset to an intermediate checkpoint.

    Args:
        acc_chunks (list): Accumulated chunks
        embeddings (list): Accumulated embeddings
        batch_size (int): Batch size
    """
    import pyarrow as pa
    from datasets import Dataset

    table = pa.Table.from_arrays(
        [
            pa.array([chunk[0] for chunk in acc_chunks]),  # id
            pa.array([chunk[1] for chunk in acc_chunks]),  # url
            pa.array([chunk[2] for chunk in acc_chunks]),  # title
            pa.array([chunk[3] for chunk in acc_chunks]),  # text
            pa.array(embeddings),
        ],
        names=["id", "url", "title", "text", "embedding"],
    )
    path_parent_folder = f"{CHECKPOINT_DIR}/{MODEL_SLUG}-{batch_size}"
    dataset = Dataset(table)
    dataset.save_to_disk(path_parent_folder)
    EMBEDDING_CHECKPOINT_VOLUME.commit()
    print(f"Saved checkpoint at {path_parent_folder}")

@app.function(
    image=modal.Image.debian_slim().uv_pip_install(
        "datasets", "pyarrow", "hf_transfer", "huggingface_hub"
    ),
    volumes={
        DATASET_DIR: DATASET_READ_VOLUME,
        CHECKPOINT_DIR: EMBEDDING_CHECKPOINT_VOLUME,
        MODEL_DIR: MODEL_CACHE_VOLUME,
    },
    timeout=86400,
    secrets=[custom_secret],
)
def embed_dataset(down_scale: float = 1, batch_size: int = 512 * 50):
    """
    Embeds a dataset with the Text Embeddings Inference container.

    Args:
        down_scale (float): The fraction of the training data to select. Defaults to 1.
        batch_size (int): The batch size to use. Defaults to 512 * 50.

    Returns:
        dict: A dictionary containing the benchmark results.
    """
    import datetime
    import time

    dataset_chars = 19560538957  # sum(map(len, dataset["train"]["text"]))
    subset = load_dataset_from_disk(down_scale)
    model = TextEmbeddingsInference()
    text_chunks = generate_chunks_from_dataset(subset, chunk_size=512)
    batches = generate_batches(text_chunks, batch_size=batch_size)

    start = time.perf_counter()
    acc_chunks = []
    embeddings = []
    for resp in model.embed.map(
        batches,
        order_outputs=False,
        return_exceptions=True,
        wrap_return_exceptions=False,
    ):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue

        batch_chunks, batch_embeddings = resp

        acc_chunks.extend(batch_chunks)
        embeddings.extend(batch_embeddings)

    end = time.perf_counter()

    duration = end - start
    characters = sum(map(len, [chunk[3] for chunk in acc_chunks]))
    characters_per_sec = int(characters / duration)
    extrapolated_duration_cps_fmt = str(
        datetime.timedelta(seconds=dataset_chars / characters_per_sec)
    )
    resp = {
        "downscale": down_scale,
        "batch_size": batch_size,
        "n_gpu": GPU_CONCURRENCY,
        "duration_mins": duration / 60,
        "characters_per_sec": characters_per_sec,
        "extrapolated_duration": extrapolated_duration_cps_fmt,
    }

    if SAVE_TO_DISK:
        save_dataset_to_intermediate_checkpoint(acc_chunks, embeddings, batch_size)

    return resp


@app.local_entrypoint()
def full_job():
    batch_size = 512 * 150
    with open("benchmarks.json", "a") as f:
        benchmark = embed_dataset.remote(batch_size=batch_size)
        f.write(json.dumps(benchmark, indent=2) + "\n")
    

@app.function(
    image=base_image,
    timeout=86400,  # 24 hours max
    secrets=[custom_secret],
    retries=3,
)
async def embed_transcript(transcript_uid: str) -> dict:
    """
    Generate embeddings for a transcript and store in vector_store table.
    
    This function runs asynchronously after transcription completes.
    It does not block the main processing pipeline.
    
    Args:
        transcript_uid: UUID of the transcript to embed
    
    Returns:
        dict: Summary of embedding generation (chunk count, duration, etc.)
    """
    import time
    from uuid import UUID

    from munshi_machine.core import config
    from munshi_machine.db.crud.transcript import find_transcript_by_uid
    from munshi_machine.db.database import engine
    from munshi_machine.lib.chunking_utils import chunk_transcript
    from munshi_machine.models.private.vector_store import VectorStore
    from sqlmodel import Session
    
    logger = config.get_logger(__name__)
    start_time = time.perf_counter()
    
    try:
        # Load transcript from database
        with Session(engine, expire_on_commit=False) as session:
            transcript = find_transcript_by_uid(UUID(transcript_uid), session)
            
            if not transcript:
                raise ValueError(f"Transcript {transcript_uid} not found")
            
            # Prefer cleaned_transcript, fallback to raw transcript
            text_to_embed = transcript.cleaned_transcript or transcript.transcript
            
            if not text_to_embed or len(text_to_embed.strip()) == 0:
                raise ValueError(f"Transcript {transcript_uid} has no content")
            
            if transcript.embeddings_generated:
                logger.info(f"[EMBEDDING] Embeddings already generated for {transcript_uid}, skipping...")
                return 0

            # Check if already generated (idempotency)
            if transcript.embeddings_generated:
                logger.info(
                    f"[EMBED_ASYNC] Embeddings already exist for {transcript_uid}, skipping"
                )
                return {
                    "transcript_uid": transcript_uid,
                    "success": True,
                    "skipped": True,
                    "reason": "already_generated"
                }
            
            logger.info(
                f"[EMBED_ASYNC] Starting embedding for {transcript_uid} "
                f"(using {'cleaned' if transcript.cleaned_transcript else 'raw'} transcript, "
                f"length: {len(text_to_embed)} chars)"
            )
            
            # Store transcript text for processing (since we'll close session)
            transcript_text = text_to_embed
            transcript_title = transcript.title
        
        # Step 1: Chunk the transcript
        chunks = chunk_transcript(
            text=transcript_text,
            chunk_size=1200,
            overlap=200
        )
        logger.info(f"[EMBED_ASYNC] Generated {len(chunks)} chunks")
        
        if len(chunks) == 0:
            raise ValueError(f"Chunking produced 0 chunks for {transcript_uid}")
        
        # Step 2: Prepare for embedding
        embedding_inputs = [
            (f"{transcript_uid}_{chunk_idx}", "", transcript_title or "", chunk_text)
            for chunk_idx, chunk_text in chunks
        ]
        
        # Step 3: Generate embeddings
        logger.info(f"[EMBED_ASYNC] Generating embeddings for {len(chunks)} chunks")
        model = TextEmbeddingsInference()
        returned_chunks, embeddings = model.embed.remote(embedding_inputs)
        
        # Validate dimensions
        if embeddings.shape[1] != 384:
            raise ValueError(f"Expected 384 dimensions, got {embeddings.shape[1]}")
        
        logger.info(f"[EMBED_ASYNC] Generated embeddings: shape {embeddings.shape}")
        
        # Step 4: Store in database (transaction)
        with Session(engine, expire_on_commit=False) as session:
            # Create vector records
            vector_records = []
            for (chunk_idx, chunk_text), embedding_vector in zip(chunks, embeddings):
                vector_record = VectorStore(
                    transcript_id=UUID(transcript_uid),
                    chunk_text=chunk_text,
                    chunk_index=chunk_idx,
                    embedding=embedding_vector.tolist(),
                    created_at=time.time()
                )
                vector_records.append(vector_record)
            
            # Insert all vectors
            session.add_all(vector_records)
            
            # Update transcript.embeddings_generated = True
            transcript = find_transcript_by_uid(UUID(transcript_uid), session)
            transcript.embeddings_generated = True
            
            # Commit both vectors and flag update in single transaction
            session.commit()
        
        elapsed = time.perf_counter() - start_time
        result = {
            "transcript_uid": transcript_uid,
            "chunk_count": len(chunks),
            "duration_sec": elapsed,
            "success": True,
            "skipped": False
        }
        
        logger.info(
            f"[EMBED_ASYNC] Completed {transcript_uid}: "
            f"{len(chunks)} chunks in {elapsed:.2f}s, flag set to True"
        )
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"[EMBED_ASYNC] Failed for {transcript_uid} after {elapsed:.2f}s: {e}",
            exc_info=True
        )
        
        return {
            "transcript_uid": transcript_uid,
            "success": False,
            "error": str(e),
            "duration_sec": elapsed
        }




@app.function(
    image=base_image,
    timeout=300,  # 5 minutes
    secrets=[custom_secret],
    retries=3,
)
async def embed_insights(transcript_uid: str) -> dict:
    """
    Generate embeddings for individual insights from a transcript.
    
    This function:
    1. Fetches transcript from DB
    2. Parses JSON from transcript.summary field
    3. Extracts insights (tangents are not embedded)
    4. Embeds each insight individually
    5. Stores in insight_vectors table
    
    Args:
        transcript_uid: UUID of the transcript to process
    
    Returns:
        dict: Summary with counts and success status
    """
    import time
    from uuid import UUID

    from munshi_machine.core import config
    from munshi_machine.db.crud.transcript import find_transcript_by_uid
    from munshi_machine.db.database import engine
    from munshi_machine.models.private.insight_vector import InsightVector, InsightType
    from sqlmodel import Session, select, func
    
    logger = config.get_logger(__name__)
    start_time = time.perf_counter()
    
    try:
        # Load transcript from database
        with Session(engine, expire_on_commit=False) as session:
            # Convert to UUID if string, otherwise use as-is
            uid = UUID(transcript_uid) if isinstance(transcript_uid, str) else transcript_uid
            transcript = find_transcript_by_uid(uid, session)
            
            if not transcript:
                raise ValueError(f"Transcript {transcript_uid} not found")
            
            if not transcript.summary:
                logger.warning(f"[EMBED_INSIGHTS] No summary for {transcript_uid}, skipping")
                return {
                    "transcript_uid": transcript_uid,
                    "success": True,
                    "skipped": True,
                    "reason": "no_summary"
                }
            
            # Check if insights already embedded
            existing_count = session.exec(
                select(InsightVector)
                .where(InsightVector.transcript_id == UUID(transcript_uid))
            ).all()
            
            if len(existing_count) > 0:
                logger.info(f"[EMBED_INSIGHTS] {len(existing_count)} insights already embedded for {transcript_uid}, skipping")
                return {
                    "transcript_uid": transcript_uid,
                    "success": True,
                    "skipped": True,
                    "reason": "already_embedded",
                    "existing_count": len(existing_count)
                }
            
            # Parse JSON from summary
            logger.info(f"[EMBED_INSIGHTS] Parsing summary for {transcript_uid}")
            
            try:
                summary_data = json.loads(transcript.summary)
                insights = summary_data.get("insights", [])
                tangents = summary_data.get("tangents", [])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"[EMBED_INSIGHTS] Could not parse summary as JSON for {transcript_uid}, skipping")
                return {
                    "transcript_uid": transcript_uid,
                    "success": True,
                    "skipped": True,
                    "reason": "invalid_json"
                }
            
            # Collect only insights to embed (tangents don't need embeddings)
            texts_to_embed = []
            metadata = []  # Store type and index for each text
            
            for idx, insight in enumerate(insights):
                text = insight.get("text", "").strip()
                if text:
                    texts_to_embed.append(text)
                    metadata.append({"type": InsightType.INSIGHT, "index": idx, "text": text})
            
            if len(texts_to_embed) == 0:
                logger.warning(f"[EMBED_INSIGHTS] No insights found for {transcript_uid}")
                return {
                    "transcript_uid": transcript_uid,
                    "success": True,
                    "skipped": True,
                    "reason": "no_insights"
                }
            
            logger.info(f"[EMBED_INSIGHTS] Found {len(metadata)} insights to embed")
        
        # Step 2: Generate embeddings (outside session to avoid timeouts)
        logger.info(f"[EMBED_INSIGHTS] Generating embeddings for {len(texts_to_embed)} items")
        
        # Prepare inputs in format expected by TextEmbeddingsInference
        embedding_inputs = [
            (f"{transcript_uid}_{i}", "", "", text)
            for i, text in enumerate(texts_to_embed)
        ]
        
        model = TextEmbeddingsInference()
        _, embeddings = model.embed.remote(embedding_inputs)
        
        logger.info(f"[EMBED_INSIGHTS] Generated embeddings: shape {embeddings.shape}")
        
        # Validate dimensions
        if embeddings.shape[1] != 384:
            raise ValueError(f"Expected 384 dimensions, got {embeddings.shape[1]}")
        
        # Step 3: Store in database
        with Session(engine, expire_on_commit=False) as session:
            vector_records = []
            for meta, embedding_vector in zip(metadata, embeddings):
                vector_record = InsightVector(
                    transcript_id=UUID(transcript_uid),
                    text=meta["text"],
                    type=meta["type"],
                    index=meta["index"],
                    embedding=embedding_vector.tolist(),
                    created_at=time.time()
                )
                vector_records.append(vector_record)
            
            session.add_all(vector_records)
            session.commit()
        
        elapsed = time.perf_counter() - start_time
        
        logger.info(
            f"[EMBED_INSIGHTS] Completed {transcript_uid}: "
            f"{len(metadata)} insights embedded in {elapsed:.2f}s"
        )
        
        # Check if we should auto-trigger card generation
        try:
            with Session(engine, expire_on_commit=False) as session:
                total_insights = session.exec(
                    select(func.count(InsightVector.uid))
                ).one()
                
                # Import here to avoid circular dependency
                from munshi_machine.models.private.insight_card import InsightCard
                
                # Try to count cards - table may not exist yet
                try:
                    total_cards = session.exec(
                        select(func.count(InsightCard.uid))
                    ).one()
                except Exception:
                    # Table doesn't exist yet, skip auto-generation
                    logger.info("[AUTO_GEN] InsightCard table not found, skipping auto-generation")
                    total_cards = None
                
                if total_cards is not None:
                    # Auto-generate cards if:
                    # - We have at least 10 insights
                    # - We have fewer than 5 cards OR insights-to-cards ratio > 5:1
                    should_generate = (
                        total_insights >= 10 and
                        (total_cards < 5 or total_insights / max(total_cards, 1) > 5)
                    )
                    
                    if should_generate:
                        logger.info(
                            f"[AUTO_GEN] Triggering card generation: "
                            f"{total_insights} insights, {total_cards} cards"
                        )
                        from munshi_machine.functions.generate_cards import generate_all_cards
                        function_call = generate_all_cards.spawn()
                        logger.info(
                            f"[AUTO_GEN] Card generation job spawned: {function_call.object_id}"
                        )
                
        except Exception as auto_gen_err:
            logger.debug(f"Auto card generation check skipped: {auto_gen_err}")
        
        return {
            "transcript_uid": transcript_uid,
            "success": True,
            "skipped": False,
            "insight_count": len(metadata),
            "duration_sec": elapsed
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(
            f"[EMBED_INSIGHTS] Failed for {transcript_uid} after {elapsed:.2f}s: {e}",
            exc_info=True
        )
        
        return {
            "transcript_uid": transcript_uid,
            "success": False,
            "error": str(e),
            "duration_sec": elapsed
        }


@app.function(
    image=tei_image,
    timeout=120,  # 2 minutes for search
    secrets=[custom_secret],
    retries=2,
)
async def search_over_transcripts(
    query: str,
    podcast_uid: str | None = None,
    collection_uid: str | None = None,
    limit: int = 50
) -> dict:
    """
    Semantic search within a specific podcast's or collection's episodes.
    
    Searches across all COMPLETED transcripts with embeddings in the target podcast or collection.
    Uses vector similarity search to find the most relevant text chunks.
    
    Args:
        query: The search query text
        podcast_uid: UUID of the podcast to search within (optional)
        collection_uid: UUID of the collection to search within (optional)
        limit: Maximum number of chunks to return (default: 50)
    
    Returns:
        dict: Search results with matching chunks and metadata
    """
    import time

    from munshi_machine.core import config
    from munshi_machine.db.crud.vector_store import similarity_search
    from munshi_machine.db.database import engine
    from munshi_machine.lib.gemini.processor import get_rag_answer
    from munshi_machine.lib.search_helpers import \
        get_eligible_transcripts_for_podcast, \
        get_eligible_transcripts_for_collection
    from munshi_machine.models.private.transcript import Transcript
    from sqlalchemy.orm import selectinload
    from sqlmodel import Session, select
    
    logger = config.get_logger("search_podcast")
    start_time = time.perf_counter()
    
    target_uid = podcast_uid or collection_uid
    search_type = "podcast" if podcast_uid else "collection"
    
    logger.info(f"[SEARCH] Starting search | Query: '{query[:100]}...' | Type: {search_type} | ID: {target_uid} | Limit: {limit}")
    
    try:
        # Validate inputs
        if not query or len(query.strip()) == 0:
            logger.warning("[SEARCH] Empty query provided")
            return {
                "success": False,
                "error": "Query cannot be empty"
            }
            
        if not podcast_uid and not collection_uid:
            logger.warning("[SEARCH] No search target provided")
            return {
                "success": False,
                "error": "Either podcast_uid or collection_uid must be provided"
            }
        
        logger.info(f"[SEARCH] Step 1/5: Fetching {search_type} and filtering transcripts")
        fetch_start = time.perf_counter()
        
        # Step 1: Get target (podcast or collection) and filter transcripts
        if podcast_uid:
            fetch_result = get_eligible_transcripts_for_podcast(podcast_uid)
        else:
            fetch_result = get_eligible_transcripts_for_collection(collection_uid)
        
        if not fetch_result["success"]:
            logger.warning(f"[SEARCH] {fetch_result['error']}")
            return fetch_result
            
        eligible_transcript_ids = fetch_result["eligible_transcript_ids"]
        stats = fetch_result["stats"]
        
        # Extract metadata based on type
        if podcast_uid:
            title = fetch_result.get("podcast_title")
            author = fetch_result.get("podcast_author")
            logger.info(f"[SEARCH] Found podcast: '{title}' by {author}")
        else:
            title = fetch_result.get("collection_title")
            author = fetch_result.get("collection_author")
            logger.info(f"[SEARCH] Found collection: '{title}' by {author}")
        
        logger.info(
            f"[SEARCH] Transcript status breakdown: "
            f"Total={stats['total_transcripts']}, "
            f"Completed={stats['completed_transcripts']}, "
            f"WithEmbeddings={stats['transcripts_with_embeddings']}, "
            f"Eligible={stats['eligible_count']}"
        )
        
        logger.info(
            f"[SEARCH] Eligible transcript IDs: "
            f"{[str(uid)[:8] + '...' for uid in eligible_transcript_ids[:5]]}"
            f"{' (+ more)' if len(eligible_transcript_ids) > 5 else ''}"
        )
        fetch_duration = time.perf_counter() - fetch_start
        
        # Step 2: Convert query to embedding
        logger.info(f"[SEARCH] Step 2/5: Generating embedding for query (length: {len(query)} chars)")
        embed_start = time.perf_counter()
        
        # Prepare query in format expected by TextEmbeddingsInference
        embedding_input = [("query", "", "", query)]
        
        model = TextEmbeddingsInference()
        _, embeddings = model.embed.remote(embedding_input)
        query_embedding = embeddings[0].tolist()
        embed_duration = time.perf_counter() - embed_start
        
        logger.info(
            f"[SEARCH] Query embedding generated: {len(query_embedding)} dimensions | "
            f"Duration: {embed_duration:.3f}s"
        )
        
        # Step 3: Perform similarity search
        logger.info(
            f"[SEARCH] Step 3/5: Performing vector similarity search | "
            f"Searching across {len(eligible_transcript_ids)} transcripts"
        )
        search_start = time.perf_counter()
        
        with Session(engine, expire_on_commit=False) as session:
            results = similarity_search(
                query_embedding=query_embedding,
                limit=limit,
                transcript_ids=eligible_transcript_ids,
                session=session
            )
            search_duration = time.perf_counter() - search_start
            
            logger.info(
                f"[SEARCH] Similarity search completed: {len(results)} results | "
                f"Duration: {search_duration:.3f}s"
            )
            
            if len(results) > 0:
                # Log distance statistics
                distances = [float(dist) for _, dist in results]
                logger.info(
                    f"[SEARCH] Distance statistics: "
                    f"Min={min(distances):.4f}, "
                    f"Max={max(distances):.4f}, "
                    f"Avg={sum(distances)/len(distances):.4f}"
                )
            
            # Step 4: Format results
            logger.info("[SEARCH] Step 4/5: Formatting results with transcript metadata")
            
            format_start = time.perf_counter()
            
            # Fetch all transcripts with relationships in one query for efficiency
            transcript_ids_in_results = list(set(vs.transcript_id for vs, _ in results))
            transcript_map = {}
            
            if transcript_ids_in_results:
                stmt = (
                    select(Transcript)
                    .where(Transcript.uid.in_(transcript_ids_in_results))
                    .options(selectinload(Transcript.podcast))
                )
                transcripts = session.exec(stmt).all()
                transcript_map = {t.uid: t for t in transcripts}
                logger.info(f"[SEARCH] Loaded {len(transcript_map)} transcript metadata records")
            
            formatted_results = []
            for vector_store, distance in results:
                # Fetch transcript from map
                transcript = transcript_map.get(vector_store.transcript_id)
                
                formatted_results.append({
                    "chunk_text": vector_store.chunk_text,
                    "chunk_index": vector_store.chunk_index,
                    "distance": float(distance),
                    "transcript_id": str(vector_store.transcript_id),
                    "episode_title": transcript.title if transcript else None,
                    "episode_guid": transcript.episode_guid if transcript else None,
                })
            
            format_duration = time.perf_counter() - format_start
            logger.info(f"[SEARCH] Results formatted | Duration: {format_duration:.3f}s")
            
            # Step 5: Generate answer using LLM (RAG)
            logger.info("[SEARCH] Step 5/5: Generating RAG answer using Gemini")
            generate_start = time.perf_counter()
            
            generated_answer = None
            if formatted_results:
                try:
                    generated_answer, answer_title = await get_rag_answer(query[:1000], formatted_results)
                    logger.info("[SEARCH] Answer generated successfully")
                except Exception as e:
                    logger.error(f"[SEARCH] RAG generation failed: {e}")
                    generated_answer = "Error generating answer."
            else:
                generated_answer = "No relevant context found to answer the query."
                
            generate_duration = time.perf_counter() - generate_start
            logger.info(f"[SEARCH] RAG generation completed | Duration: {generate_duration:.3f}s")
        
        elapsed = time.perf_counter() - start_time
        
        logger.info(
            f"[SEARCH] Search completed successfully | "
            f"Total duration: {elapsed:.3f}s | "
            f"Results: {len(formatted_results)} | "
            f"Query: '{query[:50]}...'"
        )
        
        return {
            "success": True,
            "query": query,
            "answer_title": answer_title or "Answer",
            "answer": generated_answer,
            "podcast_uid": podcast_uid,
            "collection_uid": collection_uid,
            "title": title,
            "author": author,
            # Backward compatibility fields
            "podcast_title": title if podcast_uid else None,
            "podcast_author": author if podcast_uid else None,
            "results": formatted_results,
            "result_count": len(formatted_results),
            "eligible_transcript_count": len(eligible_transcript_ids),
            "search_duration_sec": elapsed,
            "timing": {
                "fetch_sec": fetch_duration,
                "embedding_sec": embed_duration,
                "search_sec": search_duration,
                "formatting_sec": format_duration,
                "generate_sec": generate_duration,
                "total_sec": elapsed
            }
        }
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(
            f"[SEARCH] Search failed | "
            f"Duration: {elapsed:.3f}s | "
            f"Error: {str(e)} | "
            f"Query: '{query[:50]}...' | "
            f"Target ID: {target_uid}",
            exc_info=True
        )
        return {
            "success": False,
            "error": str(e),
            "search_duration_sec": elapsed
        }
