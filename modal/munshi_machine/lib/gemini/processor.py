"""
Gemini Module
"""

from google import genai
from google.genai import errors

import os
import asyncio
import json
import time
from typing import Dict, Any, Optional

from .config import (
    GEMINI_MODELS, GENERATION_CONFIGS,
    TOKEN_LIMITS, ERROR_HANDLING, MODEL_FALLBACKS
)

def log_gemini(message: str, level: str = "INFO"):
    """Enhanced logging for Gemini operations"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[GEMINI {level}] {timestamp} - {message}")

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback to character count / 4 (rough approximation)
        return len(text) // 4

class GeminiProcessor:
    """Enhanced Gemini processor for transcripts and summaries"""
    
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    
    async def ask_gemini_with_retry(
        self, 
        prompt: str, 
        content: str, 
        task_type: str = "summary",
        max_retries: int = None,
        system_instruction: str = None,
    ) -> Any:
        """
        Enhanced Gemini API call with structured output support
        """
        max_retries = max_retries or ERROR_HANDLING["max_retries"]
        
        log_gemini(f"🚀 Starting Gemini request - task_type: {task_type}, content_tokens: {count_tokens(content)}")
        log_gemini(f"📋 Using model: {GEMINI_MODELS[task_type]}, max_retries: {max_retries}")
        
        # Prepare config
        config = GENERATION_CONFIGS[task_type].copy()
        if system_instruction:
            config['system_instruction'] = system_instruction
        
        # Minimal model fallback: try primary then two flash models; few retries, simple waits
        primary = GEMINI_MODELS[task_type]
        fallbacks = [m for m in MODEL_FALLBACKS if m != primary]
        model_trials = [primary, *fallbacks]

        for attempt in range(max_retries):
            log_gemini(f"Attempt {attempt + 1}/{max_retries} - task={task_type}")
            last_error: Optional[Exception] = None

            for model_name in model_trials:
                try:
                    resp = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=model_name,
                        contents=[prompt, content],
                        config=config,
                    )
                    # Validate
                    if "response_schema" in config or task_type.startswith("cleaning"):
                        if resp.parsed:
                            return resp.parsed
                        last_error = Exception("No parsed data")
                        continue
                    else:
                        if getattr(resp, "text", None):
                            return resp.text
                        last_error = Exception("Empty text")
                        continue
                except errors.APIError as api_err:
                    # Try next model; if all fail, handle after loop
                    last_error = api_err
                    continue
                except Exception as e:
                    last_error = e
                    continue

            # All models failed this attempt
            if attempt < max_retries - 1:
                # Simple wait strategy: rate-limit aware else generic
                msg = str(last_error) if last_error else ""
                if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
                    wait_time = ERROR_HANDLING["rate_limit_delay"]
                elif "timeout" in msg.lower() or "deadline" in msg.lower():
                    wait_time = ERROR_HANDLING["initial_delay"] * 2
                else:
                    wait_time = ERROR_HANDLING["initial_delay"]
                await asyncio.sleep(wait_time)
                continue
            # Exhausted
            raise last_error or Exception("Gemini request failed for all models")
    
    def _build_metadata_block(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Build minimal metadata block with only title, podcast, author. Empty string if none present."""
        if not metadata or not isinstance(metadata, dict):
            return ""
        lines = []
        def _take(key: str):
            v = metadata.get(key)
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            return s
        title = _take("title")
        podcast = _take("podcast")
        author = _take("author")
        if title:
            lines.append(f"title: {title}")
        if podcast:
            lines.append(f"podcast: {podcast}")
        if author:
            lines.append(f"author: {author}")
        return "\n".join(lines)

    async def get_cleaned_transcript(self, transcript_text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhanced transcript cleaning with JSON response parsing
        """
        start_time = time.time()
        log_gemini(f"🧹 Starting transcript cleaning - input length: {count_tokens(transcript_text)} tokens")
        
        from .prompts import clean_transcript_prompt
        
        # Check transcript length
        if count_tokens(transcript_text) < 100:
            log_gemini(f"📏 Transcript too short ({count_tokens(transcript_text)} tokens), returning as-is")
            return transcript_text
        
        log_gemini(f"📊 Using token limit: {TOKEN_LIMITS['cleaning_normal']} for chunking")
        
        # Smart chunking with sentence boundaries
        chunks = self._smart_chunk_text(transcript_text, TOKEN_LIMITS["cleaning_normal"])
        if len(chunks) > 1:
            log_gemini(f"📦 Created {len(chunks)} chunks for cleaning and starting parallel processing")
        
        # Build prompt with optional metadata block
        meta = self._build_metadata_block(metadata)
        prompt_with_meta = (meta + "\n\n" if meta else "") + clean_transcript_prompt

        # Process chunks in parallel with error handling
        tasks = [
            self.ask_gemini_with_retry(
                prompt_with_meta, 
                chunk, 
                task_type="cleaning_normal",
                max_retries=3
            ) 
            for chunk in chunks
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            log_gemini(f"📥 Received {len(results)} results from parallel processing")
            
            # Combine results with error recovery
            cleaned_chunks = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    log_gemini(f"❌ Chunk {i+1} processing failed: {str(result)}", "ERROR")
                    # Use original chunk as fallback
                    cleaned_chunks.append(chunks[i])
                    continue
                
                # Always expect a list for cleaned_text
                cleaned_text = '\n\n'.join(result.cleaned_text)
                log_gemini(f"✅ Chunk {i+1} cleaned: {count_tokens(cleaned_text)} tokens")
                cleaned_chunks.append(cleaned_text)
            
            final_result = self._merge_cleaned_chunks(cleaned_chunks)
            
            total_time = time.time() - start_time
            log_gemini(f"✅ Cleaning completed in {total_time:.1f}s - output: {count_tokens(final_result)} tokens")
            log_gemini(f"📊 Compression ratio: {count_tokens(transcript_text)} → {count_tokens(final_result)} tokens ({(count_tokens(final_result)/count_tokens(transcript_text)*100):.1f}%)")
            
            return final_result
            
        except Exception as e:
            log_gemini(f"💥 Critical error in transcript cleaning: {str(e)}", "ERROR")
            raise e

    async def get_cleaned_speaker_transcript(self, speaker_transcript: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Clean speaker transcript with JSON response parsing
        Returns dict with cleaned_transcript and speaker_mappings
        """
        start_time = time.time()
        log_gemini(f"👥 Starting speaker transcript cleaning - input length: {count_tokens(speaker_transcript)} tokens")
        
        from .prompts import clean_speaker_transcript_prompt
        
        # Check transcript length
        if count_tokens(speaker_transcript) < 100:
            log_gemini(f"📏 Speaker transcript too short ({count_tokens(speaker_transcript)} tokens), returning as-is")
            return {"cleaned_transcript": speaker_transcript, "speaker_mappings": {}}
        
        # Smart chunking for speaker transcripts (larger chunks to preserve speaker context)
        chunk_limit = int(TOKEN_LIMITS["cleaning_speaker"])
        log_gemini(f"📊 Using token limit: {chunk_limit} for speaker context preservation")
        
        chunks = self._smart_chunk_text(speaker_transcript, chunk_limit)
        log_gemini(f"📦 Created {len(chunks)} speaker chunks")
        
        # Build prompt with optional metadata block
        meta = self._build_metadata_block(metadata)
        prompt_with_meta = (meta + "\n\n" if meta else "") + clean_speaker_transcript_prompt

        # Process chunks in parallel (true concurrency across speaker chunks)
        cleaned_chunks = [None] * len(chunks)
        all_speaker_mappings = {}
        
        async def _process_chunk(idx: int, chunk: str):
            try:
                response = await self.ask_gemini_with_retry(
                    prompt_with_meta,
                    chunk,
                    task_type="cleaning_speaker",
                    max_retries=3
                )
                cleaned_text = '\n\n'.join(response.cleaned_transcript)
                cleaned_chunks[idx] = cleaned_text
                # Collect mappings if present
                if hasattr(response, 'speaker_ids') and hasattr(response, 'speaker_names'):
                    if len(response.speaker_ids) == len(response.speaker_names):
                        for j in range(len(response.speaker_ids)):
                            speaker_id = response.speaker_ids[j]
                            speaker_name = response.speaker_names[j]
                            all_speaker_mappings[speaker_id] = speaker_name
            except Exception as e:
                log_gemini(f"❌ Speaker chunk {idx+1} failed: {str(e)}", "ERROR")
                cleaned_chunks[idx] = chunk  # fallback to original chunk

        tasks = [asyncio.create_task(_process_chunk(i, c)) for i, c in enumerate(chunks)]
        await asyncio.gather(*tasks)
        
        final_transcript = self._merge_cleaned_chunks(cleaned_chunks)
        
        total_time = time.time() - start_time
        log_gemini(f"✅ Speaker cleaning completed in {total_time:.1f}s")
        log_gemini(f"📊 Final transcript: {count_tokens(final_transcript)} tokens")
        log_gemini(f"👥 Total speaker mappings: {len(all_speaker_mappings)}")
        log_gemini(f"🎭 Speaker mappings: {all_speaker_mappings}")
        
        return {
            "cleaned_transcript": final_transcript,
            "speaker_mappings": all_speaker_mappings
        }

    def _smart_chunk_text(self, text: str, max_tokens: int) -> list[str]:
        """
        Intelligent text chunking that preserves sentence boundaries and context
        """
        import tiktoken
        import re
        
        encoding = tiktoken.get_encoding("cl100k_base")
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(encoding.encode(sentence))
            
            # Handle sentences longer than max_tokens
            if sentence_tokens > max_tokens:
                log_gemini(f"⚠️ Long sentence {i+1}: {sentence_tokens} tokens > {max_tokens}, splitting", "WARN")
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                
                # Split long sentence by tokens
                tokens = encoding.encode(sentence)
                for j in range(0, len(tokens), max_tokens):
                    chunk_tokens = tokens[j:j + max_tokens]
                    token_chunk = encoding.decode(chunk_tokens)
                    chunks.append(token_chunk)
                continue
            
            # Check if adding sentence exceeds limit
            if current_tokens + sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence + " "
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        log_gemini(f"✅ Chunking complete: {len(chunks)} chunks created")
        return chunks
    
    def _merge_cleaned_chunks(self, chunks: list[str]) -> str:
        """
        Intelligently merge cleaned chunks, handling potential overlaps
        """
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Simple merge - could be enhanced with overlap detection
        merged = chunks[0]
        
        for chunk in chunks[1:]:
            # Add space if not already present
            if not merged.endswith(' ') and not chunk.startswith(' '):
                merged += " "
            merged += chunk
        
        return merged

    async def get_summary(self, vid: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Generate comprehensive summary with insights and tangents for a video ID
        Returns ComprehensiveSummaryResponse object
        """
        log_gemini(f"📋 Starting comprehensive summary generation for video {vid}")
        
        from .prompts import comprehensive_summary_prompt
        from .models import ComprehensiveSummaryResponse

        try:
            # Get transcript data
            log_gemini(f"📂 Loading transcript data for {vid}")
            
            if not text:
                log_gemini(f"❌ No transcript data found for video {vid}", "ERROR")
                return ComprehensiveSummaryResponse(
                    summary="Error: No transcript data available for summarization",
                    insights=[],
                    tangents=[]
                )
            
            transcript_text = text 
            log_gemini(f"📊 Loaded transcript: {count_tokens(transcript_text)} tokens")
            
            # Check if transcript is too short
            if count_tokens(transcript_text) < 100:
                log_gemini(f"📏 Transcript too short for summarization: {count_tokens(transcript_text)} tokens")
                return ComprehensiveSummaryResponse(
                    summary=transcript_text,
                    insights=[],
                    tangents=[]
                )
        
            # Build prompt with metadata
            meta = self._build_metadata_block(metadata)
            prompt_with_meta = (meta + "\n\n" if meta else "") + comprehensive_summary_prompt
            
            # Get structured response from Gemini
            summary_response = await self.ask_gemini_with_retry(
                prompt_with_meta,
                transcript_text,
                task_type="summary",
                max_retries=3
            )
            
            log_gemini(f"✅ Summary generated: {count_tokens(summary_response.summary)} tokens")
            log_gemini(f"💡 Extracted {len(summary_response.insights)} insights")
            log_gemini(f"🎯 Extracted {len(summary_response.tangents)} tangents")
            
            # Print insights
            if summary_response.insights:
                log_gemini("=" * 80)
                log_gemini("💡 INSIGHTS (theme-aligned):")
                for i, insight in enumerate(summary_response.insights, 1):
                    log_gemini(f"  {i}. {insight.text}")
                log_gemini("=" * 80)
            
            # Print tangents
            if summary_response.tangents:
                log_gemini("=" * 80)
                log_gemini("🎯 TANGENTS (off-topic but interesting):")
                for i, tangent in enumerate(summary_response.tangents, 1):
                    log_gemini(f"  → {tangent.text}")
                log_gemini("=" * 80)
            
            return summary_response
            
        except Exception as e:
            log_gemini(f"❌ Error generating summary: {str(e)}", "ERROR")
            raise e

    async def get_rag_answer(self, query: str, chunks: list[Dict[str, Any]]) -> str:
        """
        Generate an answer to a query based on retrieved chunks.
        """
        start_time = time.time()
        log_gemini(f"🤖 Starting RAG answer generation for query: '{query[:50]}...' with {len(chunks)} chunks")
        
        from .prompts import rag_answer_prompt, MUNSHI_SYSTEM_PROMPT
        
        # Format context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            title = chunk.get("episode_title", "Unknown Episode")
            uid = chunk.get("transcript_id", "")
            text = chunk.get("chunk_text", "")
            
            # Create a clear block for each chunk with metadata
            chunk_block = f"--- Source {i+1} ---\nTitle: {title}\nUID: {uid}\nContent:\n{text}\n"
            context_parts.append(chunk_block)
            
        context = "\n".join(context_parts)
        
        # Calculate tokens
        context_tokens = count_tokens(context)
        log_gemini(f"📚 RAG Context size: {context_tokens} tokens")
        
        full_prompt = rag_answer_prompt.format(context=context, query=query)
        
        try:
            answer_obj = await self.ask_gemini_with_retry(
                full_prompt, 
                "", # Content included in prompt
                task_type="rag_answer",
                max_retries=2,
                system_instruction=MUNSHI_SYSTEM_PROMPT
            )
            
            if hasattr(answer_obj, 'answer_html'):
                answer = answer_obj.answer_html
            else:
                answer = str(answer_obj)
            
            elapsed = time.time() - start_time
            log_gemini(f"✅ RAG answer generated in {elapsed:.2f}s: {count_tokens(str(answer))} tokens")
            return str(answer), answer_obj.answer_title

        except Exception as e:
            log_gemini(f"❌ Error generating RAG answer: {str(e)}", "ERROR")
            return "I apologize, but I was unable to generate an answer from the podcast transcripts at this time."
    
    async def generate_card_content(
        self,
        seed_insight: dict,
        similar_insights: list[dict],
        transcript_contexts: list[dict]
    ) -> dict:
        """
        Generate a structured insight card using LLM.
        
        LLM analyzes patterns and determines card type (Echo, Bridge, or Fracture).
        
        Args:
            seed_insight: The primary insight that triggered this card
            similar_insights: List of similar insights with similarity scores
            transcript_contexts: List of transcript metadata and summaries
        
        Returns:
            dict: Generated card with LLM-determined type and content
        """
        start_time = time.time()
        
        try:
            # Build context for LLM
            context = {
                "seed_insight": seed_insight["text"],
                "similar_insights": [
                    {
                        "text": ins["text"],
                        "similarity": ins["similarity"],
                        "transcript_id": ins["transcript_id"]
                    }
                    for ins in similar_insights
                ],
                "transcripts": transcript_contexts
            }
            
            # Create unified prompt where LLM decides card type
            prompt = self._create_unified_card_prompt(context)
            
            log_gemini(f"Generating card for seed insight: {seed_insight['text'][:80]}...")
            
            # Call LLM (using flash model for speed)
            result = await self.ask_gemini_with_retry(
                prompt=prompt,
                content=json.dumps(context, indent=2),
                task_type="general",
                max_retries=2
            )
            
            # Parse response
            try:
                card_data = json.loads(result)
            except json.JSONDecodeError:
                log_gemini("Failed to parse card JSON, attempting fallback", "WARNING")
                # Fallback: treat as text and structure manually
                card_data = {
                    "type": "echo",
                    "title": seed_insight["text"][:100],
                    "content": result
                }
            
            # Add metadata
            card_data["seed_insight_uid"] = seed_insight["uid"]
            card_data["involved_insight_uids"] = [seed_insight["uid"]] + [ins["uid"] for ins in similar_insights]
            card_data["source_count"] = len(similar_insights) + 1
            card_data["generated_at"] = time.time()
            
            elapsed = time.time() - start_time
            log_gemini(
                f"✅ {card_data.get('type', 'unknown').upper()} card generated in {elapsed:.2f}s: "
                f"{card_data.get('title', '')[:60]}"
            )
            
            return card_data
            
        except Exception as e:
            log_gemini(f"❌ Error generating card: {str(e)}", "ERROR")
            return None
    
    def _create_unified_card_prompt(self, context: dict) -> str:
        """Create unified prompt where LLM determines card type"""
        from .prompts import card_generation_prompt
        
        avg_sim = sum(s['similarity'] for s in context['similar_insights']) / len(context['similar_insights']) if context['similar_insights'] else 0
        
        return card_generation_prompt.format(
            seed_insight=context['seed_insight'],
            similar_count=len(context['similar_insights']),
            avg_similarity=avg_sim
        )


# Global processor instance
processor = GeminiProcessor()

async def get_cleaned_transcript(transcript_text, metadata):
    """Clean transcript using enhanced processing with JSON response"""
    return await processor.get_cleaned_transcript(transcript_text, metadata)


async def get_cleaned_speaker_transcript(speaker_transcript, metadata):
    """Clean speaker transcript with JSON response parsing"""
    return await processor.get_cleaned_speaker_transcript(speaker_transcript, metadata)


async def get_rag_answer(query: str, chunks: list[Dict[str, Any]]) -> str:
    """Generate RAG answer from chunks"""
    return await processor.get_rag_answer(query, chunks)


async def generate_card_content(
    seed_insight: dict,
    similar_insights: list[dict],
    transcript_contexts: list[dict]
) -> dict:
    """Generate insight card using LLM (LLM determines card type)"""
    return await processor.generate_card_content(
        seed_insight, similar_insights, transcript_contexts
    )
