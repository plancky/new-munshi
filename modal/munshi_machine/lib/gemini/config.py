# Gemini AI Configuration
from .models import GeminiResponse, GeminiSpeakerResponse, RagAnswerResponse, ComprehensiveSummaryResponse
from google.genai import types

# Model configurations
GEMINI_MODELS = {
    "summary": "gemini-2.5-pro",
    "rag_answer": "gemini-3-flash-preview",
    "cleaning_normal": "gemini-2.5-flash-lite",
    "cleaning_speaker": "gemini-2.5-flash-lite",
    "general": "gemini-2.5-flash",  # For card generation and other general tasks
}

# Model fallbacks (in order) to try if the primary model errors or returns unusable output
MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# Generation configurations for different tasks
GENERATION_CONFIGS = {
    "summary": {
        "temperature": 0.7,
        "max_output_tokens": 60000,
        "top_p": 0.9,
        "response_mime_type": "application/json",
        "response_schema": ComprehensiveSummaryResponse,
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    },
    "rag_answer": {
        "temperature": 0.3,
        "max_output_tokens": 8192,
        "top_p": 0.8,
        "response_mime_type": "application/json",
        "response_schema": RagAnswerResponse,
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    },
    "cleaning_normal": {
        "temperature": 0.2,
        "max_output_tokens": 65000,
        "response_mime_type": "application/json",
        "response_schema": GeminiResponse,
        "thinking_config": types.ThinkingConfig(thinking_budget=0)
    },
    "cleaning_speaker": {
        "temperature": 0.2,
        "max_output_tokens": 65000,    
        "response_mime_type": "application/json",
        "response_schema": GeminiSpeakerResponse,
        "thinking_config": types.ThinkingConfig(thinking_budget=0)
    },
    "general": {
        "temperature": 0.7,
        "max_output_tokens": 8192,
        "top_p": 0.9,
        "response_mime_type": "application/json",
        "thinking_config": types.ThinkingConfig(thinking_budget=0)
    },
}

# Request timeouts (in seconds)
REQUEST_TIMEOUTS = {
    "summary": 600,
    "rag_answer": 120,
    "cleaning_normal": 600,
    "cleaning_speaker": 600,
    "general": 180,
}

# Token limits for processing
TOKEN_LIMITS = {
    "cleaning_normal": 20000,
    "cleaning_speaker": 20000,
    "summary": 800000,
    "rag_answer": 100000,
    "general": 100000,
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "min_transcript_length": 100,  # Minimum tokens for processing
    "max_chunk_overlap": 50,       # Tokens to overlap between chunks
    "confidence_threshold": 0.7,   # Minimum confidence for auto-processing
}

# Error handling strategies
ERROR_HANDLING = {
    "max_retries": 5,
    "initial_delay": 60,  # Start with 60 seconds for rate limit errors
    "rate_limit_delay": 60,  # Wait 60 seconds for 429 errors
    "log_errors": True,
} 