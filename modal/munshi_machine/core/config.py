import logging
import pathlib


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# embedding model id on hf
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"

# model is stored on the app image itself at this path
MODEL_DIR = "/model"

CACHE_DIR = "/cache"
RAW_AUDIO_DIR = pathlib.Path(CACHE_DIR, "raw_audio")

# Completed episode transcriptions. Stored as flat files with
# files structured as '{guid_hash}-{model_slug}.json'.
TRANSCRIPTIONS_DIR = pathlib.Path(CACHE_DIR, "transcriptions")

# python dependencies
BASE_PYTHON_PACKAGES = [
    "google-genai==1.25.0",
    "tiktoken",
    "requests",
    "packaging",
    "wheel",
    "ffmpeg-python",
    "mutagen",
    "python-multipart",
    "httpx",
    "sqlmodel",
    "psycopg[binary]",
    "blake3",
    "fastapi",
    "pgvector"
]

ML_PYTHON_PACKAGES = [
    # Core PyTorch - using newer versions but keeping compatibility
    "torch==2.7.1",
    "torchaudio==2.7.1",
    "numpy==2.0.2",
    
    # WhisperX and dependencies
    "git+https://github.com/m-bain/whisperx.git@v3.4.0",
    "ctranslate2==4.4.0",
    
    # Additional optimizations
    "ninja",
    "hf-transfer~=0.1", 
    "pyannote.audio==3.3.2",
]

NEMO_PYTHON_PACKAGES = [
    "torch==2.7.1",
    "torchaudio==2.7.1",
    "librosa==0.11.0",
    "hf-transfer~=0.1",
    "huggingface-hub[hf-xet]==0.32.4",
    "cuda-python==12.8.0",
    "nemo-toolkit[asr]==2.4.0",
    "soundfile==0.13.1",
    "omegaconf==2.3.0",
]

PYTHON_PACKAGES = BASE_PYTHON_PACKAGES + ML_PYTHON_PACKAGES
NEMO_PYTHON_PACKAGES = BASE_PYTHON_PACKAGES + NEMO_PYTHON_PACKAGES


APT_PACKAGES = [
    "git",
    "ffmpeg",
    "libpq-dev"
]

CUDNN_PACKAGES = [
    "libcudnn8",
    "libcudnn8-dev"
]

# Collections: max items (episodes + uploads) per collection
COLLECTION_MAX_ITEMS = 50
