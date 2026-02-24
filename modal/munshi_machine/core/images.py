from modal import Image
import modal
from . import config

def download_parakeet_models():
    """Pre-download Parakeet model to the image to reduce cold starts"""
    import os

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    try:
        import nemo.collections.asr as nemo_asr

        print("📥 Pre-downloading Parakeet model...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v3"
        )

        target_path = os.path.join(config.MODEL_DIR, "parakeet-tdt-0.6b-v3.nemo")
        asr_model.save_to(target_path)
        print(f"✅ Parakeet model saved to {target_path}")
    except Exception as e:
        print(f"⚠️ Parakeet model download failed (will download at runtime): {e}")

def download_embedding_model():
    """Pre-download the BGE embedding model for text-embeddings-inference"""
    import os
    from huggingface_hub import snapshot_download
    
    model_id = config.EMBEDDING_MODEL_ID 
    cache_dir = config.MODEL_DIR
    
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        print(f"📥 Pre-downloading embedding model: {model_id}...")
        # Download to the cache directory in HuggingFace format
        # text-embeddings-inference expects the model in HF cache structure
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
        )
        print(f"✅ Embedding model downloaded to {cache_dir}")
    except Exception as e:
        print(f"⚠️ Embedding model download failed (will download at runtime): {e}")


"""
Base Image to run get_audio function
"""
base_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install(*config.APT_PACKAGES)
    .pip_install(*config.BASE_PYTHON_PACKAGES)
)


nemo_image = (
    Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CXX": "g++", "CC": "g++"})
    .apt_install(*config.APT_PACKAGES)
    .pip_install(*config.NEMO_PYTHON_PACKAGES)
    # Pre-download Parakeet model for faster cold starts
    .run_function(download_parakeet_models)
)

DOCKER_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0",
    # "ghcr.io/huggingface/text-embeddings-inference:0.4.0" # Ampere 80 for A100s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.3.0"  # Turing for T4s.
)
tei_image = (
    modal.Image.from_registry(
        DOCKER_IMAGE[0],
        add_python="3.11",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .uv_pip_install("httpx", "numpy", "huggingface_hub", *config.BASE_PYTHON_PACKAGES)
    .run_function(download_embedding_model)
)