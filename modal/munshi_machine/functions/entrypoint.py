import modal
from modal import asgi_app

from munshi_machine.api.api import web_app
from munshi_machine.core import config
from munshi_machine.core.app import app, custom_secret
from munshi_machine.core.images import base_image
from munshi_machine.core.volumes import audio_storage_vol, transcriptions_vol


# Mount FastApi web api app
@app.function(
    image=base_image,
    volumes={
        str(config.RAW_AUDIO_DIR): audio_storage_vol,
        str(config.TRANSCRIPTIONS_DIR): transcriptions_vol,
    },
    secrets=[custom_secret],
    max_containers=4,
    scaledown_window=200,
    timeout=2000,
)
@modal.concurrent(max_inputs=100)
@asgi_app()
def entrypoint():
    return web_app


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    # from .others import init_transcription
    # test_id = "local_test123"
    # audio = init_transcription.remote(test_id)
    from munshi_machine.functions.migration import migrate_transcriptions_to_db

    print(migrate_transcriptions_to_db.remote())
