from fastapi import FastAPI
from .middleware import RequestLoggingMiddleware, add_cors  # noqa: E402

# Logger
from ..core import config

logger = config.get_logger("MAIN")

# FastAPI config
web_app = FastAPI()
add_cors(web_app)
web_app.add_middleware(
    RequestLoggingMiddleware,
    log_level="info"
)

# import modal functions and classes into this namespace for modal
from munshi_machine.functions.transcribe import Parakeet  # noqa: E402

ParakeetCls = Parakeet


# register v1 routers
from .v1 import (
    uploads,
    podcasts,
    episodes,
    stats,
    fetch,
    batch,
    search,
    collections,
    whispers,
)  # noqa: E402

web_app.include_router(fetch.router, prefix="")
web_app.include_router(uploads.router, prefix="")
web_app.include_router(podcasts.router, prefix="")
web_app.include_router(episodes.router, prefix="")
web_app.include_router(stats.router, prefix="")
web_app.include_router(batch.router, prefix="")
web_app.include_router(search.router, prefix="")
web_app.include_router(collections.router, prefix="")
web_app.include_router(whispers.router, prefix="")
