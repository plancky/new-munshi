from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from modal import is_local
from starlette.requests import Request
from starlette.responses import Response
from munshi_machine.core.config import get_logger
import time

from munshi_machine.core.env import ENV_VARS

logger = get_logger("RequestsLogger")

def add_cors(app: FastAPI):
    import os

    frontend_app_url = os.environ.get(ENV_VARS.FRONTEND_APP_URL)
    origins = [frontend_app_url]
    is_dev_container = os.environ.get(ENV_VARS.ENV, "development") == "development"
    if is_local() or is_dev_container:
        origins = ["*"]
        print("Running in dev mode...")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS", "DELETE"],
        allow_headers=["*"],
    )
    return app

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, log_level="info", exclude_paths: list[str] = None):
        super().__init__(app)
        self.log_level = log_level
        self.exclude_paths = exclude_paths or []

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        log = getattr(logger, self.log_level)

        client = request.client.host
        method = request.method
        path = request.url.path

        log(f"Received {method} request for {path} from {client}")

        start = time.time()
        response: Response = await call_next(request)
        duration_ms = (time.time() - start) * 1000

        log(
            f"Completed {method} {path} from {client} "
            f"with status {response.status_code} in {duration_ms:.2f} ms"
        )

        return response