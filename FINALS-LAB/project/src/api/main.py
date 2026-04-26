"""FastAPI application entrypoint for Student Placement Analytics."""

from __future__ import annotations

import importlib
import logging
from contextlib import asynccontextmanager
from typing import Final

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.batch import router as batch_router
from src.api.routes.predict import router as predict_router
from src.api.routes.stats import router as stats_router
from src.api.routes.students import router as students_router

LOGGER = logging.getLogger(__name__)

APP_TITLE: Final[str] = "Student Placement Analytics API"
APP_VERSION: Final[str] = "1.0.0"


def _validate_inference_startup() -> None:
    """Ensure ML inference artifacts are loaded and ready."""
    try:
        predict_module = importlib.import_module("src.ml.predict")
        get_runtime_status = getattr(predict_module, "get_runtime_status", None)
        if not callable(get_runtime_status):
            raise RuntimeError("src.ml.predict.get_runtime_status is unavailable")

        runtime_status = get_runtime_status()
        if not isinstance(runtime_status, dict) or not runtime_status.get("model_loaded"):
            raise RuntimeError("Inference runtime status indicates model is not loaded")

        LOGGER.info(
            "Inference ready: feature_count=%s threshold=%s",
            runtime_status.get("feature_count"),
            runtime_status.get("threshold"),
        )
    except Exception as exc:
        LOGGER.exception("Startup validation failed for inference runtime")
        raise RuntimeError("Startup validation failed: inference runtime unavailable") from exc


def _register_routes(app: FastAPI) -> None:
    """Register API routers and health endpoint."""
    app.include_router(predict_router)
    app.include_router(stats_router)
    app.include_router(students_router)
    app.include_router(batch_router)

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    _validate_inference_startup()
    yield


def create_app() -> FastAPI:
    """Application factory used by ASGI servers and tests."""
    app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(app)
    return app


app = create_app()