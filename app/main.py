"""Bootstrap aplikasi FastAPI (MVC) — lifespan, error handler, router, View.

Alur (arsitektur.md §3.2):
    View -> Router (file ini) -> Controller -> Service -> Model
"""

from __future__ import annotations

import logging
import logging.config
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import BASE_DIR, get_config
from app.controllers import health_controller, history_controller, predict_controller
from app.errors import AppError
from app.middleware import RateLimiter, rate_limit_guard
from app.models.ml.classifier import Classifier
from app.models.ml.face_gate import FaceGate, create_detector
from app.models.ml.labels import LabelMap
from app.schemas.error_schema import error_payload

logger = logging.getLogger("app")

VIEWS_DIR = Path(__file__).resolve().parent / "views" / "static"


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )


def _record_startup_error(errors: list[str], message: str) -> None:
    errors.append(message)
    logger.error("GAGAL STARTUP: %s", message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Muat label, model .keras, dan Face Gate SEKALI saat aplikasi mulai."""
    cfg = get_config()
    _configure_logging(cfg.log_level)

    app.state.started_at = time.time()
    app.state.ready = False
    app.state.labelmap = None
    app.state.classifier = None
    app.state.face_gate = None
    app.state.db = None
    app.state.startup_error = None

    errors: list[str] = []

    # 1) label (prasyarat numerik #1 — A3)
    try:
        app.state.labelmap = LabelMap.load(cfg.class_indices_path)
        logger.info("Label: %s", list(app.state.labelmap))
    except Exception as exc:  # noqa: BLE001
        _record_startup_error(errors, str(getattr(exc, "message", exc)))

    # 2) model .keras — tetap dicoba walau label belum ada (validasi "jumlah output ==
    #    jumlah label" otomatis dilewati); prediksi tetap ditahan oleh LABELS_NOT_READY
    try:
        app.state.classifier = Classifier.load(cfg, app.state.labelmap)
    except Exception as exc:  # noqa: BLE001
        _record_startup_error(errors, str(getattr(exc, "message", exc)))

    # 3) Face Gate (fail-closed)
    try:
        app.state.face_gate = FaceGate(create_detector(cfg), cfg)
        logger.info("Face Gate siap: %s", app.state.face_gate.detector_name)
    except Exception as exc:  # noqa: BLE001
        _record_startup_error(errors, str(getattr(exc, "message", exc)))

    # 4) Riwayat (opsional & fail-soft — gagal = fitur mati, bukan startup error)
    if cfg.history_enabled:
        try:
            from app.models.db import Database

            app.state.db = Database(cfg.database_path_resolved)
            logger.info("Riwayat aktif: %s", cfg.database_path_resolved)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Riwayat nonaktif: %s", getattr(exc, "message", exc))

    app.state.startup_error = " | ".join(errors) if errors else None
    app.state.ready = not errors

    if errors and cfg.strict_startup:
        raise RuntimeError(
            "STRICT_STARTUP=true dan komponen belum siap:\n- " + "\n- ".join(errors)
        )
    if errors:
        logger.warning(
            "Aplikasi berjalan dalam mode DEGRADED (STRICT_STARTUP=false). "
            "Endpoint prediksi akan membalas 503 sampai masalah beres."
        )

    yield

    if app.state.db is not None:
        app.state.db.close()
    app.state.ready = False


def create_app() -> FastAPI:
    cfg = get_config()

    app = FastAPI(
        title="Sistem Klasifikasi Citra Hasil Generate AI",
        version="1.0.0",
        description=(
            "Klasifikasi citra hasil generate AI (Gemini, DALL·E, Midjourney, "
            "Stable Diffusion). Hanya menerima SATU wajah manusia dengan cakupan "
            "wajah sampai pundak; gambar di luar ketentuan ditolak sebelum inferensi."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- request id + rate limit + logging akses ---------------------------
    limiter = RateLimiter(cfg.rate_limit_per_minute)
    if limiter.enabled:
        logger.info("Rate limit aktif: %d req/menit/IP", limiter.per_minute)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        request.state.request_id = request_id

        # hanya endpoint API yang dibatasi (aset View/tenaga tetap bebas)
        if request.url.path.startswith("/api/"):
            limited = await rate_limit_guard(request, limiter)
            if limited is not None:
                limited.headers["X-Request-ID"] = request_id
                return limited

        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.debug(
            "%s %s -> %s (%.0fms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    # --- error handler seragam --------------------------------------------
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        if exc.detail:
            logger.info("AppError %s: %s", exc.code, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                **error_payload(exc.code, exc.message, exc.requirement_ref),
                "request_id": getattr(request.state, "request_id", "-"),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()))
        return JSONResponse(
            status_code=422,
            content={
                **error_payload(
                    "VALIDATION_ERROR",
                    f"Field tidak valid: {loc or 'request'} — {first.get('msg', '')}",
                ),
                "request_id": getattr(request.state, "request_id", "-"),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        messages = {
            404: "Endpoint tidak ditemukan.",
            405: "Metode HTTP tidak diizinkan.",
        }
        return JSONResponse(
            status_code=exc.status_code,
            content={
                **error_payload(
                    f"HTTP_{exc.status_code}",
                    str(exc.detail) if exc.detail else messages.get(exc.status_code, "Error."),
                ),
                "request_id": getattr(request.state, "request_id", "-"),
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        logger.exception("Error tak terduga: %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                **error_payload(
                    "INFERENCE_FAILED",
                    "Terjadi kesalahan pada server. Coba lagi.",
                ),
                "request_id": getattr(request.state, "request_id", "-"),
            },
        )

    # --- router controller --------------------------------------------------
    app.include_router(predict_controller.router)
    app.include_router(health_controller.router)
    app.include_router(history_controller.router)

    # --- View (static SPA) --------------------------------------------------
    if VIEWS_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(VIEWS_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(VIEWS_DIR / "index.html")

    return app


app = create_app()


def run() -> None:  # pragma: no cover - entrypoint
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover
    run()
