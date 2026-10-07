"""Controller kesehatan & metadata — laporan kesiapan komponen + ketentuan §2.2."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from app.config import Config, get_config
from app.services.requirements_notice import requirements_notice

router = APIRouter(tags=["system"])


def _uptime(request: Request) -> float:
    started = getattr(request.app.state, "started_at", None)
    if not started:
        return 0.0
    return round(__import__("time").time() - started, 1)


@router.get(
    "/health",
    summary="Liveness/readiness probe",
    description=(
        "Selalu 200 (liveness) — dipakai untuk memastikan proses hidup. "
        "Dengan `?ready=true` membalas 503 bila model/detektor/label belum siap "
        "(readiness — dipakai Docker HEALTHCHECK & orkestrator)."
    ),
)
def health(
    request: Request,
    ready: bool = Query(False, description="true = mode readiness (503 bila belum siap)"),
) -> dict:
    state = request.app.state
    payload = {
        "status": "ok" if getattr(state, "ready", False) else "degraded",
        "model_loaded": getattr(state, "classifier", None) is not None,
        "face_detector_loaded": getattr(state, "face_gate", None) is not None,
        "labels_loaded": getattr(state, "labelmap", None) is not None,
        "startup_error": getattr(state, "startup_error", None),
        "uptime_s": _uptime(request),
    }
    if ready and not all(
        payload[k] for k in ("model_loaded", "face_detector_loaded", "labels_loaded")
    ):
        return JSONResponse(status_code=503, content=payload)
    return payload


@router.get("/api/v1/model/info", summary="Metadata model .keras & label")
def model_info(request: Request) -> dict:
    state = request.app.state
    classifier = getattr(state, "classifier", None)
    labelmap = getattr(state, "labelmap", None)
    gate = getattr(state, "face_gate", None)
    return {
        "model": classifier.info.as_dict() if classifier else None,
        "labels": list(labelmap) if labelmap else None,
        "face_detector": getattr(gate, "detector_name", None),
        "ready": bool(classifier and gate and labelmap),
    }


@router.get(
    "/api/v1/requirements",
    summary="Ketentuan unggah yang wajib ditampilkan ke pengguna (§2.2)",
)
def requirements(cfg: Config = Depends(get_config)) -> dict:
    return requirements_notice(
        max_upload_mb=cfg.max_upload_mb, min_side_px=cfg.min_side_px
    )
