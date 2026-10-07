"""Controller prediksi — menerima unggahan, memanggil Service, menyusun respons.

Aturan: handler setipis mungkin, TANPA logika ML/SQL/manipulasi piksel.
"""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.config import Config, get_config
from app.deps import get_classification_service
from app.errors import AppError
from app.schemas.predict_schema import (
    FaceCheckInfo,
    ModelMeta,
    PredictionInfo,
    PredictResponse,
    ScoredLabel,
)
from app.services.classification_service import ClassificationResult, ClassificationService

router = APIRouter(prefix="/api/v1", tags=["predict"])
logger = logging.getLogger("app.predict")


def _to_response(result: ClassificationResult, request_id: str) -> PredictResponse:
    """Susun DTO dari hasil use-case (pemetaan murni, tanpa logika)."""
    return PredictResponse(
        face_check=FaceCheckInfo(**result.face_check.as_dict()),
        prediction=PredictionInfo(
            label=result.prediction.label,
            confidence=result.prediction.confidence,
            status=result.prediction.status,
            top_k=[ScoredLabel(**s.__dict__) for s in result.prediction.top_k],
        ),
        model=ModelMeta(**result.model_info),
        processing_time_ms=result.processing_time_ms,
        request_id=request_id,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Klasifikasi 1 gambar hasil generate AI",
    description=(
        "Menerima gambar JPG/PNG/WEBP (maks. 10 MB) yang berisi SATU wajah manusia "
        "dengan cakupan wajah sampai pundak. Gambar di luar ketentuan ditolak sebelum "
        "inferensi dengan kode error NO_FACE_DETECTED, MULTIPLE_FACES, dst."
    ),
)
async def classify(
    request: Request,
    file: UploadFile = File(..., description="File gambar (JPG/PNG/WEBP, maks 10 MB)"),
    top_k: int = Form(4, ge=1, le=4, description="Jumlah label teratas yang dikembalikan"),
    agree_terms: bool = Form(
        False,
        description=(
            "Penanda bahwa klien sudah menampilkan ketentuan §2.2 ke pengguna. "
            "Bukan pengganti Face Gate — pemeriksaan tetap berjalan di server."
        ),
    ),
    service: ClassificationService = Depends(get_classification_service),
) -> PredictResponse:
    request_id = getattr(request.state, "request_id", uuid4().hex)
    if not agree_terms:
        # dicatat untuk audit; tetap diproses karena gate server-side yang menentukan
        logger.debug("Request tanpa agree_terms=true (%s)", file.filename)

    result = service.classify(
        await file.read(),
        top_k=top_k,
        filename=file.filename,
        request_id=request_id,
    )
    return _to_response(result, request_id)


@router.post(
    "/predict/batch",
    summary="Klasifikasi hingga N gambar sekaligus",
    description=(
        "Setiap file dinilai terpisah: satu file ditolak tidak membatalkan file lain. "
        "Item gagal membalas objek `error` (kode + pesan + requirement_ref)."
    ),
)
async def classify_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Daftar file gambar (maks 10)"),
    top_k: int = Form(4, ge=1, le=4),
    service: ClassificationService = Depends(get_classification_service),
    cfg: Config = Depends(get_config),
) -> dict:
    if len(files) > cfg.batch_max_files:
        raise AppError(
            f"Maksimal {cfg.batch_max_files} file per batch.",
            code="BATCH_TOO_LARGE",
            status_code=400,
            detail=f"files={len(files)}",
        )

    started = perf_counter()
    results: list[dict] = []
    for upload in files:
        item_id = uuid4().hex
        try:
            result = service.classify(
                await upload.read(),
                top_k=top_k,
                filename=upload.filename,
                request_id=item_id,
            )
            payload = _to_response(result, item_id).model_dump()
            results.append({"filename": upload.filename, "success": True, **payload})
        except AppError as exc:
            results.append(
                {
                    "filename": upload.filename,
                    "success": False,
                    "error": exc.to_payload(),
                    "request_id": item_id,
                }
            )

    return {
        "success": True,
        "count": len(results),
        "results": results,
        "processing_time_ms": int((perf_counter() - started) * 1000),
    }
