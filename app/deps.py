"""Penyedia dependensi FastAPI (dependency injection).

Controller hanya menerima object lewat `Depends(...)` — tidak pernah meng-import
model ML secara langsung. Saat testing, cukup daftarkan `app.dependency_overrides`
untuk `get_classifier` / `get_face_gate` (lihat tests/conftest.py).
"""

from __future__ import annotations

from fastapi import Depends, Request

from app.config import Config, get_config
from app.errors import FaceCheckUnavailableError, LabelsNotReadyError, ModelNotReadyError
from app.models.db import HistoryUnavailableError
from app.models.ml.classifier import Classifier
from app.models.ml.face_gate import FaceGate
from app.models.ml.labels import LabelMap
from app.services.classification_service import ClassificationService
from app.services.history_service import HistoryService
from app.services.image_validator import ImageValidator


def get_app_config() -> Config:
    return get_config()


def get_labelmap(request: Request) -> LabelMap:
    labelmap: LabelMap | None = getattr(request.app.state, "labelmap", None)
    if labelmap is None:
        reason = getattr(request.app.state, "startup_error", None) or "belum dimuat"
        raise LabelsNotReadyError(
            "Konfigurasi label model belum siap.", detail=str(reason)
        )
    return labelmap


def get_classifier(request: Request) -> Classifier:
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        reason = getattr(request.app.state, "startup_error", None) or "belum dimuat"
        raise ModelNotReadyError(
            "Model klasifikasi belum siap. Coba lagi beberapa saat.", detail=str(reason)
        )
    return classifier


def get_face_gate(request: Request) -> FaceGate:
    gate = getattr(request.app.state, "face_gate", None)
    if gate is None:
        reason = getattr(request.app.state, "startup_error", None) or "belum dimuat"
        raise FaceCheckUnavailableError(
            "Layanan pemeriksaan gambar tidak tersedia; permintaan ditolak "
            "agar ketentuan tetap terpenuhi.",
            detail=str(reason),
        )
    return gate


def get_history_service(request: Request) -> HistoryService:
    """Versi ketat — dipakai endpoint riwayat (bila DB mati → 503)."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HistoryUnavailableError(
            "Riwayat tidak tersedia. Periksa konfigurasi DATABASE_PATH / HISTORY_ENABLED."
        )
    return HistoryService(db)


def get_optional_history(request: Request) -> HistoryService | None:
    """Versi longgar — riwayat tidak boleh menggagalkan prediksi."""
    db = getattr(request.app.state, "db", None)
    return HistoryService(db) if db is not None else None


def get_classification_service(
    cfg: Config = Depends(get_app_config),
    labelmap: LabelMap = Depends(get_labelmap),
    gate: FaceGate = Depends(get_face_gate),
    classifier: Classifier = Depends(get_classifier),
    history: HistoryService | None = Depends(get_optional_history),
) -> ClassificationService:
    return ClassificationService(
        cfg=cfg,
        validator=ImageValidator(cfg),
        gate=gate,
        classifier=classifier,
        labelmap=labelmap,
        history=history,
    )
