"""Use-case utama: klasifikasi 1 gambar (§3.1 — dipanggil Controller).

Alur: Tier 1 (file) → Tier 2 (Face Gate) → Tier 3 (preprocess) → inferensi → ambang.
Riwayat disimpan untuk prediksi sukses maupun penolakan (audit §2.2), bersifat
fail-soft: kegagalan menulis riwayat tidak membatalkan hasil prediksi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from app.config import Config
from app.errors import AppError, FaceGateRejected
from app.models.ml.classifier import Classifier
from app.models.ml.face_gate import FaceGate, FaceGateResult
from app.models.ml.labels import LabelMap
from app.models.ml.postprocessor import PredictionResult, summarize
from app.models.ml.preprocessor import preprocess
from app.services.image_validator import ImageValidator

if TYPE_CHECKING:
    from app.services.history_service import HistoryService

logger = logging.getLogger("app.classification")

# Status error yang layak dicatat sebagai penolakan konten (Tier 1/2),
# bukan gangguan infrastruktur (503/500) yang tidak menyangkut gambar.
AUDITABLE_STATUS = {400, 413, 415, 422}


@dataclass(frozen=True)
class ClassificationResult:
    prediction: PredictionResult
    face_check: FaceGateResult
    model_info: dict
    processing_time_ms: int


class ClassificationService:
    """Orkestrasi: Tier 1 -> Tier 2 -> Tier 3 -> inferensi -> ambang."""

    def __init__(
        self,
        cfg: Config,
        validator: ImageValidator,
        gate: FaceGate,
        classifier: Classifier,
        labelmap: LabelMap,
        history: "HistoryService | None" = None,
    ) -> None:
        self._cfg = cfg
        self._validator = validator
        self._gate = gate
        self._classifier = classifier
        self._labelmap = labelmap
        self._history = history

    def classify(
        self,
        raw: bytes,
        *,
        top_k: int | None = None,
        filename: str | None = None,
        request_id: str = "-",
    ) -> ClassificationResult:
        started = perf_counter()

        try:
            prediction, gate_result = self._run(raw, top_k=top_k, filename=filename)
        except AppError as exc:
            elapsed = int((perf_counter() - started) * 1000)
            logger.info(
                "Tolak %s -> %s (%dms)", filename or "tanpa-nama", exc.code, elapsed
            )
            if self._history and exc.status_code in AUDITABLE_STATUS:
                self._history.record_rejection(
                    request_id=request_id,
                    filename=filename,
                    code=exc.code,
                    processing_ms=elapsed,
                )
            raise

        elapsed = int((perf_counter() - started) * 1000)
        logger.info(
            "Prediksi %s -> %s (%.3f, %s) %dms",
            filename or "tanpa-nama",
            prediction.label,
            prediction.confidence,
            prediction.status,
            elapsed,
        )
        if self._history:
            self._history.record_prediction(
                request_id=request_id,
                filename=filename,
                label=prediction.label,
                confidence=prediction.confidence,
                status=prediction.status,
                processing_ms=elapsed,
            )
        return ClassificationResult(
            prediction=prediction,
            face_check=gate_result,
            model_info=self._classifier.info.as_dict(),
            processing_time_ms=elapsed,
        )

    # --- pipeline ----------------------------------------------------------

    def _run(
        self, raw: bytes, *, top_k: int | None, filename: str | None
    ) -> tuple[PredictionResult, FaceGateResult]:
        # Tier 1 — file
        self._validator.validate(raw, filename)

        # Tier 2 — ketentuan pengguna (Face Gate); gagal = TIDAK diklasifikasi
        gate_result = self._gate.check(raw)
        if not gate_result.passed:
            raise FaceGateRejected(
                gate_result.message or "Gambar tidak memenuhi ketentuan unggahan.",
                code=gate_result.code or "FACE_GATE_REJECTED",
                requirement_ref=gate_result.requirement_ref,
                detail=f"faces={gate_result.faces_detected} coverage={gate_result.face_coverage}",
            )

        # Tier 3 — preprocessing (kontrak [0, 255], tanpa /255)
        batch: Any = preprocess(raw, self._cfg)

        # Inferensi (mode non-training) + ambang
        probs = self._classifier.predict(batch)
        prediction = summarize(
            probs.reshape(-1),
            self._labelmap,
            top_k=top_k or self._cfg.default_top_k,
            high=self._cfg.confidence_high,
            medium=self._cfg.confidence_medium,
        )
        return prediction, gate_result
