"""Use-case riwayat — menyimpan hasil prediksi DAN alasan penolakan.

Bersifat **fail-soft**: kegagalan menulis riwayat tidak boleh menggagalkan
prediksi yang sebenarnya sudah sukses (hanya dicatat sebagai warning).
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.db import Database, HistoryUnavailableError

logger = logging.getLogger("app.history")


class HistoryService:
    def __init__(self, db: Database) -> None:
        self._db = db

    # --- tulis -------------------------------------------------------------

    def record_prediction(
        self,
        *,
        request_id: str,
        filename: str | None,
        label: str,
        confidence: float,
        status: str,
        processing_ms: int | None = None,
    ) -> None:
        """Simpan prediksi sukses."""
        try:
            self._db.record(
                request_id=request_id,
                filename=filename,
                label=label,
                confidence=confidence,
                status=status,
                face_status="passed",
                processing_ms=processing_ms,
            )
        except HistoryUnavailableError as exc:
            logger.warning("Riwayat prediksi tidak tersimpan: %s", exc.detail)

    def record_rejection(
        self,
        *,
        request_id: str,
        filename: str | None,
        code: str,
        processing_ms: int | None = None,
        face_status: str = "rejected",
    ) -> None:
        """Simpan penolakan (Tier 1 maupun Face Gate) untuk audit §2.2."""
        try:
            self._db.record(
                request_id=request_id,
                filename=filename,
                reject_code=code,
                face_status=face_status,
                processing_ms=processing_ms,
            )
        except HistoryUnavailableError as exc:
            logger.warning("Riwayat penolakan tidak tersimpan: %s", exc.detail)

    # --- baca --------------------------------------------------------------

    def list(self, page: int = 1, size: int = 20) -> tuple[list[dict[str, Any]], int]:
        return self._db.list(page, size)

    def get(self, row_id: int) -> dict[str, Any] | None:
        return self._db.get(row_id)

    def delete(self, row_id: int) -> bool:
        return self._db.delete(row_id)
