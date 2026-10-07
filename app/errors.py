"""Lapisan error terpusat.

Setiap error membawa `code` (kode API), `message` (pesan ramah pengguna) dan
opsional `requirement_ref` (tautan ke butir ketentuan §2.2) sehingga Controller
cukup melempar exception dan handler di `main.py` yang menerjemahkannya ke JSON.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class untuk semua error yang bisa dikomunikasikan ke client."""

    code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        requirement_ref: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        self.requirement_ref = requirement_ref
        self.detail = detail  # hanya untuk log server, TIDAK dikirim ke client

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.requirement_ref:
            payload["requirement_ref"] = self.requirement_ref
        return payload


# --- Tier 1: file -----------------------------------------------------------


class InvalidImageError(AppError):
    code = "INVALID_IMAGE"
    status_code = 400


class PayloadTooLargeError(AppError):
    code = "PAYLOAD_TOO_LARGE"
    status_code = 413


class UnsupportedMediaTypeError(AppError):
    code = "UNSUPPORTED_MEDIA_TYPE"
    status_code = 415


# --- Tier 2: Face Gate (ketentuan pengguna §2.2) ----------------------------


class FaceGateRejected(AppError):
    """Gambar ditolak Face Gate. `code` diisi dari hasil pemeriksaan."""

    code = "FACE_GATE_REJECTED"
    status_code = 422


class FaceCheckUnavailableError(AppError):
    """Detektor wajah tidak bisa dijalankan — fail-closed, prediksi DITOLAK."""

    code = "FACE_CHECK_UNAVAILABLE"
    status_code = 503


# --- Model ML ---------------------------------------------------------------


class ModelNotReadyError(AppError):
    code = "MODEL_NOT_READY"
    status_code = 503


class LabelsNotReadyError(AppError):
    code = "LABELS_NOT_READY"
    status_code = 503


class InferenceFailedError(AppError):
    code = "INFERENCE_FAILED"
    status_code = 500
