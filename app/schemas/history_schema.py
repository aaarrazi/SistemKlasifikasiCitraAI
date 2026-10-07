"""DTO untuk riwayat prediksi & penolakan."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    id: int
    request_id: str
    created_at: str
    filename: str | None = None
    label: str | None = Field(default=None, description="NULL bila ditolak")
    confidence: float | None = None
    status: str | None = None
    face_status: str | None = None
    reject_code: str | None = Field(default=None, description="Kode penolakan §2.2")
    processing_ms: int | None = None

    model_config = {"extra": "allow"}


class HistoryPage(BaseModel):
    items: list[HistoryItem]
    page: int
    size: int
    total: int


class HistoryDeleted(BaseModel):
    deleted: bool
    id: int
