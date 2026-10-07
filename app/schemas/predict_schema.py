"""Skema respons prediksi (lihat arsitektur.md §6.2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScoredLabel(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionInfo(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: str = Field(description="high | medium | uncertain")
    top_k: list[ScoredLabel]


class FaceCheckInfo(BaseModel):
    status: str = Field(description="passed | rejected")
    faces_detected: int
    face_coverage: float | None = None
    detector: str
    code: str | None = None


class ModelMeta(BaseModel):
    name: str
    version: str
    keras_version: str | None = None
    path: str | None = None
    input_shape: list | None = None
    output_units: int | None = None


class PredictResponse(BaseModel):
    success: bool = True
    face_check: FaceCheckInfo
    prediction: PredictionInfo
    model: ModelMeta
    processing_time_ms: int
    request_id: str
