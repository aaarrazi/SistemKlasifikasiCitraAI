"""DTO request/response (Pydantic) — kontrak antara Controller dan client."""

from app.schemas.error_schema import ErrorBody, ErrorResponse
from app.schemas.predict_schema import (
    FaceCheckInfo,
    ModelMeta,
    PredictionInfo,
    PredictResponse,
    ScoredLabel,
)

__all__ = [
    "ErrorBody",
    "ErrorResponse",
    "FaceCheckInfo",
    "ModelMeta",
    "PredictionInfo",
    "PredictResponse",
    "ScoredLabel",
]
