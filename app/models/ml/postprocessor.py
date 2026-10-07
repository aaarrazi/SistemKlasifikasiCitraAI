"""Post-processing — softmax → label + confidence + status ambang (§7.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.ml.labels import LabelMap

STATUS_HIGH = "high"
STATUS_MEDIUM = "medium"
STATUS_UNCERTAIN = "uncertain"

UNKNOWN_LABEL = "unknown"


@dataclass(frozen=True)
class ScoredLabel:
    label: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    status: str
    top_k: tuple[ScoredLabel, ...]

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "status": self.status,
            "top_k": [{"label": s.label, "confidence": s.confidence} for s in self.top_k],
        }


def summarize(
    probs: np.ndarray,
    labelmap: LabelMap,
    *,
    top_k: int = 4,
    high: float = 0.70,
    medium: float = 0.40,
) -> PredictionResult:
    """Terjemahkan vektor probabilitas → PredictionResult.

    - max < medium  -> status `uncertain`, label `unknown` (model dipaksa tidak
      mengklaim bila keyakinannya rendah).
    - medium..high  -> status `medium`, label terbaik + peringatan.
    - >= high       -> status `high`.
    """
    values = np.asarray(probs, dtype=np.float64).reshape(-1)
    if values.size != len(labelmap):
        raise ValueError(
            f"Jumlah output model ({values.size}) tidak sama dengan jumlah label ({len(labelmap)})."
        )

    order = np.argsort(values)[::-1]
    n = max(1, min(int(top_k), values.size))
    top = tuple(
        ScoredLabel(label=labelmap.name(int(i)), confidence=float(values[int(i)]))
        for i in order[:n]
    )

    best = top[0]
    if best.confidence < medium:
        return PredictionResult(
            label=UNKNOWN_LABEL,
            confidence=best.confidence,
            status=STATUS_UNCERTAIN,
            top_k=top,
        )

    status = STATUS_HIGH if best.confidence >= high else STATUS_MEDIUM
    return PredictionResult(label=best.label, confidence=best.confidence, status=status, top_k=top)
