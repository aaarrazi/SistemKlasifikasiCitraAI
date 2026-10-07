"""Wrapper model `.keras` — memuat sekali (singleton) dan menjalankan inferensi.

Catatan (arsitektur.md §7.1 / A11):
- `load_model(..., compile=False)` — optimizer & loss tidak dipakai saat inferensi.
- `model(x, training=False)` — memastikan Dropout(0.4) NONAKTIF saat prediksi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Config
from app.errors import InferenceFailedError, ModelNotReadyError
from app.models.ml.labels import LabelMap

logger = logging.getLogger("app.classifier")


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    keras_version: str | None
    path: str
    input_shape: tuple
    output_units: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "keras_version": self.keras_version,
            "path": Path(self.path).name,
            "input_shape": list(self.input_shape),
            "output_units": self.output_units,
        }


def _import_keras() -> Any:
    """Impor Keras (preferensi: paket `keras` standalone, fallback `tf.keras`)."""
    try:
        import keras  # keras 3.13.2 — sesuai metadata file model (A9)

        return keras
    except ImportError:
        pass
    try:
        from tensorflow import keras as tf_keras  # type: ignore

        return tf_keras
    except ImportError as exc:
        raise ModelNotReadyError(
            "TensorFlow/Keras belum terpasang. Jalankan Fase 0: buat venv Python "
            "3.11/3.12 lalu `pip install -r requirements.txt`.",
            detail=str(exc),
        ) from exc


class Classifier:
    """Model CNN untuk klasifikasi platform generator (EfficientNetB0)."""

    def __init__(
        self,
        model: Any,
        labelmap: LabelMap | None,
        cfg: Config,
        keras_version: str | None,
    ) -> None:
        self._model = model
        self._labelmap = labelmap
        self._cfg = cfg
        self._keras_version = keras_version
        out_units = getattr(model, "output_shape", None)
        self._info = ModelInfo(
            name=cfg.model_name,
            version=cfg.model_version,
            keras_version=keras_version,
            path=str(cfg.model_path),
            input_shape=tuple(getattr(model, "input_shape", cfg.input_shape)),
            output_units=int(out_units[-1]) if out_units else 0,
        )

    # --- pemuat ------------------------------------------------------------

    @classmethod
    def load(cls, cfg: Config, labelmap: LabelMap | None) -> "Classifier":
        path = Path(cfg.model_path)
        if not path.exists():
            raise ModelNotReadyError(
                f"File model tidak ditemukan: {path.name}",
                detail=f"MODEL_PATH={path}",
            )

        keras = _import_keras()
        try:
            model = keras.models.load_model(path, compile=False)
        except Exception as exc:  # noqa: BLE001 - pesan Keras bervariasi
            raise ModelNotReadyError(
                "File model gagal dimuat. Pastikan versi Keras sesuai metadata "
                "(keras 3.13.2) — lihat arsitektur.md A9.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        cls._validate(model, labelmap, cfg)
        version = getattr(keras, "__version__", None)
        logger.info("Model dimuat: %s (keras %s)", path.name, version)
        return cls(model=model, labelmap=labelmap, cfg=cfg, keras_version=version)

    @staticmethod
    def _validate(model: Any, labelmap: LabelMap | None, cfg: Config) -> None:
        in_shape = tuple(getattr(model, "input_shape", ()) or ())
        if in_shape and len(in_shape) == 4:
            expected = cfg.input_shape
            # bandingkan dimensi non-batch (A1)
            if tuple(in_shape[1:]) != tuple(expected[1:]):
                raise ModelNotReadyError(
                    "Bentuk input model tidak sesuai konfigurasi.",
                    detail=f"model={in_shape} config={expected}",
                )

        if labelmap is None:  # verifikasi tanpa label (Fase 0 sebagian)
            return

        out_shape = tuple(getattr(model, "output_shape", ()) or ())
        units = int(out_shape[-1]) if out_shape else 0
        if units and units != len(labelmap):
            raise ModelNotReadyError(
                f"Jumlah output model ({units}) tidak sama dengan jumlah label "
                f"({len(labelmap)}). Periksa class_indices.json.",
                detail=f"output_shape={out_shape}",
            )

    # --- inferensi ---------------------------------------------------------

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Jalankan inferensi. Selalu mode non-training (A11)."""
        try:
            output = self._model(batch, training=False)
            if hasattr(output, "numpy"):
                output = output.numpy()
            return np.asarray(output, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inferensi gagal")
            raise InferenceFailedError(
                "Terjadi kesalahan saat mengklasifikasi gambar.", detail=str(exc)
            ) from exc

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def labelmap(self) -> LabelMap | None:
        return self._labelmap
