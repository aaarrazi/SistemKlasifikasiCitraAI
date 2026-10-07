"""Preprocessing — SATU-SATUNYA tempat transformasi piksel (arsitektur.md §7.3).

Kontrak input model (A1 + A2, terverifikasi dari config.json):
    input (1, 224, 224, 3) float32 rentang [0, 255]

Model EfficientNetB0 di dalam file .keras SUDAH melakukan:
    Rescaling(1/255) -> Normalization(axis=3) -> Rescaling([2.09, 2.11, 2.11])

Karena itu di sini dilarang membagi pixel dengan 255 (normalisasi dobel
menurunkan akurasi diam-diam — risiko R1).
"""

from __future__ import annotations

import logging
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import Config
from app.errors import InvalidImageError

logger = logging.getLogger("app.preprocessor")


def open_rgb(raw: bytes) -> Image.Image:
    """Decode bytes → PIL Image RGB. Melempar InvalidImageError bila korup."""
    try:
        img = Image.open(BytesIO(raw))
        img.load()  # paksa decode penuh → menangkap file terpotong/rusak
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "File bukan gambar yang valid atau rusak.",
            detail=str(exc),
        ) from exc
    return img.convert("RGB")


def preprocess(raw: bytes, cfg: Config) -> np.ndarray:
    """Bytes gambar → array float32 (1, H, W, 3) rentang [0, 255].

    Tidak ada pembagian 255 dan tidak ada augmentasi.

    Interpolasi = NEAREST **bukan pilihan estetis**: `flow_from_directory` (yang
    dipakai training) memakai interpolasi default Keras = `nearest`. Menguji
    160 gambar dataset ke model asli (`scripts/derive_class_indices.py`):

        nearest  96,9 %   ← sesuai training, DIPAKAI
        lanczos  85,6 %
        bicubic  84,4 %
        bilinear 81,2 %   ← versi lama, kalah 15,7 poin (R1: akurasi turun diam-diam)
    """
    img = open_rgb(raw)
    img = img.resize((cfg.img_width, cfg.img_height), Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.float32)  # [0, 255] — JANGAN /255 (A2)
    arr = np.expand_dims(arr, axis=0)
    return arr


def describe_range(arr: np.ndarray) -> str:
    """Ringkasan rentang nilai untuk log/debug (bukan untuk keputusan prediksi)."""
    return (
        f"shape={arr.shape} dtype={arr.dtype} "
        f"min={float(arr.min()):.1f} max={float(arr.max()):.1f}"
    )
