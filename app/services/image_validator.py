"""Tier 1 — validasi file unggahan (sebelum Face Gate).

Cek: tidak kosong -> ukuran -> ekstensi -> tipe format nyata -> decode penuh.
`python-magic` dipakai bila tersedia (MIME nyata, bukan hanya ekstensi);
tanpa magic, fallback ke format yang dilaporkan Pillow.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.config import Config
from app.errors import (
    InvalidImageError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from app.models.ml.labels import requirement_message

logger = logging.getLogger("app.image_validator")

# peta MIME -> nama format Pillow
_MIME_TO_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


class ImageValidator:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    # --- publik ------------------------------------------------------------

    def validate(self, raw: bytes, filename: str | None = None) -> None:
        """Lewati semua pemeriksaan Tier 1. Melempar AppError bila gagal."""
        cfg = self._cfg

        if not raw:
            raise InvalidImageError("File kosong.")

        if len(raw) > cfg.max_upload_bytes:
            raise PayloadTooLargeError(
                f"Ukuran file melebihi batas {cfg.max_upload_mb} MB.",
                detail=f"bytes={len(raw)}",
            )

        # ekstensi (cepat, tapi TIDAK dipercaya sendirian)
        if filename:
            ext = Path(filename).suffix.lower()
            if ext and ext not in cfg.allowed_ext:
                raise UnsupportedMediaTypeError(
                    requirement_message("UNSUPPORTED_MEDIA_TYPE"),
                    detail=f"ext={ext}",
                )

        fmt = self._sniff_format(raw)
        if fmt is None:
            raise InvalidImageError("File bukan gambar yang valid atau rusak.")
        if fmt not in cfg.allowed_formats:
            raise UnsupportedMediaTypeError(
                requirement_message("UNSUPPORTED_MEDIA_TYPE"),
                detail=f"format={fmt}",
            )

        # decode penuh — menangkap file terpotong / decompression bomb
        try:
            img = Image.open(BytesIO(raw))
            img.load()
        except Image.DecompressionBombError as exc:
            raise InvalidImageError(
                "Dimensi gambar melebihi batas keamanan.", detail=str(exc)
            ) from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError(
                "File bukan gambar yang valid atau rusak.", detail=str(exc)
            ) from exc

    # --- internal ----------------------------------------------------------

    @staticmethod
    def _sniff_format(raw: bytes) -> str | None:
        """Nama format Pillow ('JPEG'/'PNG'/'WEBP'/...) atau None bila tak dikenal."""
        # 1) python-magic bila tersedia — memeriksa byte header sungguhan
        try:
            import magic  # type: ignore

            mime = magic.from_buffer(raw, mime=True)
            if mime:
                return _MIME_TO_FORMAT.get(mime, mime)
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 - magic gagal -> fallback Pillow
            logger.debug("python-magic gagal: %s", exc)

        # 2) fallback: Pillow
        try:
            with Image.open(BytesIO(raw)) as img:
                return img.format
        except (UnidentifiedImageError, OSError, ValueError):
            return None
