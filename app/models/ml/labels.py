"""Pemetaan indeks output softmax → nama platform + kamus pesan ketentuan.

`class_indices.json` adalah SATU-SATUNYA sumber urutan label. File ini dibuat
saat training, misalnya:

    import json
    json.dump(train_generator.class_indices, open("class_indices.json", "w"))

Jangan pernah mengurutkan label secara alfabetis (arsitektur.md §2.3 A3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.errors import LabelsNotReadyError

# --- Kamus pesan ketentuan §2.2 (dipakai saat menolak unggahan) --------------

REQUIREMENT_MESSAGES: dict[str, str] = {
    "NO_FACE_DETECTED": (
        "Tidak ditemukan wajah manusia pada gambar. Sistem hanya menerima foto "
        "potret satu orang dari wajah sampai pundak."
    ),
    "MULTIPLE_FACES": (
        "Terdeteksi lebih dari satu orang. Unggah foto yang hanya berisi satu orang."
    ),
    "FACE_TOO_SMALL": (
        "Wajah terlalu kecil dalam gambar. Gunakan potret wajah–pundak, bukan foto "
        "seluruh badan atau kerumunan."
    ),
    "FACE_TOO_LARGE": (
        "Wajah terpotong/terlalu dekat. Posisikan gambar agar wajah dan bahu terlihat "
        "seluruhnya."
    ),
    "INVALID_FRAMING": (
        "Posisi wajah di luar bingkai yang sesuai. Pusatkan wajah dalam bingkai potret."
    ),
    "IMAGE_TOO_LOW_RES": (
        "Resolusi terlalu rendah. Gunakan gambar dengan sisi terpendek minimal 224 px."
    ),
    "INVALID_IMAGE": "File bukan gambar yang valid atau rusak. Gunakan JPG, PNG, atau WEBP.",
    "PAYLOAD_TOO_LARGE": "Ukuran file melebihi batas 10 MB.",
    "UNSUPPORTED_MEDIA_TYPE": "Format file tidak didukung. Gunakan JPG, PNG, atau WEBP.",
    "MODEL_NOT_READY": "Model klasifikasi belum siap. Coba lagi beberapa saat.",
    "FACE_CHECK_UNAVAILABLE": (
        "Layanan pemeriksaan gambar sedang tidak tersedia. Permintaan tidak "
        "diproses demi menjaga ketentuan unggahan."
    ),
    "LABELS_NOT_READY": "Konfigurasi label model belum lengkap. Hubungi pengelola.",
    "INFERENCE_FAILED": "Terjadi kesalahan saat mengklasifikasi gambar. Coba lagi.",
}

REQUIREMENT_REFS: dict[str, str] = {
    "NO_FACE_DETECTED": "§2.2 — ❌ Gambar tanpa wajah",
    "MULTIPLE_FACES": "§2.2 — ❌ Lebih dari satu orang dalam satu gambar",
    "FACE_TOO_SMALL": "§2.2 — ❌ Seluruh badan / kerumunan",
    "FACE_TOO_LARGE": "§2.2 — ❌ Close-up terpotong",
    "INVALID_FRAMING": "§2.2 — ✅ Cakupan wajah–pundak",
    "IMAGE_TOO_LOW_RES": "§2.2 — ❌ Resolusi rendah",
}


def requirement_message(code: str) -> str:
    """Pesan ramah pengguna untuk sebuah kode error (fallback ke pesan generik)."""
    return REQUIREMENT_MESSAGES.get(code, REQUIREMENT_MESSAGES["INFERENCE_FAILED"])


def requirement_ref(code: str) -> str | None:
    """Tautan ke butir ketentuan §2.2 (None bila kode bukan pelanggaran ketentuan)."""
    return REQUIREMENT_REFS.get(code)


@dataclass(frozen=True)
class LabelMap:
    """Urutan label hasil training, terbaca dari `class_indices.json`."""

    ordered: tuple[str, ...]

    # --- konstruktor -------------------------------------------------------

    @classmethod
    def from_mapping(cls, mapping: dict[str, int]) -> "LabelMap":
        if not mapping:
            raise LabelsNotReadyError("class_indices.json kosong.")
        if not all(isinstance(v, int) for v in mapping.values()):
            raise LabelsNotReadyError("Nilai class_indices.json harus berupa integer.")
        count = len(mapping)
        if sorted(mapping.values()) != list(range(count)):
            raise LabelsNotReadyError(
                "Index label harus berurutan 0..N-1 tanpa duplikat.",
                detail=f"mapping={mapping!r}",
            )
        ordered = tuple(name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1]))
        return cls(ordered=ordered)

    @classmethod
    def load(cls, path: Path | str) -> "LabelMap":
        path = Path(path)
        if not path.exists():
            raise LabelsNotReadyError(
                f"File label tidak ditemukan: {path.name}. "
                "Buat dari skrip training, mis. json.dump(train_generator.class_indices, ...)",
                detail=f"path={path}",
            )
        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabelsNotReadyError(
                "File label tidak bisa dibaca (JSON tidak valid).", detail=str(exc)
            ) from exc

        if isinstance(data, list):  # bentuk alternatif: ["dalle", "gemini", ...]
            data = {name: idx for idx, name in enumerate(data)}
        if not isinstance(data, dict):
            raise LabelsNotReadyError("Format file label tidak dikenali.")
        return cls.from_mapping({str(k): int(v) for k, v in data.items()})

    # --- akses -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ordered)

    def __iter__(self):
        return iter(self.ordered)

    @property
    def count(self) -> int:
        return len(self.ordered)

    def name(self, index: int) -> str:
        if 0 <= index < len(self.ordered):
            return self.ordered[index]
        raise IndexError(f"Index label di luar jangkauan: {index}")

    def index(self, name: str) -> int:
        return self.ordered.index(name)
