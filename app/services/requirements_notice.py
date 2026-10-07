"""Teks ketentuan unggahan (§2.2) — sumber tunggal copy yang tampil ke pengguna.

Dipakai oleh:
- View (diambil lewat GET /api/v1/requirements sebelum upload)
- deskripsi endpoint pada dokumentasi OpenAPI
"""

from __future__ import annotations

TITLE = "Ketentuan Unggah Gambar"

INTRO = (
    "Sistem ini mengklasifikasi citra hasil generate AI pada 4 platform "
    "(Gemini, DALL·E, Midjourney, Stable Diffusion). Model hanya dilatih pada "
    "citra wajah manusia dengan posisi wajah sampai pundak, sehingga gambar yang "
    "Anda unggah wajib memenuhi ketentuan berikut."
)

ALLOWED = [
    "Satu orang saja dalam gambar",
    "Wajah terlihat jelas, cakupan wajah sampai pundak (head & shoulders)",
    "Wajah tidak tertutup masker/tangan/rambut dan tidak terpotong tepi frame",
    "Pencahayaan cukup dan gambar tidak blur",
]

PROHIBITED = [
    "Lebih dari satu orang dalam satu gambar",
    "Seluruh badan, atau close-up hanya sebagian wajah (mis. hanya mata)",
    "Pemandangan, benda, hewan, logo, teks, meme, atau gambar tanpa wajah",
    "Gambar gelap, buram, atau resolusi rendah",
]

TECHNICAL = {
    "formats": ["JPG", "PNG", "WEBP"],
    "max_size_mb": 10,
    "min_side_px": 224,
}

WARNING = (
    "Hasil klasifikasi hanya berlaku untuk gambar dalam batasan di atas. "
    "Gambar di luar batasan akan ditolak dan tidak akan diklasifikasi."
)

AGREE_TEXT = "Saya memahami dan menyetujui ketentuan di atas."


def requirements_notice(max_upload_mb: int = 10, min_side_px: int = 224) -> dict:
    """Payload ketentuan untuk View & OpenAPI (nilai teknis dari konfigurasi)."""
    return {
        "title": TITLE,
        "intro": INTRO,
        "allowed": list(ALLOWED),
        "prohibited": list(PROHIBITED),
        "technical": {
            "formats": list(TECHNICAL["formats"]),
            "max_size_mb": max_upload_mb,
            "min_side_px": min_side_px,
        },
        "warning": WARNING,
        "agree_text": AGREE_TEXT,
    }
