"""Fase 0 — unduh model detektor wajah BlazeFace untuk MediaPipe Tasks API.

Jalankan dari root proyek (sekali saja):

    python scripts/download_face_model.py

File disimpan ke `models/blaze_face_short_range.tflite` (± 230 KB) — dibutuhkan
`FaceGate` (arsitektur.md §7.2). mediapipe >= 1.0 sudah TIDAK menyediakan API lama
`mp.solutions`, sehingga detektor membutuhkan file model ini secara eksplisit.

Sumber: storage.googleapis.com/mediapipe-models (dipakai resmi oleh contoh MediaPipe).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
DEST = ROOT / "models" / "blaze_face_short_range.tflite"
MIN_BYTES = 100_000  # file valid jauh lebih besar dari ini


def main() -> int:
    if DEST.exists() and DEST.stat().st_size >= MIN_BYTES:
        print(f"[OK] Sudah ada: {DEST} ({DEST.stat().st_size:,} byte)")
        return 0

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh model detektor wajah...\n  {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp, DEST.open("wb") as out:
            while chunk := resp.read(1 << 16):
                out.write(chunk)
    except Exception as exc:  # noqa: BLE001
        if DEST.exists():
            DEST.unlink()  # buang potongan yang tidak utuh
        print(f"[!] Gagal mengunduh: {exc}")
        print("    Coba lagi, atau unduh manual URL di atas lalu simpan sebagai:")
        print(f"    {DEST}")
        return 1

    size = DEST.stat().st_size
    if size < MIN_BYTES:
        DEST.unlink()
        print(f"[!] Unduhan rusak (hanya {size} byte).")
        return 1

    print(f"[OK] Tersimpan: {DEST} ({size:,} byte)")
    print("    Jalankan berikutnya: python scripts/verify_model.py  lalu  pytest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
