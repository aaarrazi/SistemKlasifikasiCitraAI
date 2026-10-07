"""Uji detektor wajah SUNGGUHAN (MediaPipe Tasks API + BlazeFace).

Auto-skip bila mediapipe/model belum ada — jalankan di `.venv` (Fase 0).
Memakai foto contoh bawaan matplotlib (potret nyata) untuk memastikan detektor
benar-benar menemukan wajah, bukan sekadar bisa dibuat objeknya.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("mediapipe", reason="mediapipe belum terpasang — jalankan di .venv (Fase 0)")

from app.config import get_config  # noqa: E402
from app.models.ml.face_gate import FaceGate, MediaPipeDetector  # noqa: E402

CFG = get_config()


def _sample_photo() -> Path:
    """Foto contoh bawaan matplotlib (potret Grace Hopper).

    `get_sample_data` bisa mengembalikan path ATAU objek file terbuka
    (bergantung versi matplotlib), jadi keduanya ditangani.
    """
    try:
        import matplotlib.cbook as cbook

        result = cbook.get_sample_data("grace_hopper.jpg")
        path = Path(getattr(result, "name", result))
        if hasattr(result, "close"):
            result.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"foto contoh tidak tersedia: {exc}")
    if not path.exists():
        pytest.skip(f"foto contoh tidak ditemukan: {path}")
    return path


def _detector() -> MediaPipeDetector:
    if not CFG.face_model_path_resolved.exists():
        pytest.skip("model detektor belum diunduh — python scripts/download_face_model.py")
    return MediaPipeDetector(model_path=CFG.face_model_path_resolved, min_confidence=0.5)


def test_detektor_menemukan_tepat_satu_wajah():
    det = _detector()
    photo = _sample_photo()
    faces = det.detect(Image.open(photo))

    assert len(faces) == 1, f"expect 1 wajah, dapat {len(faces)}"
    face = faces[0]
    assert face.confidence >= 0.5
    # kotak di dalam bingkai gambar
    img = Image.open(photo)
    assert face.left >= 0 and face.top >= 0
    assert face.left + face.width <= img.width + 1
    # cakupan (F5) wajar untuk potret wajah–pundak
    coverage = face.height / img.height
    assert 0.15 <= coverage <= 0.75, f"coverage={coverage:.3f}"


def test_face_gate_menerima_foto_nyata():
    """E2E: foto potret valid harus lolos seluruh aturan F1–F6."""
    photo = _sample_photo()
    result = FaceGate(_detector(), CFG).check(photo.read_bytes())

    assert result.passed is True, f"{result.code}: {result.message}"
    assert result.faces_detected == 1
    assert result.detector == "mediapipe_face_detection"
    assert result.face_coverage and 0.15 <= result.face_coverage <= 0.75


def test_face_gate_menolak_gambar_tanpa_wajah():
    """E2E: citra tanpa wajah (noise) harus ditolak NO_FACE_DETECTED."""
    from tests.conftest import make_image_bytes

    result = FaceGate(_detector(), CFG).check(make_image_bytes(400, 500, seed=11))
    assert result.passed is False
    assert result.code == "NO_FACE_DETECTED"
    assert result.requirement_ref
