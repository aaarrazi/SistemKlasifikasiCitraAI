"""Unit test Face Gate — aturan F1..F6 (arsitektur.md §7.2)."""

from __future__ import annotations

import pytest

from app.config import get_config
from app.errors import FaceCheckUnavailableError
from app.models.ml.face_gate import Face, FaceGate
from tests.conftest import make_image_bytes


class FixedDetector:
    """Detektor tetap: mengembalikan daftar kotak yang sudah ditentukan."""

    name = "fixed"

    def __init__(self, faces=None, boom: bool = False):
        self.faces = faces or []
        self.boom = boom

    def detect(self, img):
        if self.boom:
            raise RuntimeError("simulasi crash detektor")
        return self.faces


CFG = get_config()

# kotak wajah valid untuk gambar 512x640: coverage 0.406, pusat (0.50, 0.33)
NORMAL_FACE = Face(left=156, top=80, width=200, height=260, confidence=0.95)


def gate(faces=None, boom=False) -> FaceGate:
    return FaceGate(FixedDetector(faces=faces, boom=boom), CFG)


def test_lolos_satu_wajah_framing_benar():
    result = gate([NORMAL_FACE]).check(make_image_bytes())
    assert result.passed is True
    assert result.code is None
    assert result.faces_detected == 1
    assert result.detector == "fixed"
    assert 0.15 <= result.face_coverage <= 0.75


def test_f3_tidak_ada_wajah():
    result = gate([]).check(make_image_bytes())
    assert result.passed is False
    assert result.code == "NO_FACE_DETECTED"
    assert result.message and "wajah" in result.message
    assert result.requirement_ref


def test_f4_lebih_dari_satu_wajah():
    other = Face(left=20, top=90, width=120, height=170, confidence=0.9)
    result = gate([NORMAL_FACE, other]).check(make_image_bytes())
    assert result.code == "MULTIPLE_FACES"
    assert result.faces_detected == 2


def test_f5_wajah_terlalu_kecil():
    tiny = Face(left=200, top=300, width=60, height=60, confidence=0.9)  # 60/640=0.094
    result = gate([tiny]).check(make_image_bytes())
    assert result.code == "FACE_TOO_SMALL"


def test_f5_wajah_terlalu_besar():
    huge = Face(left=10, top=20, width=480, height=600, confidence=0.9)  # 600/640=0.94
    result = gate([huge]).check(make_image_bytes())
    assert result.code == "FACE_TOO_LARGE"


def test_f6_wajah_di_luar_bingkai():
    edge = Face(left=400, top=80, width=110, height=260, confidence=0.9)  # pusat x=0.889
    result = gate([edge]).check(make_image_bytes())
    assert result.code == "INVALID_FRAMING"


def test_f1_resolusi_terlalu_rendah():
    result = gate([NORMAL_FACE]).check(make_image_bytes(100, 100))
    assert result.code == "IMAGE_TOO_LOW_RES"


def test_f2_rasio_aspek_terlalu_lebar():
    result = gate([NORMAL_FACE]).check(make_image_bytes(1000, 400))  # 2.5 > 2.0
    assert result.code == "INVALID_FRAMING"


def test_file_bukan_gambar():
    result = gate([NORMAL_FACE]).check(b"bukan-gambar")
    assert result.passed is False
    assert result.code == "INVALID_IMAGE"


def test_fail_closed_sa_detektor_crash():
    """Detektor gagal → request DITOLAK, bukan dilanjutkan (§7.2)."""
    with pytest.raises(FaceCheckUnavailableError) as exc:
        gate(boom=True).check(make_image_bytes())
    assert exc.value.code == "FACE_CHECK_UNAVAILABLE"
    assert exc.value.status_code == 503


def test_detektor_off_selalu_gagal():
    """FACE_DETECTOR=off harus menolak semua prediksi (fail-closed)."""
    from app.errors import FaceCheckUnavailableError as Err

    from app.models.ml.face_gate import create_detector

    with pytest.raises(Err):
        create_detector(CFG)  # conftest menyet FACE_DETECTOR=off
