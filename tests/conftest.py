"""Konfigurasi pytest.

PENTING: environment variables diatur SEBELUM modul `app` diimpor,
karena konfigurasi di-cache (`app.config.get_config`).
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(exist_ok=True)

# --- env untuk testing (lihat arsitektur.md §5) -----------------------------
os.environ["CLASS_INDICES_PATH"] = str(FIXTURES / "fake_class_indices.json")
# Path model sengaja dibuat TIDAK ADA: agar test tidak memuat model 46 MB di setiap
# lifespan (lambat). Verifikasi model asli ada di scripts/verify_model.py dan
# tests/test_model_load_real.py (auto-skip bila Keras tidak terpasang).
os.environ["MODEL_PATH"] = str(FIXTURES / "missing_model.keras")
os.environ["STRICT_STARTUP"] = "false"
os.environ["MAX_UPLOAD_MB"] = "1"
os.environ["FACE_DETECTOR"] = "off"  # startup pasti gagal → test pakai fake detector
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"   # uji RateLimiter dilakukan terpisah (unit)
os.environ["DATABASE_PATH"] = ":memory:"     # tiap TestClient dapat DB riwayat segar
os.environ["BATCH_MAX_FILES"] = "2"

from app.deps import get_classifier, get_face_gate  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from app.models.ml.classifier import ModelInfo  # noqa: E402
from app.models.ml.face_gate import FaceGateResult  # noqa: E402
from app.models.ml.labels import requirement_message, requirement_ref  # noqa: E402

# --- helpers ---------------------------------------------------------------

LABELS_FAKE = ["dalle", "gemini", "midjourney", "stable_diffusion"]
FAKE_PROBS = (0.05, 0.07, 0.83, 0.05)  # -> index 2 = "midjourney"


def make_image_bytes(width: int = 512, height: int = 640, seed: int = 0) -> bytes:
    """PNG RGB acak (deterministik) — max pixel 255, jadi menguji kontrak [0,255]."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


class FakeClassifier:
    """Pengganti model .keras (keras tidak terpasang di lingkungan test)."""

    def __init__(self, probs: tuple[float, ...] = FAKE_PROBS):
        self._probs = np.asarray(probs, dtype=np.float64)
        self.info = ModelInfo(
            name="fake-model",
            version="test",
            keras_version=None,
            path="fake.keras",
            input_shape=(None, 224, 224, 3),
            output_units=len(probs),
        )

    def predict(self, batch: np.ndarray) -> np.ndarray:
        # kontrak preprocessing (A2): (1,224,224,3) float32 rentang [0,255]
        assert batch.shape == (1, 224, 224, 3), batch.shape
        assert batch.dtype == np.float32, batch.dtype
        assert float(batch.max()) > 1.0, (
            "Input model harus rentang [0,255] — jangan dibagi 255 (arsitektur.md A2)"
        )
        return self._probs.reshape(1, -1)


class FakeFaceGate:
    """Face Gate yang selalu lolos (1 wajah)."""

    detector_name = "fake_detector"

    def check(self, raw: bytes) -> FaceGateResult:
        return FaceGateResult(
            passed=True, faces_detected=1, face_coverage=0.406, detector=self.detector_name
        )


class RejectingFaceGate:
    """Face Gate yang menolak — dipakai menguji jalur 422."""

    detector_name = "fake_detector"

    def __init__(self, code: str = "MULTIPLE_FACES", faces: int = 2):
        self.code = code
        self.faces = faces

    def check(self, raw: bytes) -> FaceGateResult:
        return FaceGateResult(
            passed=False,
            code=self.code,
            message=requirement_message(self.code),
            requirement_ref=requirement_ref(self.code),
            faces_detected=self.faces,
            face_coverage=0.4,
            detector=self.detector_name,
        )


class BrokenFaceGate:
    """Detektor rusak — menguji sifat FAIL-CLOSED (503)."""

    detector_name = "broken_detector"

    def check(self, raw: bytes) -> FaceGateResult:
        from app.errors import FaceCheckUnavailableError

        raise FaceCheckUnavailableError("detektor crash", detail="simulasi")


# --- fixtures --------------------------------------------------------------


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    fastapi_app.dependency_overrides[get_classifier] = lambda: FakeClassifier()
    fastapi_app.dependency_overrides[get_face_gate] = lambda: FakeFaceGate()
    with TestClient(fastapi_app) as test_client:
        yield test_client
    fastapi_app.dependency_overrides.clear()


@pytest.fixture()
def portrait_png() -> bytes:
    return make_image_bytes()


@pytest.fixture()
def no_overrides_client():
    """Client tanpa override — menguji perilaku saat komponen belum siap."""
    from fastapi.testclient import TestClient

    fastapi_app.dependency_overrides.clear()
    with TestClient(fastapi_app) as test_client:
        yield test_client
