"""Uji model ASLI `ModelKlasifikasiGenerativeAIfix3.keras` (Fase 0).

Test ini **auto-skip** bila Keras/TensorFlow tidak terpasang (mis. Python 3.14),
sehingga `pytest` tetap hijau di lingkungan tanpa TF. Jalankan di `.venv`
Python 3.11/3.12 untuk hasil sesungguhnya.

Yang dibuktikan:
  A1 — input (1, 224, 224, 3) float32
  A2 — preprocessing menghasilkan skala [0, 255] (tanpa pembagian 255)
  A3 — jumlah unit output cocok dengan jumlah label
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("keras", reason="Keras/TensorFlow belum terpasang — jalankan di .venv (Fase 0)")

from app.config import BASE_DIR, get_config  # noqa: E402
from app.models.ml.classifier import Classifier  # noqa: E402
from app.models.ml.labels import LabelMap  # noqa: E402
from app.models.ml.preprocessor import preprocess  # noqa: E402
from tests.conftest import make_image_bytes  # noqa: E402

MODEL = BASE_DIR / "ModelKlasifikasiGenerativeAIfix3.keras"
FAKE_LABELS = Path(__file__).parent / "fixtures" / "fake_class_indices.json"

pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="file model .keras tidak ditemukan"
)


@pytest.fixture(scope="module")
def classifier() -> Classifier:
    cfg = replace(get_config(), model_path=MODEL)
    labelmap = LabelMap.load(FAKE_LABELS)
    return Classifier.load(cfg, labelmap)


def test_model_asli_dimuat_dengan_kontrak_a1(classifier: Classifier):
    info = classifier.info
    assert info.input_shape == (None, 224, 224, 3), info.input_shape
    assert info.output_units == 4, info.output_units
    assert info.keras_version and info.keras_version.startswith("3.13")


def test_model_asli_memprediksi_dari_input_255(classifier: Classifier):
    cfg = replace(get_config(), model_path=MODEL)
    arr = preprocess(make_image_bytes(seed=3), cfg)

    # A1 + A2
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert float(arr.max()) > 1.0, "input harus skala [0,255] (A2)"

    probs = classifier.predict(arr)
    assert probs.shape == (1, 4)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert abs(float(probs.sum()) - 1.0) < 1e-4, "softmax harus berjumlah 1"


def test_model_asli_deterministik(classifier: Classifier):
    cfg = replace(get_config(), model_path=MODEL)
    arr = preprocess(make_image_bytes(seed=3), cfg)
    first = classifier.predict(arr)
    second = classifier.predict(arr)
    assert np.allclose(first, second, atol=1e-6)


def test_prediksi_mode_non_training(classifier: Classifier):
    """Dropout(0.4) harus nonaktif — dua panggilan identik (A11)."""
    cfg = replace(get_config(), model_path=MODEL)
    arr = preprocess(make_image_bytes(seed=9), cfg)
    a = classifier.predict(arr)
    b = classifier.predict(arr)
    assert np.array_equal(a, b), "inferensi harus deterministik (training=False)"
