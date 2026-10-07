"""Unit test Model ML: label map & preprocessor (kontrak A1 + A2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.config import get_config
from app.errors import InvalidImageError, LabelsNotReadyError
from app.models.ml.labels import LabelMap
from app.models.ml.preprocessor import open_rgb, preprocess
from tests.conftest import make_image_bytes


# --- labels -----------------------------------------------------------------


def test_labelmap_urut_berdasarkan_indeks():
    lm = LabelMap.from_mapping({"zeta": 1, "alfa": 0, "beta": 2})
    assert lm.ordered == ("alfa", "zeta", "beta")
    assert lm.name(0) == "alfa"
    assert lm.index("beta") == 2
    assert len(lm) == 3


def test_labelmap_tolak_index_tidak_berurutan():
    with pytest.raises(LabelsNotReadyError):
        LabelMap.from_mapping({"a": 0, "b": 2})


def test_labelmap_muat_dari_file():
    lm = LabelMap.load(Path(__file__).parent / "fixtures" / "fake_class_indices.json")
    assert lm.ordered == ("dalle", "gemini", "midjourney", "stable_diffusion")


def test_labelmap_file_hilang_memberi_error_jelas(tmp_path: Path):
    with pytest.raises(LabelsNotReadyError) as exc:
        LabelMap.load(tmp_path / "class_indices.json")
    assert "class_indices" in str(exc.value.message)


def test_labelmap_json_rusak(tmp_path: Path):
    p = tmp_path / "class_indices.json"
    p.write_text("{tidak valid", encoding="utf-8")
    with pytest.raises(LabelsNotReadyError):
        LabelMap.load(p)


# --- preprocessor -----------------------------------------------------------


def test_preprocess_kontrak_input_model():
    """A1: (1,224,224,3) float32 — A2: rentang [0,255] TANPA pembagian 255."""
    cfg = get_config()
    arr = preprocess(make_image_bytes(512, 640), cfg)

    assert arr.shape == (1, cfg.img_height, cfg.img_width, 3)
    assert arr.dtype == np.float32
    assert float(arr.max()) > 1.0, "Nilai piksel harus tetap skala [0,255] (A2)"


def test_preprocess_konversi_ke_rgb():
    cfg = get_config()
    arr = preprocess(make_image_bytes(300, 300), cfg)
    assert arr.shape[-1] == 3


def test_preprocess_file_rusak():
    with pytest.raises(InvalidImageError):
        preprocess(b"bukan-gambar", get_config())


def test_open_rgb_menolak_bytes_kosong():
    with pytest.raises(InvalidImageError):
        open_rgb(b"")


def test_preprocess_hasil_konsisten():
    """Preprocessing deterministik — dua kali hasil identik."""
    raw = make_image_bytes(seed=7)
    a = preprocess(raw, get_config())
    b = preprocess(raw, get_config())
    assert np.array_equal(a, b)
