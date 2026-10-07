"""Integration test Controller — `TestClient` memanggil endpoint sungguhan."""

from __future__ import annotations

from app.deps import get_classifier, get_face_gate
from app.errors import ModelNotReadyError
from app.main import app as fastapi_app
from tests.conftest import (
    FAKE_PROBS,
    BrokenFaceGate,
    RejectingFaceGate,
    make_image_bytes,
)

PORTRAIT = make_image_bytes()  # 512x640 PNG


def post_image(client, *, content: bytes | None = None, filename: str = "potret.png",
                data: dict | None = None):
    payload = {"top_k": "4", "agree_terms": "true"}
    if data:
        payload.update(data)
    return client.post(
        "/api/v1/predict",
        files={"file": (filename, content if content is not None else PORTRAIT, "image/png")},
        data=payload,
    )


# --- View & endpoint dasar ---------------------------------------------------


def test_index_sajikan_view(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Ketentuan" in res.text


def test_health_laporkan_komponen(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    for key in ("status", "model_loaded", "face_detector_loaded", "labels_loaded", "uptime_s"):
        assert key in body
    assert body["labels_loaded"] is True  # fake_class_indices.json ikut termuat


def test_health_readiness_memberi_503_bila_belum_siap(client):
    """`/health` tetap 200 (liveness); `?ready=true` → 503 (readiness/Docker)."""
    live = client.get("/health")
    assert live.status_code == 200
    assert live.json()["labels_loaded"] is True

    ready = client.get("/health", params={"ready": "true"})
    assert ready.status_code == 503  # detektor dimatikan oleh env test (FACE_DETECTOR=off)
    body = ready.json()
    assert body["status"] == "degraded"
    assert body["face_detector_loaded"] is False
    assert body["startup_error"]


def test_requirements_menyediakan_copy_ketentuan(client):
    res = client.get("/api/v1/requirements")
    assert res.status_code == 200
    body = res.json()
    assert body["title"]
    assert any("satu orang" in item.lower() for item in body["allowed"])
    assert any("lebih dari satu" in item.lower() for item in body["prohibited"])
    assert body["technical"]["max_size_mb"] == 1  # sesuai env testing
    assert body["agree_text"]


def test_model_info(client):
    res = client.get("/api/v1/model/info")
    assert res.status_code == 200
    body = res.json()
    assert body["labels"] == ["dalle", "gemini", "midjourney", "stable_diffusion"]


# --- prediksi sukses ---------------------------------------------------------


def test_predict_sukses(client):
    res = post_image(client)
    assert res.status_code == 200, res.text
    body = res.json()

    assert body["success"] is True
    assert body["prediction"]["label"] == "midjourney"  # index 2 dari FAKE_PROBS
    assert abs(body["prediction"]["confidence"] - FAKE_PROBS[2]) < 1e-6
    assert body["prediction"]["status"] == "high"
    assert len(body["prediction"]["top_k"]) == 4

    assert body["face_check"]["status"] == "passed"
    assert body["face_check"]["faces_detected"] == 1

    assert body["model"]["name"] == "fake-model"
    assert isinstance(body["processing_time_ms"], int)
    assert body["request_id"]
    assert res.headers.get("X-Request-ID")


def test_predict_top_k_dibatasi(client):
    res = post_image(client, data={"top_k": "2"})
    assert res.status_code == 200
    assert len(res.json()["prediction"]["top_k"]) == 2


def test_predict_tanpa_agree_terms_tetap_diproses(client):
    """Checkbox hanya UX — validasi sesungguhnya tetap di server (§2.2)."""
    res = post_image(client, data={"agree_terms": "false"})
    assert res.status_code == 200
    assert res.json()["face_check"]["status"] == "passed"


# --- Tier 1: file -----------------------------------------------------------


def test_tolak_file_bukan_gambar(client):
    res = post_image(client, content=b"isi teks biasa", filename="catatan.txt")
    assert res.status_code == 415
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_tolak_gambar_rusak(client):
    res = post_image(client, content=b"bukan-gambar-tapi-ekstensi-png", filename="rusak.png")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_IMAGE"


def test_tolak_file_terlalu_besar(client):
    res = post_image(client, content=b"x" * 1_100_000, filename="besar.png")
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_tolak_top_k_tidak_valid(client):
    res = post_image(client, data={"top_k": "0"})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "VALIDATION_ERROR"


# --- Tier 2: Face Gate (ketentuan §2.2) -------------------------------------


def test_tolak_pelanggaran_ketentuan_bersama_requirement_ref(client):
    fastapi_app.dependency_overrides[get_face_gate] = lambda: RejectingFaceGate(
        code="MULTIPLE_FACES", faces=2
    )
    res = post_image(client)
    assert res.status_code == 422

    err = res.json()["error"]
    assert err["code"] == "MULTIPLE_FACES"
    assert "lebih dari satu" in err["message"].lower()
    assert err["requirement_ref"].startswith("§2.2")


def test_detektor_crash_menolak_semua_fail_closed(client):
    fastapi_app.dependency_overrides[get_face_gate] = lambda: BrokenFaceGate()
    res = post_image(client)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "FACE_CHECK_UNAVAILABLE"


# --- Model belum siap -------------------------------------------------------


def test_model_tidak_siap_membalas_503(client):
    def _raise():
        raise ModelNotReadyError("model belum dimuat")

    fastapi_app.dependency_overrides[get_classifier] = _raise
    res = post_image(client)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "MODEL_NOT_READY"


def test_server_degraded_tanpa_override(no_overrides_client):
    """Tanpa komponen siap (detektor dimatikan), server tetap jalan tapi siaga 503."""
    health = no_overrides_client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["face_detector_loaded"] is False  # env test: FACE_DETECTOR=off
    assert health["startup_error"]

    res = post_image(no_overrides_client)
    assert res.status_code == 503
    # komponen pertama yang gagal (label/model/gate) yang menentukan kode
    assert res.json()["error"]["code"] in {
        "LABELS_NOT_READY",
        "MODEL_NOT_READY",
        "FACE_CHECK_UNAVAILABLE",
    }
