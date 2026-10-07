"""Test Fase 8 — riwayat (audit §2.2), endpoint batch, dan rate limiter."""

from __future__ import annotations

import time

from app.deps import get_classifier, get_face_gate
from app.errors import ModelNotReadyError
from app.middleware import RateLimiter
from tests.conftest import RejectingFaceGate, make_image_bytes

PORTRAIT = make_image_bytes()
TXT = b"bukan-gambar"


def _post(client, content: bytes, filename: str):
    return client.post(
        "/api/v1/predict",
        files={"file": (filename, content, "image/png")},
        data={"top_k": "4", "agree_terms": "true"},
    )


# --- riwayat ----------------------------------------------------------------


def test_history_menyimpan_prediksi_sukses(client):
    res = _post(client, PORTRAIT, "potret.png")
    assert res.status_code == 200
    request_id = res.json()["request_id"]

    page = client.get("/api/v1/history").json()
    assert page["total"] == 1
    item = page["items"][0]
    assert item["request_id"] == request_id
    assert item["label"] == "midjourney"
    assert item["status"] == "high"
    assert item["face_status"] == "passed"
    assert item["reject_code"] is None
    assert item["processing_ms"] >= 0


def test_history_menyimpan_alasan_penolakan(client):
    fastapi_overrides(client, gate=RejectingFaceGate(code="NO_FACE_DETECTED", faces=0))

    res = _post(client, PORTRAIT, "pemandangan.png")
    assert res.status_code == 422

    page = client.get("/api/v1/history").json()
    assert page["total"] == 1
    item = page["items"][0]
    assert item["reject_code"] == "NO_FACE_DETECTED"
    assert item["label"] is None
    assert item["face_status"] == "rejected"


def test_history_tidak_mencatat_gangguan_infrastruktur(client):
    """503 (model belum siap) bukan penolakan konten → tidak masuk riwayat."""
    def _raise():
        raise ModelNotReadyError("model belum dimuat")

    client.app.dependency_overrides[get_classifier] = _raise

    res = _post(client, PORTRAIT, "potret.png")
    assert res.status_code == 503

    assert client.get("/api/v1/history").json()["total"] == 0


def test_history_delete(client):
    assert _post(client, PORTRAIT, "potret.png").status_code == 200
    item_id = client.get("/api/v1/history").json()["items"][0]["id"]

    res = client.delete(f"/api/v1/history/{item_id}")
    assert res.status_code == 200
    assert res.json() == {"deleted": True, "id": item_id}
    assert client.get("/api/v1/history").json()["total"] == 0

    res = client.delete(f"/api/v1/history/{item_id}")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "HISTORY_NOT_FOUND"


def test_history_paginasi(client):
    for i in range(3):
        assert _post(client, PORTRAIT, f"potret-{i}.png").status_code == 200

    page1 = client.get("/api/v1/history", params={"page": 1, "size": 2}).json()
    page2 = client.get("/api/v1/history", params={"page": 2, "size": 2}).json()
    assert page1["total"] == 3
    assert len(page1["items"]) == 2
    assert len(page2["items"]) == 1
    # terbaru di halaman pertama
    assert page1["items"][0]["id"] > page2["items"][0]["id"]


def test_history_unavailable_bila_db_mati(client):
    """Endpoint riwayat harus 503 bila DB tidak tersedia — tapi prediksi tetap jalan."""
    db = client.app.state.db
    try:
        client.app.state.db = None

        res = client.get("/api/v1/history")
        assert res.status_code == 503
        assert res.json()["error"]["code"] == "HISTORY_UNAVAILABLE"

        # prediksi TETAP diproses (history bersifat fail-soft, bukan wajib)
        assert _post(client, PORTRAIT, "potret.png").status_code == 200
    finally:
        client.app.state.db = db


# --- batch ------------------------------------------------------------------


def _post_batch(client, files):
    return client.post(
        "/api/v1/predict/batch",
        files=[("files", (name, content, "image/png")) for name, content in files],
        data={"top_k": "4"},
    )


def test_batch_mencampur_sukses_dan_penolakan(client):
    res = _post_batch(client, [("valid.png", PORTRAIT), ("catatan.txt", TXT)])
    assert res.status_code == 200

    body = res.json()
    assert body["success"] is True
    assert body["count"] == 2

    first, second = body["results"]
    assert first["success"] is True
    assert first["prediction"]["label"] == "midjourney"
    assert first["face_check"]["status"] == "passed"

    # satu file gagal TIDAK membatalkan file lainnya
    assert second["success"] is False
    assert second["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"
    assert second["request_id"]


def test_batch_melebihi_batas(client):
    files = [("a.png", PORTRAIT), ("b.png", PORTRAIT), ("c.png", PORTRAIT)]
    res = _post_batch(client, files)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "BATCH_TOO_LARGE"


def test_batch_setiap_item_punya_request_id(client):
    """Setiap item batch punya request_id sendiri-sendiri (untuk audit)."""
    res = _post_batch(client, [("a.png", PORTRAIT), ("b.png", PORTRAIT)])
    items = res.json()["results"]
    ids = [i["request_id"] for i in items]
    assert all(ids)
    assert ids[0] != ids[1]


# --- rate limiter (unit) ----------------------------------------------------


def test_rate_limiter_menolak_setelah_batas():
    limiter = RateLimiter(per_minute=2, window_seconds=60.0)
    assert limiter.check("1.1.1.1:/api/v1/predict") == (True, 0)
    assert limiter.check("1.1.1.1:/api/v1/predict") == (True, 0)

    allowed, retry_after = limiter.check("1.1.1.1:/api/v1/predict")
    assert allowed is False
    assert retry_after >= 1

    # IP lain tidak terpengaruh
    assert limiter.check("2.2.2.2:/api/v1/predict") == (True, 0)


def test_rate_limiter_nonaktif_saat_nol():
    limiter = RateLimiter(per_minute=0)
    assert limiter.enabled is False
    for _ in range(50):
        assert limiter.check("1.1.1.1:/x") == (True, 0)


def test_rate_limiter_jendela_waktu_berakhir():
    limiter = RateLimiter(per_minute=1, window_seconds=0.05)
    assert limiter.check("ip:/x")[0] is True
    assert limiter.check("ip:/x")[0] is False
    time.sleep(0.08)
    assert limiter.check("ip:/x")[0] is True


# --- helper -----------------------------------------------------------------


def fastapi_overrides(client, *, gate=None, classifier=None) -> None:
    from app.main import app as fastapi_app

    if gate is not None:
        fastapi_app.dependency_overrides[get_face_gate] = lambda: gate
    if classifier is not None:
        fastapi_app.dependency_overrides[get_classifier] = lambda: classifier
