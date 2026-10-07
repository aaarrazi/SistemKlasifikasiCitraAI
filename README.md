# Sistem Klasifikasi Citra Hasil Generate AI

Klasifikasi citra hasil generate AI pada 4 platform — **Gemini · DALL·E · Midjourney · Stable Diffusion**
— menggunakan CNN (EfficientNetB0) yang sudah dilatih sebelumnya, dengan arsitektur
**MVC (Controller / Service / Model / View)** berbasis FastAPI.

Dokumen lengkap: **[`arsitektur.md`](./arsitektur.md)**

> ⚠️ **Batasan model:** hanya dilatih pada citra **satu wajah manusia, cakupan wajah sampai
> pundak**. Sistem menampilkan ketentuan ke pengguna **sebelum unggah** (§2.2) dan
> **menolak** gambar di luar ketentuan lewat **Face Gate** sebelum inferensi (§7.2).

## Struktur (MVC)

```
app/
├── main.py              # Bootstrap: lifespan, error handler, router, View
├── config.py            # Semua konstanta (env-driven)
├── deps.py              # Dependency injection (FastAPI Depends)
├── controllers/         # CONTROLLER — hanya HTTP
├── services/            # Use-case: validasi → Face Gate → preprocess → prediksi
├── models/ml/           # MODEL: classifier .keras, face_gate, preprocessor, labels
├── schemas/             # DTO request/response (Pydantic)
└── views/static/        # VIEW: index.html + css + js (hanya bicara ke API)
```

## Prasyarat (Fase 0)

| Item | Status |
|------|--------|
| Python **3.11 / 3.12** (venv `.venv`, sudah dibuat) | ✅ TensorFlow 2.21 + Keras 3.13.2 terpasang |
| `ModelKlasifikasiGenerativeAIfix3.keras` | ✅ sudah ada di root |
| `models/blaze_face_short_range.tflite` | ✅ model detektor wajah (±230 KB) |
| `class_indices.json` | ✅ **sudah ada** — `{"DALLE":0, "GEMINI":1, "MIDJOURNEY":2, "STABLE_D":3}` |

### Catatan saat clone dari GitHub

| File | Ikut repo? | Keterangan |
|------|-----------|------------|
| Kode (`app/`, `tests/`, `scripts/`) | ✅ | ±1 MB |
| `class_indices.json` | ✅ | wajib ada — tanpa ini prediksi menolak (`LABELS_NOT_READY`) |
| `models/blaze_face_short_range.tflite` | ✅ (±230 KB) | model detektor wajah; bisa diunduh ulang `python scripts/download_face_model.py` |
| **`*.keras` (46,6 MB)** | ❌ **tidak** | dikecualikan `.gitignore` → **letakkan file model di root** proyek (atau set env `MODEL_PATH`) sebelum menjalankan |
| Dataset 800 gambar | ❌ | berada di luar proyek; ubah konstanta `DATASET` di `scripts/derive_class_indices.py` & `evaluate_dataset.py` bila pindah mesin |

### Asal-usul `class_indices.json` (A3)

Urutan label **tidak ditebak** — dibuktikan ke model asli oleh
`scripts/derive_class_indices.py`: 24 permutasi label × 4 varian interpolasi diuji
pada 160 gambar dataset (`C:\Berkas Arrazi\SKRIPSI_ARRAZI\Dataset`).

Hasilnya dua temuan sekaligus:

```
96,9 %  nearest (interpolasi default Keras saat training)  DALLE|GEMINI|MIDJOURNEY|STABLE_D
85,6 %  lanczos   |   84,4 % bicubic   |   81,2 % bilinear (varian lama aplikasi)
```

1. **Urutan label** = alfabet folder → `DALLE=0, GEMINI=1, MIDJOURNEY=2, STABLE_D=3` (96,9 % vs peluang 25 %)
2. **Interpolasi resize wajib `NEAREST`** — memakai `BILINEAR` membuat akurasi turun **15,7 poin** diam-diam (risiko R1)

Bila Anda punya akses ke lingkungan training, versi otoritatifnya tetap:

```python
import json
json.dump(train_generator.class_indices, open("class_indices.json", "w"), indent=2)
```

Ulangi pembuktian ulang kapan pun: `python scripts/derive_class_indices.py`

## Menjalankan

```bash
# 1) Fase 0 — buat .venv Python 3.11/3.12 + install semua dependensi
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
.venv\Scripts\activate          # Windows

# 2) unduh model detektor wajah BlazeFace (sekali saja, ±230 KB)
python scripts/download_face_model.py

# 3) konfigurasi
copy .env.example .env          # isi CLASS_INDICES_PATH dsb.

# 4) verifikasi model (A1/A2/A3)
python scripts/verify_model.py

# 5) uji
pytest

# 6) jalankan
uvicorn app.main:app --reload
# buka http://127.0.0.1:8000   (UI)  dan  /docs (Swagger)
```

Setelah semua siap, `/health` harus `status: ok` dan `/health?ready=true` membalas **200**.

### Evaluasi akurasi (Fase 7)

```bash
# lewat API penuh (Tier 1 → Face Gate → model): 32/32 lolos gate benar semua
python scripts/evaluate_dataset.py --mode=http --per-class 10

# model murni tanpa gate: 95,0 % top-1 pada 80 gambar + confusion matrix
python scripts/evaluate_dataset.py --mode=direct --per-class 20
```

Tanpa `class_indices.json`, TensorFlow/mediapipe, atau model BlazeFace, server **tetap jalan**
dalam mode *degraded* (`/health` → `status: degraded`, `/health?ready=true` → **503**) dan
endpoint prediksi membalas **503** dengan alasan — lihat `STRICT_STARTUP` di `.env.example`
untuk mode produksi (gagal-start).

## API utama

| Method | Path | Keterangan |
|--------|------|-----------|
| POST | `/api/v1/predict` | Unggah gambar → label + confidence (ditolak bila di luar ketentuan) |
| POST | `/api/v1/predict/batch` | Banyak gambar dalam 1 request — item ditangani terpisah (maks `BATCH_MAX_FILES`) |
| GET | `/api/v1/requirements` | Teks ketentuan §2.2 untuk View/klien |
| GET | `/api/v1/model/info` | Metadata model & urutan label |
| GET | `/api/v1/history?page=1` | Riwayat prediksi **&** alasan penolakan (SQLite, terbaru dulu) |
| DELETE | `/api/v1/history/{id}` | Hapus 1 entri riwayat |
| GET | `/health` | Kesiapan model / detektor wajah / DB — `?ready=true` → 503 bila belum siap |

Semua path `/api/` dibatasi `RATE_LIMIT_PER_MINUTE` per IP (default 60, `0` = mati);
respons `429` menyertakan header `Retry-After`.

Contoh penolakan (422):

```json
{
  "success": false,
  "error": {
    "code": "MULTIPLE_FACES",
    "message": "Terdeteksi lebih dari satu orang. Unggah foto yang hanya berisi satu orang.",
    "requirement_ref": "§2.2 — ❌ Lebih dari satu orang dalam satu gambar"
  },
  "request_id": "..."
}
```

## Pengujian

```bash
pytest -q          # 56 test (di .venv): Model ML, Face Gate, Service, Controller
                   # (fail-closed), riwayat SQLite, batch, rate limiter,
                   # + model .keras ASLI & detektor BlazeFace ASLI
```

Catatan:

- Test API memakai `class_indices` **sintetis** (`tests/fixtures/fake_class_indices.json`)
  serta classifier/detector tiruan → mayoritas test jalan **tanpa** TensorFlow.
- `tests/test_model_load_real.py` (model asli) dan `tests/test_mediapipe_detector.py`
  (detektor asli pada foto nyata) **auto-skip** bila Keras/mediapipe belum ada —
  jalankan di `.venv` agar ikut teruji.
- Tanpa `class_indices.json`, `/health` tetap `degraded` (`LABELS_NOT_READY`)
  meski model sudah termuat.
