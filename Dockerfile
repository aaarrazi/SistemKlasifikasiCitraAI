# Python 3.12 — TensorFlow belum mendukung 3.14 (lihat arsitektur.md §2.3)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STRICT_STARTUP=true \
    MODEL_PATH=/app/ModelKlasifikasiGenerativeAIfix3.keras \
    CLASS_INDICES_PATH=/app/class_indices.json

WORKDIR /app

# dependensi dulu agar lapisan Docker cache terpakai saat kode berubah
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Seluruh isi proyek (lihat .dockerignore). class_indices.json ikut terbawa HANYA
# bila file sudah ada — bila belum, container langsung berhenti (STRICT_STARTUP=true)
# dengan pesan yang jelas (arsitektur.md §7.1).
COPY . .

EXPOSE 8000

# readiness (bukan liveness): 503 bila model/detektor/label belum siap
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys;r=urllib.request.urlopen('http://127.0.0.1:8000/health?ready=true');sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
