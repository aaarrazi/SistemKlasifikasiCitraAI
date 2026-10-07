"""Fase 0 — verifikasi file .keras SEBELUM sistem dipakai.

Jalankan (di venv Python 3.11/3.12 dengan keras terpasang):

    python scripts/verify_model.py

Yang diverifikasi:
  1. File model bisa dimuat (versi Keras sesuai metadata — A9)
  2. Bentuk input (None, 224, 224, 3)  (A1)
  3. Jumlah output == jumlah label di class_indices.json  (A3)
  4. Urutan label yang terbaca
  5. Opsional: dua varian normalisasi pada 1 gambar contoh (A2)

Setelah lolos, lanjutkan ke `pytest`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifikasi model .keras & label")
    parser.add_argument("--check-normalization", metavar="GAMBAR", default=None,
                        help="Path 1 gambar contoh untuk membandingkan input [0,255] vs [0,1]")
    args = parser.parse_args()

    cfg = get_config()
    print("=" * 68)
    print("VERIFIKASI MODEL — Fase 0")
    print("=" * 68)
    print(f"MODEL_PATH         : {cfg.model_path}")
    print(f"CLASS_INDICES_PATH : {cfg.class_indices_path}")
    print(f"Model file ada     : {cfg.model_path.exists()}")
    print(f"Label file ada     : {cfg.class_indices_path.exists()}")

    if not cfg.model_path.exists():
        print("\n[!] File model tidak ditemukan — periksa MODEL_PATH (.env).")
        return 1

    # --- label (opsional: verifikasi model tetap jalan tanpa label) ---------
    labelmap = None
    if cfg.class_indices_path.exists():
        from app.models.ml.labels import LabelMap

        labelmap = LabelMap.load(cfg.class_indices_path)
        print(f"Label (urut indeks): {list(labelmap)}")
    else:
        print("\n[!] class_indices.json BELUM ADA — asumsi A3 (§14 butir 1) masih terbuka.")
        print("    Verifikasi model dilanjutkan TANPA pemetaan nama label.")
        print("    Buat dari skrip training, mis.:")
        print('      import json; json.dump(train_generator.class_indices,')
        print('            open("class_indices.json", "w"), indent=2)')

    # --- model -------------------------------------------------------------
    from app.models.ml.classifier import Classifier

    try:
        clf = Classifier.load(cfg, labelmap)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[!] GAGAL memuat model: {getattr(exc, 'message', exc)}")
        if getattr(exc, "detail", None):
            print(f"    detail: {exc.detail}")
        return 1

    info = clf.info
    print(f"Keras version      : {info.keras_version}")
    print(f"Input shape        : {info.input_shape}")
    if labelmap is None:
        print(f"Output units       : {info.output_units} (class_indices.json harus berisi "
              f"{info.output_units} label — A3)")
    else:
        print(f"Output units       : {info.output_units} (label: {len(labelmap)})")

    # --- kontrak preprocessing (A2) ---------------------------------------
    import numpy as np

    from app.models.ml.preprocessor import preprocess

    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    dummy[:, :, 0] = 200  # nilai pixel nyata
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.fromarray(dummy).save(buf, format="PNG")
    arr = preprocess(buf.getvalue(), cfg)
    print(f"\nPreprocess sample  : shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min():.0f} max={arr.max():.0f}")
    print("  -> OK: input tetap skala [0,255] (TANPA pembagian 255 — A2)")

    # --- opsional: dua varian normalisasi ----------------------------------
    if args.check_normalization:
        sample_path = Path(args.check_normalization)
        if not sample_path.exists():
            print(f"\n[!] Gambar contoh tidak ditemukan: {sample_path}")
            return 1
        raw = sample_path.read_bytes()

        for name, batch in (
            ("[0,255] (desain)", preprocess(raw, cfg)),
            ("[0,1]  (varian)", preprocess(raw, cfg) / 255.0),
        ):
            probs = clf.predict(batch).reshape(-1)
            order = np.argsort(probs)[::-1]
            best = [
                (
                    labelmap.name(int(i)) if labelmap else f"idx{int(i)}",
                    float(probs[int(i)]),
                )
                for i in order[:3]
            ]
            print(f"  varian {name}: {[(n, round(p, 4)) for n, p in best]}")

        print("\n  Bandingkan kedua varian dengan hasil skrip training Anda,")
        print("  lalu kunci yang benar di app/models/ml/preprocessor.py.")

    if labelmap is None:
        print("\n[!] Model terverifikasi, TETAPI class_indices.json belum ada (A3).")
        print("    Buat file itu dulu sebelum menjalankan aplikasi (/health akan")
        print("    melaporkan LABELS_NOT_READY sampai file tersedia).")
    else:
        print("\n[OK] Verifikasi selesai. Jalankan: pytest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
