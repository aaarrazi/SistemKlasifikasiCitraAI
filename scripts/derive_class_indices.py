"""Fase 0/A3 — Menurunkan `class_indices.json` secara EMPIRIS dari model + dataset.

Mengapa tidak cukup menebak urutan alfabet?
    `flow_from_directory` memang mengurutkan nama folder secara alfabet, tetapi nama
    folder di Google Drive bisa berbeda dari salinan lokal. Karena itu urutan label
    DIBUKTIKAN dengan mengukur akurasi model asli pada gambar ber-label.

Skrip ini juga menguji variAN INTERPOLASI resize, karena:
    - pelatihan memakai `flow_from_directory` (interpolasi default Keras = 'nearest')
    - aplikasi memakai BILINEAR
  Dua-tiganya diukur → yang akurasinya tertinggi dipakai (kecocokan preprocessing
  menentukan akurasi, lihat risiko R1 di arsitektur.md).

Pemakaian:
    .venv\\Scripts\\python.exe scripts\\derive_class_indices.py

Keluaran:
    class_indices.json di root proyek  (bila akurasi terbaik >= 50%)
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = Path(r"C:\Berkas Arrazi\SKRIPSI_ARRAZI\Dataset")
MODEL = ROOT / "ModelKlasifikasiGenerativeAIfix3.keras"
OUT = ROOT / "class_indices.json"

CLASSES = ["DALLE", "GEMINI", "MIDJOURNEY", "STABLE_D"]
PER_CLASS = 40          # gambar yang diuji per kelas (total 160)
BATCH = 16
SEED = 42
MIN_ACC = 0.50          # ambang penerimaan (acak = 0.25)

VARIANTS = {
    # nama            → mode PIL
    "nearest (Keras)": Image.Resampling.NEAREST,
    "bilinear (app)": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def pick_files() -> list[tuple[str, Path]]:
    rng = np.random.default_rng(SEED)
    chosen: list[tuple[str, Path]] = []
    for cls in CLASSES:
        folder = DATASET / cls
        if not folder.is_dir():
            raise SystemExit(f"[!] Folder kelas tidak ditemukan: {folder}")
        files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if len(files) < PER_CLASS:
            raise SystemExit(f"[!] {cls}: hanya {len(files)} gambar (butuh {PER_CLASS})")
        idx = rng.choice(len(files), size=PER_CLASS, replace=False)
        chosen += [(cls, files[int(i)]) for i in idx]
    return chosen


def embed(path: Path, resample: Image.Resampling) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((224, 224), resample)
    return np.asarray(img, dtype=np.float32)  # [0,255] — tanpa /255 (A2)


def main() -> int:
    if not MODEL.exists():
        return _fail(f"model tidak ada: {MODEL}")
    if not DATASET.is_dir():
        return _fail(f"dataset tidak ada: {DATASET}")

    print("=" * 74)
    print("PENURUNAN class_indices.json (EMPIRIS) — A3 / Fase 0")
    print("=" * 74)
    print(f"model   : {MODEL.name}")
    print(f"dataset : {DATASET}")
    print(f"uji     : {PER_CLASS} gambar x {len(CLASSES)} kelas = {PER_CLASS * len(CLASSES)} gambar")
    print(f"acak    : seed={SEED} (reproducible)")

    samples = pick_files()
    true_idx = np.array([CLASSES.index(c) for c, _ in samples])

    import keras  # tensorflow backend

    model = keras.models.load_model(MODEL, compile=False)
    print(f"model   : input={model.input_shape} output={model.output_shape} "
          f"keras={keras.__version__}")

    results: list[tuple[float, str, tuple[str, ...]]] = []
    for name, resample in VARIANTS.items():
        probs = np.empty((len(samples), 4), dtype=np.float32)
        for start in range(0, len(samples), BATCH):
            chunk = samples[start:start + BATCH]
            batch = np.stack([embed(p, resample) for _, p in chunk])
            probs[start:start + len(chunk)] = model(batch, training=False).numpy()

        pred_idx = probs.argmax(axis=1)
        best_for_variant = -1.0
        for perm in itertools.permutations(CLASSES):
            # perm[predicted_index] = label yang dituliskan untuk unit output tsb
            mapped = np.array([perm[i] for i in pred_idx])
            true_names = np.array([CLASSES[i] for i in true_idx])
            acc = float((mapped == true_names).mean())
            results.append((acc, name, perm))
            best_for_variant = max(best_for_variant, acc)
        print(f"  varian {name:<16} akurasi terbaik: {best_for_variant:.1%}")

    results.sort(key=lambda r: r[0], reverse=True)

    print("\n--- 6 urutan label teratas (semua varian) ---")
    for acc, var, perm in results[:6]:
        print(f"  {acc:6.1%}  [{var}]  " + " | ".join(perm))

    top_acc, top_var, top_perm = results[0]
    second = results[1]
    print(f"\nTERBAIK : {top_acc:.1%}  ({top_var})")
    print("  urutan : " + " | ".join(top_perm))
    print(f"  selisih dari urutan terbaik kedua: {top_acc - second[0]:.1%}")

    if top_acc < MIN_ACC:
        return _fail(
            f"akurasi terbaik hanya {top_acc:.1%} (ambang {MIN_ACC:.0%}). "
            "Urutan label/interpolasi belum terkonfirmasi — jangan menebak."
        )

    mapping = {name: i for i, name in enumerate(top_perm)}
    OUT.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    print(f"\n[OK] ditulis: {OUT}")
    print("     " + json.dumps(mapping))
    print("\nLangkah berikutnya:")
    print("  1) python scripts/verify_model.py   (harus: Output units 4, label cocok)")
    print("  2) pytest -q")
    print("  3) restart server -> /health harus 'ok'")
    if top_var.startswith("nearest"):
        print("\n⚠️  Varian NEAREST menang → samakan interpolasi di "
              "app/models/ml/preprocessor.py (Image.Resampling.NEAREST)")
    return 0


def _fail(msg: str) -> int:
    print(f"\n[!] GAGAL: {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
