"""Fase 7 — Evaluasi model pada dataset asli, lewat API sungguhan (end-to-end).

Menggunakan label folder dataset sebagai ground truth, sehingga akurasi yang
dilaporkan adalah akurasi SESUNGUHNYA, bukan klaim.

Dua mode:
  --mode=http    (default) kirim lewat POST /api/v1/predict → menguji SELURUH
                 tumpukan: Tier 1 → Face Gate → preprocessing → model → ambang
  --mode=direct  panggil model langsung (tanpa Face Gate) → akurasi murni model,
                 dipakai untuk confusion matrix Fase 7

Pemakaian:
    .venv\\Scripts\\python.exe scripts/evaluate_dataset.py --per-class 10
    .venv\\Scripts\\python.exe scripts/evaluate_dataset.py --mode=direct --per-class 20
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = Path(r"C:\Berkas Arrazi\SKRIPSI_ARRAZI\Dataset")
URL = "http://127.0.0.1:8000"
MAX_MB = 9  # hindari 413 (batas upload 10 MB)

# skrip dijalankan dari folder scripts/ → root proyek harus ikut di sys.path
# agar `import app.*` (mode direct) menemukan paket aplikasi
sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> dict[str, int]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def sample_files(per_class: int, seed: int) -> list[tuple[str, Path]]:
    idx_map = load_json(ROOT / "class_indices.json")
    rng = random.Random(seed)
    out: list[tuple[str, Path]] = []
    for cls in sorted(idx_map, key=lambda k: idx_map[k]):
        folder = DATASET / cls
        files = [
            p for p in folder.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            and p.stat().st_size < MAX_MB * 1024 * 1024
        ]
        if len(files) < per_class:
            raise SystemExit(f"[!] {cls}: hanya {len(files)} gambar < {MAX_MB} MB")
        out += [(cls, p) for p in rng.sample(files, per_class)]
    return out


def run_http(samples: list[tuple[str, Path]], idx_map: dict[str, int]) -> int:
    import httpx

    names = {v: k for k, v in idx_map.items()}
    accepted = correct = 0
    rejects: dict[str, int] = {}
    confusion = np.zeros((len(idx_map), len(idx_map)), dtype=int)
    rejected_samples: list[str] = []

    with httpx.Client(timeout=60) as client:
        for true_cls, path in samples:
            with path.open("rb") as fh:
                res = client.post(
                    f"{URL}/api/v1/predict",
                    files={"file": (path.name, fh, "application/octet-stream")},
                    data={"top_k": str(len(idx_map)), "agree_terms": "true"},
                )
            body = res.json()
            if res.status_code == 200:
                pred = body["prediction"]["label"]
                accepted += 1
                if pred == true_cls:
                    correct += 1
                confusion[idx_map[true_cls], idx_map[pred]] += 1
            else:
                code = body.get("error", {}).get("code", "?")
                rejects[code] = rejects.get(code, 0) + 1
                if len(rejected_samples) < 6:
                    rejected_samples.append(f"{path.name} ({true_cls}) -> {code}")

    total = len(samples)
    print(f"\n=== HASIL HTTP ({accepted}/{total} diterima Face Gate) ===")
    if accepted:
        print(f"akurasi end-to-end (yang lolos gate) : {correct}/{accepted} = "
              f"{correct / accepted:.1%}")
    else:
        print("tidak ada gambar yang lolos Face Gate")
    if rejects:
        print("penolakan Face Gate / Tier 1:")
        for code, n in sorted(rejects.items(), key=lambda kv: -kv[1]):
            print(f"   {code:<26} {n}")
        print("   contoh:")
        for line in rejected_samples:
            print(f"     - {line}")
    if accepted:
        print_matrix(confusion, names)
    return 0


def run_direct(samples: list[tuple[str, Path]], idx_map: dict[str, int]) -> int:
    from app.config import get_config
    from app.models.ml.classifier import Classifier
    from app.models.ml.labels import LabelMap
    from app.models.ml.preprocessor import preprocess

    cfg = get_config()
    labelmap = LabelMap.load(ROOT / "class_indices.json")
    clf = Classifier.load(cfg, labelmap)
    names = {i: labelmap.name(i) for i in range(len(labelmap))}

    confusion = np.zeros((len(idx_map), len(idx_map)), dtype=int)
    top1 = 0
    started = time.perf_counter()
    for true_cls, path in samples:
        batch = preprocess(path.read_bytes(), cfg)
        probs = clf.predict(batch).reshape(-1)
        pred = int(probs.argmax())
        confusion[idx_map[true_cls], pred] += 1
        if names[pred] == true_cls:
            top1 += 1
    took = time.perf_counter() - started

    total = len(samples)
    print(f"\n=== HASIL DIRECT (tanpa Face Gate) — {total} gambar dalam {took:.1f} dtk ===")
    print(f"akurasi top-1 model                 : {top1}/{total} = {top1 / total:.1%}")
    print_matrix(confusion, names)
    return 0


def print_matrix(matrix: np.ndarray, names: dict[int, str]) -> None:
    order = sorted(names)
    label = [names[i] for i in order]
    print("\nconfusion matrix (baris = kelas asli, kolom = prediksi):")
    print(f"{'':<12}" + "".join(f"{l[:11]:>13}" for l in label))
    for r, true_name in enumerate(label):
        row = matrix[order[r]]
        line = f"{true_name:<12}" + "".join(f"{int(v):>13}" for v in row)
        acc = int(row[order.index(order[r])])
        line += f"   | akurasi {acc}/{int(row.sum())} = {acc / max(1, row.sum()):.0%}"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluasi model pada dataset asli")
    ap.add_argument("--mode", choices=("http", "direct"), default="http")
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (ROOT / "class_indices.json").exists():
        return print("[!] class_indices.json belum ada — jalankan "
                     "scripts/derive_class_indices.py dulu") or 1
    if not DATASET.is_dir():
        return print(f"[!] dataset tidak ditemukan: {DATASET}") or 1

    idx_map = load_json(ROOT / "class_indices.json")
    samples = sample_files(args.per_class, args.seed)
    print(f"dataset : {DATASET}")
    print(f"sample  : {args.per_class}/kelas × {len(idx_map)} = {len(samples)} gambar "
          f"(seed {args.seed})")

    if args.mode == "http":
        try:
            import httpx  # noqa: F401
        except ImportError:
            return print("[!] httpx belum terpasang") or 1
        try:
            httpx.get(f"{URL}/health", timeout=3)
        except Exception as exc:  # noqa: BLE001
            return print(f"[!] server tidak hidup di {URL}: {exc}") or 1
        return run_http(samples, idx_map)
    return run_direct(samples, idx_map)


if __name__ == "__main__":
    sys.exit(main())
