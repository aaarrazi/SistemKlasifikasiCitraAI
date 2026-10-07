"""Konfigurasi terpusat — sumber kebenaran tunggal untuk semua konstanta.

Semua nilai bisa ditimpa lewat environment variable (lihat `.env.example`).
Jangan meng-hardcode ukuran image / threshold di luar file ini.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # root proyek


def _load_dotenv() -> None:
    """Muat `.env` bila python-dotenv tersedia (override=False: env eksternal menang)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env", override=False)
    except Exception:  # pragma: no cover - dotenv opsional
        pass


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_tuple(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    return tuple(p.strip() for p in (raw or default).split(",") if p.strip())


@dataclass(frozen=True)
class Config:
    """Seluruh konfigurasi aplikasi (immutable)."""

    # --- path ---
    base_dir: Path
    model_path: Path
    class_indices_path: Path

    # --- kontrak input model (A1) ---
    img_width: int = 224
    img_height: int = 224

    # --- batasan file (A5) ---
    max_upload_mb: int = 10
    allowed_formats: tuple[str, ...] = ("JPEG", "PNG", "WEBP")
    allowed_ext: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

    # --- Face Gate / ketentuan pengguna (§7.2) ---
    min_side_px: int = 224
    face_detector: str = "auto"
    face_model_path: str = ""  # "" -> BASE_DIR/models/blaze_face_short_range.tflite
    face_min_confidence: float = 0.70
    face_min_coverage: float = 0.15
    face_max_coverage: float = 0.75
    aspect_min: float = 0.5
    aspect_max: float = 2.0
    center_x_min: float = 0.15
    center_x_max: float = 0.85
    center_y_min: float = 0.10
    center_y_max: float = 0.70

    # --- ambang confidence model (§7.5) ---
    confidence_high: float = 0.70
    confidence_medium: float = 0.40
    default_top_k: int = 4

    # --- runtime ---
    strict_startup: bool = False
    model_name: str = "ModelKlasifikasiGenerativeAIfix3"
    model_version: str = "1.0.0"
    cors_origins: tuple[str, ...] = ("*",)
    log_level: str = "INFO"

    # --- Fase 8: riwayat, batch & proteksi ---
    history_enabled: bool = True
    database_path: str = ""  # "" -> BASE_DIR/storage/history.db; ":memory:" untuk test
    rate_limit_per_minute: int = 60  # 0 = nonaktif
    batch_max_files: int = 10

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def input_shape(self) -> tuple[None, int, int, int]:
        """Bentuk input yang diharapkan model (A1)."""
        return (None, self.img_height, self.img_width, 3)

    @property
    def database_path_resolved(self) -> str:
        """Path database riwayat — absolut, dengan fallback ke storage/history.db."""
        if self.database_path:
            if self.database_path == ":memory:":
                return self.database_path
            p = Path(self.database_path)
            return str(p if p.is_absolute() else self.base_dir / p)
        return str(self.base_dir / "storage" / "history.db")

    @property
    def face_model_path_resolved(self) -> Path:
        """Path model BlazeFace untuk MediaPipe Tasks API (diunduh sekali)."""
        p = Path(self.face_model_path) if self.face_model_path else Path(
            "models", "blaze_face_short_range.tflite"
        )
        return p if p.is_absolute() else self.base_dir / p


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Ambil konfigurasi (dipanggil sekali, di-cache)."""
    _load_dotenv()
    img_size = _env_int("IMG_SIZE", 224)
    return Config(
        base_dir=BASE_DIR,
        model_path=Path(
            _env_str("MODEL_PATH", str(BASE_DIR / "ModelKlasifikasiGenerativeAIfix3.keras"))
        ),
        class_indices_path=Path(_env_str("CLASS_INDICES_PATH", str(BASE_DIR / "class_indices.json"))),
        img_width=img_size,
        img_height=img_size,
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 10),
        min_side_px=img_size,
        face_detector=_env_str("FACE_DETECTOR", "auto").lower(),
        face_model_path=_env_str("FACE_MODEL_PATH", ""),
        face_min_confidence=_env_float("FACE_GATE_MIN_CONF", 0.70),
        face_min_coverage=_env_float("FACE_GATE_MIN_COVERAGE", 0.15),
        face_max_coverage=_env_float("FACE_GATE_MAX_COVERAGE", 0.75),
        confidence_high=_env_float("CONF_HIGH", 0.70),
        confidence_medium=_env_float("CONF_MEDIUM", 0.40),
        strict_startup=_env_bool("STRICT_STARTUP", False),
        cors_origins=_env_tuple("CORS_ORIGINS", "*"),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        history_enabled=_env_bool("HISTORY_ENABLED", True),
        database_path=_env_str("DATABASE_PATH", ""),
        rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 60),
        batch_max_files=_env_int("BATCH_MAX_FILES", 10),
    )


def reset_config_cache() -> None:
    """Hanya untuk testing: hapus cache konfigurasi."""
    get_config.cache_clear()
