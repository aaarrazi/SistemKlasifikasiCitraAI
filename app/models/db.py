"""Model data — riwayat prediksi & alasan penolakan (SQLite, stdlib `sqlite3`).

Tidak butuh dependensi tambahan. Satu koneksi dibagikan untuk semua request
(middleware jalan pada event loop yang sama), jadi setiap operasi dikunci.

Tabel `predictions` menyimpan DUA jenis baris:
  - prediksi sukses  → `label`, `confidence`, `status` terisi, `reject_code` NULL
  - penolakan        → `reject_code` terisi (mis. MULTIPLE_FACES), `label` NULL
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.errors import AppError

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id    TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    filename      TEXT,
    label         TEXT,
    confidence    REAL,
    status        TEXT,
    face_status   TEXT,
    reject_code   TEXT,
    processing_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions (created_at DESC);
"""


class HistoryUnavailableError(AppError):
    code = "HISTORY_UNAVAILABLE"
    status_code = 503


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Pembungkus tipis koneksi SQLite (thread-safe via lock)."""

    def __init__(self, path: str) -> None:
        self._path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(path, check_same_thread=False)
        except sqlite3.Error as exc:
            raise HistoryUnavailableError(
                "Database riwayat tidak bisa dibuka.", detail=str(exc)
            ) from exc
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # --- tulis -------------------------------------------------------------

    def record(
        self,
        *,
        request_id: str,
        filename: str | None,
        label: str | None = None,
        confidence: float | None = None,
        status: str | None = None,
        face_status: str | None = None,
        reject_code: str | None = None,
        processing_ms: int | None = None,
    ) -> int:
        try:
            with self._lock:
                cur = self._conn.execute(
                    """
                    INSERT INTO predictions
                        (request_id, created_at, filename, label, confidence,
                         status, face_status, reject_code, processing_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        _now(),
                        filename,
                        label,
                        confidence,
                        status,
                        face_status,
                        reject_code,
                        processing_ms,
                    ),
                )
                self._conn.commit()
                return int(cur.lastrowid)
        except sqlite3.Error as exc:
            raise HistoryUnavailableError(
                "Riwayat gagal disimpan.", detail=str(exc)
            ) from exc

    # --- baca --------------------------------------------------------------

    def list(self, page: int = 1, size: int = 20) -> tuple[list[dict[str, Any]], int]:
        page = max(1, page)
        size = max(1, min(size, 100))
        offset = (page - 1) * size
        with self._lock:
            total = int(self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
            rows = self._conn.execute(
                """
                SELECT id, request_id, created_at, filename, label, confidence,
                       status, face_status, reject_code, processing_ms
                FROM predictions
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (size, offset),
            ).fetchall()
        items = [dict(r) for r in rows]
        return items, total

    def get(self, row_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (row_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete(self, row_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM predictions WHERE id = ?", (row_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
