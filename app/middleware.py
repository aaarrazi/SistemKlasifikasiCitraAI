"""Middleware & proteksi tingkat HTTP (rate limit in-memory, tanpa dependensi)."""

from __future__ import annotations

import threading
import time
from collections import deque

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.schemas.error_schema import error_payload


def client_ip(request: Request) -> str:
    """IP klien — utamakan `X-Forwarded-For` (hop pertama) bila di belakang proxy."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


class RateLimiter:
    """Sliding window per IP. `per_minute <= 0` berarti fitur nonaktif."""

    def __init__(self, per_minute: int, window_seconds: float = 60.0) -> None:
        self.per_minute = int(per_minute)
        self.window_seconds = float(window_seconds)
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self.per_minute > 0

    def check(self, key: str) -> tuple[bool, int]:
        """(diizinkan, retry_after_detik)."""
        if not self.enabled:
            return True, 0

        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= self.per_minute:
                retry_after = int(bucket[0] + self.window_seconds - now) + 1
                return False, max(1, retry_after)

            bucket.append(now)
            self._maybe_sweep(now, cutoff)
            return True, 0

    def _maybe_sweep(self, now: float, cutoff: float) -> None:
        """Bersihkan IP yang sudah basi agar memori tidak membengkak."""
        if now - self._last_sweep < self.window_seconds:
            return
        self._last_sweep = now
        for key in [k for k, b in self._hits.items() if not b or b[-1] <= cutoff]:
            self._hits.pop(key, None)


async def rate_limit_guard(request: Request, limiter: RateLimiter) -> JSONResponse | None:
    """Dipanggil dari middleware. Mengembalikan respons 429 bila melebihi batas."""
    if not limiter.enabled:
        return None

    allowed, retry_after = limiter.check(f"{client_ip(request)}:{request.url.path}")
    if allowed:
        return None

    return JSONResponse(
        status_code=429,
        content={
            **error_payload(
                "RATE_LIMITED",
                f"Terlalu banyak permintaan. Coba lagi dalam {retry_after} detik.",
            ),
            "request_id": getattr(request.state, "request_id", "-"),
        },
        headers={"Retry-After": str(retry_after)},
    )
