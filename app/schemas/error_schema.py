"""Skema respons error seragam untuk semua endpoint (§6.3 / §6.4)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(description="Kode error, mis. NO_FACE_DETECTED")
    message: str = Field(description="Pesan ramah pengguna")
    requirement_ref: str | None = Field(
        default=None, description="Tautan ke butir ketentuan §2.2 (bila ada)"
    )


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody
    request_id: str


def error_payload(code: str, message: str, requirement_ref: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if requirement_ref:
        body["requirement_ref"] = requirement_ref
    return {"success": False, "error": body}
