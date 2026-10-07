"""Controller riwayat — baca/hapus riwayat prediksi & alasan penolakan (Fase 8)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.deps import get_history_service
from app.errors import AppError
from app.schemas.history_schema import HistoryDeleted, HistoryPage
from app.services.history_service import HistoryService

router = APIRouter(prefix="/api/v1/history", tags=["history"])


@router.get(
    "",
    response_model=HistoryPage,
    summary="Riwayat prediksi & penolakan (terbaru dahulu)",
)
def list_history(
    page: int = Query(1, ge=1, description="Nomor halaman (mulai 1)"),
    size: int = Query(20, ge=1, le=100, description="Jumlah entri per halaman"),
    service: HistoryService = Depends(get_history_service),
) -> HistoryPage:
    items, total = service.list(page=page, size=size)
    return HistoryPage(items=items, page=page, size=size, total=total)


@router.delete(
    "/{item_id}",
    response_model=HistoryDeleted,
    summary="Hapus satu entri riwayat",
)
def delete_history(
    item_id: int,
    service: HistoryService = Depends(get_history_service),
) -> HistoryDeleted:
    if not service.delete(item_id):
        raise AppError(
            "Entri riwayat tidak ditemukan.",
            code="HISTORY_NOT_FOUND",
            status_code=404,
        )
    return HistoryDeleted(deleted=True, id=item_id)
