"""Face Gate — validasi ketentuan unggahan SEBELUM inferensi (§7.2).

Aturan (arsitektur.md §7.2):
    F1  sisi terpendek >= min_side_px              -> IMAGE_TOO_LOW_RES
    F2  0.5 <= W/H <= 2.0                          -> INVALID_FRAMING
    F3  >= 1 wajah dgn confidence >= ambang        -> NO_FACE_DETECTED
    F4  tepat 1 wajah                              -> MULTIPLE_FACES
    F5  0.15 <= tinggi_wajah/tinggi_gambar <= 0.75 -> FACE_TOO_SMALL / FACE_TOO_LARGE
    F6  pusat wajah di dalam pita aman             -> INVALID_FRAMING

Sifat: FAIL-CLOSED. Bila detektor tidak tersedia/krash, request DITOLAK dengan
503 FACE_CHECK_UNAVAILABLE — tidak pernah dilanjutkan tanpa pemeriksaan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from app.config import Config
from app.errors import FaceCheckUnavailableError, InvalidImageError
from app.models.ml.labels import requirement_message, requirement_ref
from app.models.ml.preprocessor import open_rgb

logger = logging.getLogger("app.face_gate")


# --- tipe data ---------------------------------------------------------------


@dataclass(frozen=True)
class Face:
    """Bounding box wajah dalam koordinat piksel (kiri, atas, lebar, tinggi)."""

    left: float
    top: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)


@dataclass(frozen=True)
class FaceGateResult:
    passed: bool
    code: str | None = None
    message: str | None = None
    requirement_ref: str | None = None
    faces_detected: int = 0
    face_coverage: float | None = None
    detector: str = "unknown"

    def as_dict(self) -> dict:
        return {
            "status": "passed" if self.passed else "rejected",
            "faces_detected": self.faces_detected,
            "face_coverage": round(self.face_coverage, 4) if self.face_coverage else None,
            "detector": self.detector,
            "code": self.code,
        }


# --- antarmuka detektor ------------------------------------------------------


class FaceDetector(Protocol):
    name: str

    def detect(self, img: Image.Image) -> list[Face]:  # pragma: no cover - protocol
        ...


def _reject(code: str, detector: str, faces: int = 0, coverage: float | None = None) -> FaceGateResult:
    return FaceGateResult(
        passed=False,
        code=code,
        message=requirement_message(code),
        requirement_ref=requirement_ref(code),
        faces_detected=faces,
        face_coverage=coverage,
        detector=detector,
    )


# --- backend detektor --------------------------------------------------------


class MediaPipeDetector:
    """Detektor default: MediaPipe **Tasks API** + model BlazeFace (short range).

    Catatan (penting!): mediapipe >= 1.0 TIDAK lagi menyediakan API lama
    `mp.solutions.face_detection`, sehingga detektor membutuhkan file model
    yang diunduh sekali lewat `python scripts/download_face_model.py`
    (default: `models/blaze_face_short_range.tflite`, ± 230 KB).
    """

    name = "mediapipe_face_detection"

    def __init__(self, model_path: Path, min_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore
        except ImportError as exc:
            raise FaceCheckUnavailableError(
                "Detektor wajah (mediapipe) belum terpasang — jalankan Fase 0: "
                "pip install -r requirements.txt",
                detail=str(exc),
            ) from exc

        if not Path(model_path).exists():
            raise FaceCheckUnavailableError(
                "Model detektor wajah belum ada. Jalankan sekali: "
                "python scripts/download_face_model.py",
                detail=f"path={model_path}",
            )

        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        try:
            self._detector = vision.FaceDetector.create_from_options(options)
        except Exception as exc:  # noqa: BLE001
            raise FaceCheckUnavailableError(
                "Model detektor wajah gagal dimuat (file rusak/ketidakcocokan versi).",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        self._mp = mp

    def detect(self, img: Image.Image) -> list[Face]:
        import numpy as np

        rgb = np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        detections = getattr(result, "detections", None) or []

        faces: list[Face] = []
        for det in detections:
            box = det.bounding_box  # sudah dalam piksel citra masukan
            score = det.categories[0].score if getattr(det, "categories", None) else 1.0
            faces.append(
                Face(
                    left=float(box.origin_x),
                    top=float(box.origin_y),
                    width=float(box.width),
                    height=float(box.height),
                    confidence=float(score),
                )
            )
        return faces


class OpenCVDNNDetector:
    """Alternatif: OpenCV DNN (SSD ResNet-10). Butuh file model sendiri."""

    name = "opencv_dnn"

    PROTOTXT_ENV = "OPENCV_DNN_PROTOTXT"
    MODEL_ENV = "OPENCV_DNN_MODEL"

    def __init__(self, prototxt: str, caffemodel: str, min_confidence: float = 0.7) -> None:
        import os

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise FaceCheckUnavailableError(
                "opencv-python belum terpasang untuk detektor OpenCV DNN.",
                detail=str(exc),
            ) from exc
        if not (prototxt and caffemodel):
            raise FaceCheckUnavailableError(
                "Path prototxt/caffemodel OpenCV DNN belum diset "
                f"({self.PROTOTXT_ENV}, {self.MODEL_ENV}).",
            )
        self._cv2 = cv2
        self._min_confidence = min_confidence
        self._net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    def detect(self, img: Image.Image) -> list[Face]:
        import numpy as np

        rgb = img.convert("RGB")
        width, height = rgb.size
        blob = self._cv2.dnn.blobFromImage(
            np.asarray(rgb), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        faces: list[Face] = []
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score < self._min_confidence:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            left, top, right, bottom = box.astype(int)
            faces.append(
                Face(
                    left=float(left),
                    top=float(top),
                    width=float(max(0, right - left)),
                    height=float(max(0, bottom - top)),
                    confidence=score,
                )
            )
        return faces


def create_detector(cfg: Config) -> FaceDetector:
    """Pilih backend detektor sesuai `FACE_DETECTOR`. Selalu gagal-ke-tertutup."""
    mode = (cfg.face_detector or "auto").lower()

    if mode in {"off", "none", "false"}:
        raise FaceCheckUnavailableError(
            "Face Gate dinonaktifkan (FACE_DETECTOR=off). Sistem menolak semua "
            "prediksi karena ketentuan wajib diperiksa.",
        )

    if mode == "mediapipe":
        return MediaPipeDetector(
            model_path=cfg.face_model_path_resolved,
            min_confidence=cfg.face_min_confidence,
        )

    if mode == "opencv_dnn":
        import os

        return OpenCVDNNDetector(
            prototxt=os.environ.get(OpenCVDNNDetector.PROTOTXT_ENV, ""),
            caffemodel=os.environ.get(OpenCVDNNDetector.MODEL_ENV, ""),
            min_confidence=max(cfg.face_min_confidence, 0.7),
        )

    # auto: coba backend terbaik yang tersedia
    reasons: list[str] = []
    try:
        return MediaPipeDetector(
            model_path=cfg.face_model_path_resolved,
            min_confidence=cfg.face_min_confidence,
        )
    except FaceCheckUnavailableError as exc:
        reasons.append(exc.detail or exc.message)

    try:
        import os

        return OpenCVDNNDetector(
            prototxt=os.environ.get(OpenCVDNNDetector.PROTOTXT_ENV, ""),
            caffemodel=os.environ.get(OpenCVDNNDetector.MODEL_ENV, ""),
            min_confidence=max(cfg.face_min_confidence, 0.7),
        )
    except FaceCheckUnavailableError as exc:
        reasons.append(exc.detail or exc.message)

    raise FaceCheckUnavailableError(
        "Tidak ada detektor wajah yang tersedia. Pasang mediapipe "
        "(lihat requirements.txt) — sistem menolak semua prediksi selama gate "
        "tidak berjalan (fail-closed).",
        detail="; ".join(reasons),
    )


# --- gerbang utama -----------------------------------------------------------


class FaceGate:
    def __init__(self, detector: FaceDetector, cfg: Config) -> None:
        self._detector = detector
        self._cfg = cfg

    @property
    def detector_name(self) -> str:
        return getattr(self._detector, "name", "unknown")

    def check(self, raw: bytes) -> FaceGateResult:
        """Periksa ketentuan §2.1 pada file gambar. Tidak pernah melewatkan request."""
        cfg = self._cfg

        try:
            img = open_rgb(raw)
        except InvalidImageError as exc:
            return _reject("INVALID_IMAGE", self.detector_name)

        width, height = img.size

        # F1 — resolusi minimal
        if min(width, height) < cfg.min_side_px:
            return _reject("IMAGE_TOO_LOW_RES", self.detector_name)

        # F2 — rasio aspek
        aspect = width / height
        if not (cfg.aspect_min <= aspect <= cfg.aspect_max):
            return _reject("INVALID_FRAMING", self.detector_name)

        # Deteksi wajah — gagal = tolak (fail-closed)
        try:
            faces = self._detector.detect(img)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Detektor wajah gagal")
            raise FaceCheckUnavailableError(
                "Layanan pemeriksaan gambar gagal dijalankan; permintaan ditolak "
                "agar ketentuan tetap terpenuhi.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        # F3 — ada wajah
        confident = [f for f in faces if f.confidence >= cfg.face_min_confidence]
        if not confident:
            return _reject("NO_FACE_DETECTED", self.detector_name, faces=len(faces))

        # F4 — tepat satu wajah
        if len(confident) > 1:
            return _reject(
                "MULTIPLE_FACES", self.detector_name, faces=len(confident)
            )

        face = max(confident, key=lambda f: f.confidence * f.width * f.height)
        coverage = float(face.height) / float(height)

        # F5 — cakupan wajah (posisi wajah–pundak)
        if coverage < cfg.face_min_coverage:
            return _reject(
                "FACE_TOO_SMALL", self.detector_name, faces=1, coverage=coverage
            )
        if coverage > cfg.face_max_coverage:
            return _reject(
                "FACE_TOO_LARGE", self.detector_name, faces=1, coverage=coverage
            )

        # F6 — posisi wajah dalam bingkai
        cx, cy = face.center
        nx, ny = cx / width, cy / height
        in_x = cfg.center_x_min <= nx <= cfg.center_x_max
        in_y = cfg.center_y_min <= ny <= cfg.center_y_max
        if not (in_x and in_y):
            return _reject("INVALID_FRAMING", self.detector_name, faces=1, coverage=coverage)

        return FaceGateResult(
            passed=True,
            faces_detected=1,
            face_coverage=coverage,
            detector=self.detector_name,
        )
