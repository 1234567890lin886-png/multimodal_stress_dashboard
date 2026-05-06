from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class VideoSignals:
    fps: float
    timestamps: np.ndarray
    rgb_series: np.ndarray
    motion_series: np.ndarray
    lighting_series: np.ndarray
    sample_frame_bgr: Optional[np.ndarray]


@dataclass
class ROIExtraction:
    roi_bgr: np.ndarray
    mask: np.ndarray
    valid: bool
    bbox: tuple[int, int, int, int]
    full_mask: Optional[np.ndarray] = None


def _center_face_like_roi(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, x2 = int(width * 0.30), int(width * 0.70)
    y1, y2 = int(height * 0.22), int(height * 0.68)
    return frame[y1:y2, x1:x2]


class FaceMeshROIExtractor:
    """FaceMesh cheek/forehead ROI extraction adapted from the FYP pipeline."""

    CHEEK_LEFT = [234, 93, 132, 58, 172, 136, 150]
    CHEEK_RIGHT = [454, 323, 361, 288, 397, 365, 379]
    FOREHEAD = [10, 338, 297, 332, 284, 251, 389]

    def __init__(self, target_size: tuple[int, int] = (320, 240), use_forehead: bool = False):
        self.target_size = target_size
        self.use_forehead = use_forehead
        self._face_mesh = None
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except ImportError:
            self._face_mesh = None

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()

    def extract(self, frame_bgr: np.ndarray) -> ROIExtraction:
        if self._face_mesh is None:
            roi, mask, bbox = _center_roi_with_mask(frame_bgr)
            return ROIExtraction(roi, mask, False, bbox)

        original_h, original_w = frame_bgr.shape[:2]
        resized = cv2.resize(frame_bgr, self.target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            roi, mask, bbox = _center_roi_with_mask(frame_bgr)
            return ROIExtraction(roi, mask, False, bbox)

        landmarks = results.multi_face_landmarks[0].landmark
        width, height = self.target_size
        indices = self.CHEEK_LEFT + self.CHEEK_RIGHT
        if self.use_forehead:
            indices += self.FOREHEAD
        points = np.array(
            [[int(landmarks[idx].x * width), int(landmarks[idx].y * height)] for idx in set(indices)],
            dtype=np.int32,
        )
        if len(points) < 3:
            roi, mask, bbox = _center_roi_with_mask(frame_bgr)
            return ROIExtraction(roi, mask, False, bbox)

        hull = cv2.convexHull(points)
        mask_resized = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask_resized, hull, 255)
        mask = cv2.resize(mask_resized, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
        roi_bgr = frame_bgr[y : y + h, x : x + w]
        roi_mask = mask[y : y + h, x : x + w]
        skin_mask = self._skin_mask(roi_bgr, roi_mask)
        if skin_mask is not None:
            roi_mask = skin_mask
        return ROIExtraction(roi_bgr, roi_mask, True, (x, y, x + w, y + h), mask)

    @staticmethod
    def _skin_mask(roi_bgr: np.ndarray, base_mask: np.ndarray) -> Optional[np.ndarray]:
        if roi_bgr.size == 0 or np.count_nonzero(base_mask) < 100:
            return None
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb_mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, (0, 35, 35), (25, 255, 255))
        skin = cv2.bitwise_and(ycrcb_mask, hsv_mask)
        skin = cv2.bitwise_and(skin, base_mask)
        skin = cv2.medianBlur(skin, 5)
        if np.count_nonzero(skin) < max(20, int(0.03 * skin.size)):
            return None
        return skin


def _masked_mean_rgb(roi_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    if roi_bgr.size == 0 or mask.size == 0 or np.count_nonzero(mask) == 0:
        return None
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    return np.array(
        [
            np.mean(roi_rgb[:, :, 0][mask > 0]),
            np.mean(roi_rgb[:, :, 1][mask > 0]),
            np.mean(roi_rgb[:, :, 2][mask > 0]),
        ],
        dtype=float,
    )


def _center_roi_with_mask(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    x1, x2 = int(width * 0.30), int(width * 0.70)
    y1, y2 = int(height * 0.22), int(height * 0.68)
    roi = frame[y1:y2, x1:x2]
    mask = np.full(roi.shape[:2], 255, dtype=np.uint8)
    return roi, mask, (x1, y1, x2, y2)


def _roi_from_cached_mask(frame: np.ndarray, bbox: tuple[int, int, int, int], full_mask: np.ndarray) -> ROIExtraction:
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    x1, x2 = int(np.clip(x1, 0, width)), int(np.clip(x2, 0, width))
    y1, y2 = int(np.clip(y1, 0, height)), int(np.clip(y2, 0, height))
    if x2 <= x1 or y2 <= y1 or full_mask.shape[:2] != frame.shape[:2]:
        roi, mask, fallback_bbox = _center_roi_with_mask(frame)
        return ROIExtraction(roi, mask, False, fallback_bbox)
    return ROIExtraction(
        frame[y1:y2, x1:x2],
        full_mask[y1:y2, x1:x2],
        True,
        (x1, y1, x2, y2),
        full_mask,
    )


def extract_video_signals(
    video_path: str | Path,
    max_frames: int = 900,
    stride: int = 1,
    roi_refresh_frames: int = 1,
) -> VideoSignals:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    effective_fps = fps / max(stride, 1)
    rgb_values: list[np.ndarray] = []
    lighting: list[float] = []
    motion: list[float] = []
    timestamps: list[float] = []
    sample_frame: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0
    kept = 0
    roi_extractor = FaceMeshROIExtractor()
    cached_bbox: Optional[tuple[int, int, int, int]] = None
    cached_mask: Optional[np.ndarray] = None

    try:
        while kept < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            if cached_bbox is not None and cached_mask is not None and kept % max(1, roi_refresh_frames) != 0:
                extracted = _roi_from_cached_mask(frame, cached_bbox, cached_mask)
            else:
                extracted = roi_extractor.extract(frame)
                if extracted.valid and extracted.full_mask is not None:
                    cached_bbox = extracted.bbox
                    cached_mask = extracted.full_mask
            mean_rgb = _masked_mean_rgb(extracted.roi_bgr, extracted.mask)
            if mean_rgb is None:
                frame_idx += 1
                continue
            rgb_values.append(mean_rgb)
            gray = cv2.cvtColor(extracted.roi_bgr, cv2.COLOR_BGR2GRAY)
            if extracted.mask.size == gray.shape[:2] and np.count_nonzero(extracted.mask) > 0:
                gray_values = gray[extracted.mask > 0]
                lighting.append(float(np.std(gray_values) / 255.0))
            else:
                lighting.append(float(np.std(gray) / 255.0))
            if prev_gray is None or prev_gray.shape != gray.shape:
                motion.append(0.0)
            else:
                diff = cv2.absdiff(gray, prev_gray)
                if extracted.mask.size == diff.shape[:2] and np.count_nonzero(extracted.mask) > 0:
                    motion.append(float(np.mean(diff[extracted.mask > 0]) / 255.0))
                else:
                    motion.append(float(np.mean(diff) / 255.0))
            prev_gray = gray
            timestamps.append(kept / effective_fps)
            if sample_frame is None:
                sample_frame = frame.copy()
            kept += 1
            frame_idx += 1
    finally:
        roi_extractor.close()
        cap.release()
    if not rgb_values:
        raise ValueError("No frames could be processed from the video.")

    return VideoSignals(
        fps=effective_fps,
        timestamps=np.asarray(timestamps, dtype=float),
        rgb_series=np.asarray(rgb_values, dtype=float),
        motion_series=np.asarray(motion, dtype=float),
        lighting_series=np.asarray(lighting, dtype=float),
        sample_frame_bgr=sample_frame,
    )
