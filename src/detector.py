from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    cx: float
    cy: float
    bbox: tuple[int, int, int, int]  # (x, y, w, h) GLOBAL coords
    color: str  # "RED" | "YELLOW" | "GREEN"


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _crop_x_band(mask: np.ndarray, x_center: int, half_width: int) -> np.ndarray:
    h, w = mask.shape[:2]
    x0 = _clip(x_center - half_width, 0, w)
    x1 = _clip(x_center + half_width, 0, w)
    out = np.zeros_like(mask)
    out[:, x0:x1] = mask[:, x0:x1]
    return out


def _find_best_bbox(mask: np.ndarray, min_area: int, max_area: int) -> Optional[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1e18

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Penalize extreme aspect ratios (glare streaks etc.)
        aspect = w / float(h) if h > 0 else 999.0
        aspect_penalty = abs(1.0 - aspect)

        score = area - 50.0 * aspect_penalty
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    return best


def detect_traffic_light(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    min_area: int,
    max_area: int
) -> Optional[Detection]:
    x0, y0, w0, h0 = roi
    roi_img = frame[y0:y0 + h0, x0:x0 + w0]
    if roi_img.size == 0:
        return None

    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    # Thresholds
    red1_lo, red1_hi = (0, 120, 120), (10, 255, 255)
    red2_lo, red2_hi = (170, 120, 120), (180, 255, 255)

    # Make yellow strict (prevents red->yellow confusion)
    yellow_lo, yellow_hi = (20, 180, 180), (35, 255, 255)

    # Green (we’ll tune after you test)
    green_lo, green_hi = (40, 60, 60), (90, 255, 255)

    # Split ROI into vertical bands (traffic light geometry)
    red_y0, red_y1 = 0, int(0.45 * h0)
    yel_y0, yel_y1 = int(0.30 * h0), int(0.75 * h0)

    # IMPORTANT: start green lower to reduce false greens
    grn_y0, grn_y1 = int(0.65 * h0), h0

    def band(hsv_img: np.ndarray, y_start: int, y_end: int) -> np.ndarray:
        return hsv_img[y_start:y_end, :]

    # --- RED band ---
    hsv_red = band(hsv, red_y0, red_y1)
    red = cv2.inRange(hsv_red, np.array(red1_lo, np.uint8), np.array(red1_hi, np.uint8))
    red2 = cv2.inRange(hsv_red, np.array(red2_lo, np.uint8), np.array(red2_hi, np.uint8))
    red = cv2.bitwise_or(red, red2)
    red = _clean_mask(red)
    red_bbox = _find_best_bbox(red, min_area, max_area)

    # --- YELLOW band ---
    hsv_yel = band(hsv, yel_y0, yel_y1)
    yellow = cv2.inRange(hsv_yel, np.array(yellow_lo, np.uint8), np.array(yellow_hi, np.uint8))
    yellow = _clean_mask(yellow)
    yel_bbox = _find_best_bbox(yellow, min_area, max_area)

    # --- GREEN band (optionally constrained by RED x-center) ---
    hsv_grn = band(hsv, grn_y0, grn_y1)
    green = cv2.inRange(hsv_grn, np.array(green_lo, np.uint8), np.array(green_hi, np.uint8))
    green = _clean_mask(green)

    if red_bbox is not None:
        rx, ry, rw, rh = red_bbox
        red_cx = int(rx + rw / 2)
        green = _crop_x_band(green, x_center=red_cx, half_width=35)

    grn_bbox = _find_best_bbox(green, min_area, max_area)

    # Convert band-local bboxes to full ROI coords
    candidates = []

    if red_bbox is not None:
        rx, ry, rw, rh = red_bbox
        candidates.append(("RED", (rx, ry + red_y0, rw, rh)))

    if yel_bbox is not None:
        rx, ry, rw, rh = yel_bbox
        candidates.append(("YELLOW", (rx, ry + yel_y0, rw, rh)))

    if grn_bbox is not None:
        rx, ry, rw, rh = grn_bbox
        candidates.append(("GREEN", (rx, ry + grn_y0, rw, rh)))

    if not candidates:
        return None

    # Choose biggest-area candidate (stable)
    best_color, best_bbox = max(candidates, key=lambda c: c[1][2] * c[1][3])

    rx, ry, rw, rh = best_bbox
    cx = x0 + rx + rw / 2.0
    cy = y0 + ry + rh / 2.0
    bbox_global = (x0 + rx, y0 + ry, rw, rh)

    return Detection(cx=cx, cy=cy, bbox=bbox_global, color=best_color)
