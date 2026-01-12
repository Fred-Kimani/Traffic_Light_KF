from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

ROI = Tuple[int, int, int, int]
BBOX = Tuple[int, int, int, int]


@dataclass
class SignDetection:
    cx: float
    cy: float
    bbox: BBOX
    label: str = "GIVE_WAY"

    @property
    def color(self) -> str:
        return self.label


def detect_traffic_sign(frame: np.ndarray, roi: ROI, debug: bool = False, debug_prefix: str = "sign") -> Optional[SignDetection]:
    """
    Robust-ish give-way sign detector for small/noisy moving-car clips.

    Key changes vs strict version:
    - looser red thresholds (S and V lowered)
    - accept approx vertices 3..6 (triangle often becomes 4-6 under noise)
    - lower min area
    - choose best candidate by score
    - optional debug dumps (crop/mask/contours)
    """
    x, y, w, h = roi
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # LOOSER thresholds (web video + compression + glare)
    # Lower S and V so red border is still captured.
    lower_red1 = np.array([0, 40, 40], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 40, 40], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        dbg = crop.copy()
        cv2.drawContours(dbg, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(f"{debug_prefix}_crop.png", crop)
        cv2.imwrite(f"{debug_prefix}_mask.png", mask)
        cv2.imwrite(f"{debug_prefix}_contours.png", dbg)

    if not contours:
        return None

    roi_area = float(w * h)

    best: Optional[SignDetection] = None
    best_score = -1.0

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        # LOWER min area (small sign)
        if area < 40.0:
            continue
        if area > 0.90 * roi_area:
            continue

        peri = float(cv2.arcLength(cnt, True))
        if peri <= 1e-6:
            continue

        # Looser approximation: helps noisy borders
        eps = 0.06 * peri
        approx = cv2.approxPolyDP(cnt, eps, True)

        # In practice, a triangle becomes 3-6 vertices under noise.
        if not (3 <= len(approx) <= 6):
            continue

        bx, by, bw, bh = cv2.boundingRect(approx)
        if bw <= 0 or bh <= 0:
            continue

        aspect = float(bw) / float(bh)
        if aspect < 0.5 or aspect > 1.9:
            continue

        # Prefer larger + more compact shapes
        # (simple, student-level scoring)
        score = area - 0.1 * peri

        if score > best_score:
            cx = x + bx + bw / 2.0
            cy = y + by + bh / 2.0
            best_score = score
            best = SignDetection(
                cx=cx,
                cy=cy,
                bbox=(x + bx, y + by, bw, bh),
                label="GIVE_WAY",
            )

    return best
