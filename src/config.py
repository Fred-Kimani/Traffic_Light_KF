from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Config:
    video_path: str
    output_path: str

    # ROI in (x, y, w, h) pixel coords
    # Start with a guess; you can adjust after first run.
    roi: Tuple[int, int, int, int]

    # Detector settings (HSV thresholds etc. will go here later)
    min_contour_area: int = 40
    max_contour_area: int = 5000

    # Kalman settings placeholder (we’ll fill later)
    fps_assumed: float = 30.0
