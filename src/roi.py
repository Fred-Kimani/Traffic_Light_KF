import numpy as np

def crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]

def roi_to_global(cx_roi: float, cy_roi: float, roi: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, _, _ = roi
    return cx_roi + x, cy_roi + y
