import cv2
import numpy as np

def draw_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> None:
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

def draw_detection(frame: np.ndarray, bbox: tuple[int, int, int, int], label: str = "DETECTION") -> None:
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def draw_track_point(frame: np.ndarray, cx: float, cy: float, label: str = "KALMAN") -> None:
    cv2.circle(frame, (int(cx), int(cy)), 6, (255, 0, 0), -1)
    cv2.putText(frame, label, (int(cx)+8, int(cy)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
