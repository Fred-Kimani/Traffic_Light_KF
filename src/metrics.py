from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

@dataclass
class Metrics:
    missed_detections: int = 0
    det_jitter_sum: float = 0.0
    kf_jitter_sum: float = 0.0
    det_steps: int = 0
    kf_steps: int = 0

def step_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def update_metrics(m: Metrics,
                   det_prev: Optional[Tuple[float, float]],
                   det_now: Optional[Tuple[float, float]],
                   kf_prev: Optional[Tuple[float, float]],
                   kf_now: Optional[Tuple[float, float]]) -> None:
    if det_now is None:
        m.missed_detections += 1

    if det_prev is not None and det_now is not None:
        m.det_jitter_sum += step_distance(det_prev, det_now)
        m.det_steps += 1

    if kf_prev is not None and kf_now is not None:
        m.kf_jitter_sum += step_distance(kf_prev, kf_now)
        m.kf_steps += 1
