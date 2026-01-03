# src/tracker_kf.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

Point = Tuple[float, float]


@dataclass
class KFParams:
    # Measurement noise std in pixels (detector jitter)
    meas_std: float = 2.0          # night start; use ~0.5 for day
    # Acceleration noise std in pixels/sec^2 (shake unpredictability)
    accel_std: float = 3.0         # tune: higher = more responsive
    # Innovation gate in pixels (reject huge jumps)
    gate_px: float = 80.0


class KalmanTracker:
    """
    Constant-velocity Kalman Filter in 2D image space.
    State: [cx, cy, vx, vy]^T
    Measurement: [cx, cy]^T
    """

    def __init__(self, dt: float, params: Optional[KFParams] = None) -> None:
        self.dt = float(dt)
        self.params = params or KFParams()

        self.initialized: bool = False

        # state mean (4x1) and covariance (4x4)
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1e3  # large uncertainty initially

        # Matrices that depend on dt
        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.I = np.eye(4, dtype=np.float64)

        self.Q = self._make_Q(self.params.accel_std)
        self.R = self._make_R(self.params.meas_std)

    def _make_R(self, meas_std: float) -> np.ndarray:
        r = float(meas_std) ** 2
        return np.array([[r, 0.0], [0.0, r]], dtype=np.float64)

    def _make_Q(self, accel_std: float) -> np.ndarray:
        """
        Random-acceleration model -> process noise for CV state.
        accel_std is std of acceleration noise (pixels/sec^2).
        """
        q = float(accel_std) ** 2
        dt = self.dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        return q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

    def set_meas_std(self, meas_std: float) -> None:
        self.params.meas_std = float(meas_std)
        self.R = self._make_R(self.params.meas_std)

    def set_accel_std(self, accel_std: float) -> None:
        self.params.accel_std = float(accel_std)
        self.Q = self._make_Q(self.params.accel_std)

    def init(self, cx: float, cy: float) -> None:
        self.x[:] = 0.0
        self.x[0, 0] = float(cx)
        self.x[1, 0] = float(cy)
        self.x[2, 0] = 0.0
        self.x[3, 0] = 0.0
        self.P = np.eye(4, dtype=np.float64) * 10.0  # moderate certainty after init
        self.initialized = True

    def predict(self) -> Point:
        # x' = F x
        self.x = self.F @ self.x
        # P' = F P F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def update(self, meas: Optional[Point]) -> Point:
        """
        If meas is None: no update, return predicted state.
        If meas exists: perform gated KF update.
        """
        if meas is None:
            return (float(self.x[0, 0]), float(self.x[1, 0]))

        z = np.array([[float(meas[0])], [float(meas[1])]], dtype=np.float64)

        # innovation: y = z - H x'
        y = z - (self.H @ self.x)

        # innovation covariance: S = H P' H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # simple pixel gate (student-friendly): reject if innovation too large
        gate = float(self.params.gate_px)
        if (y[0, 0] ** 2 + y[1, 0] ** 2) ** 0.5 > gate:
            return (float(self.x[0, 0]), float(self.x[1, 0]))

        # Kalman gain: K = P' H^T S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # state update: x = x' + K y
        self.x = self.x + (K @ y)

        # covariance update: P = (I - K H) P'
        self.P = (self.I - (K @ self.H)) @ self.P

        return (float(self.x[0, 0]), float(self.x[1, 0]))
