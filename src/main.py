# src/main.py
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import ceil
from typing import Deque, Optional, Tuple, List

from config import Config
from detector_sign import detect_traffic_sign  # SIGN detector (yield/give-way)
from metrics import Metrics, update_metrics
from tracker_kf import KalmanTracker, KFParams
from video_io import open_video, get_fps, get_frame_size, make_writer
from visualize import draw_roi, draw_detection, draw_track_point

Point = Tuple[float, float]
ROI = Tuple[int, int, int, int]   # (x, y, w, h)
BBOX = Tuple[int, int, int, int]  # (x, y, w, h) GLOBAL coords


@dataclass
class ColorVoteConfig:
    window: int = 12
    majority: float = 0.60
    majority_green: float = 0.70
    margin: int = 2
    max_consecutive_misses: int = 12


class TemporalColorVoter:
    def __init__(self, cfg: ColorVoteConfig) -> None:
        self.cfg = cfg
        self._hist: Deque[str] = deque(maxlen=cfg.window)
        self._stable: Optional[str] = None
        self._miss_streak: int = 0

    @property
    def stable(self) -> Optional[str]:
        return self._stable

    def update(self, observed: Optional[str]) -> Optional[str]:
        if observed is None:
            self._hist.append("NONE")
            self._miss_streak += 1
        else:
            self._hist.append(observed)
            self._miss_streak = 0

        if self._miss_streak >= self.cfg.max_consecutive_misses:
            self._stable = None
            return self._stable

        counts = Counter([c for c in self._hist if c != "NONE"])
        if not counts:
            return self._stable

        winner, winner_votes = counts.most_common(1)[0]
        maj = self.cfg.majority_green if winner == "GREEN" else self.cfg.majority
        needed = ceil(self.cfg.window * maj)

        if winner_votes < needed:
            return self._stable

        if self._stable is not None and winner != self._stable:
            stable_votes = counts.get(self._stable, 0)
            if (winner_votes - stable_votes) < self.cfg.margin:
                return self._stable

        self._stable = winner
        return self._stable


def _dist(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _clamp_roi(roi: ROI, w: int, h: int) -> ROI:
    x, y, rw, rh = roi
    x = max(0, min(x, w - rw))
    y = max(0, min(y, h - rh))
    return (x, y, rw, rh)


def _expand_roi(roi: ROI, expand_px: int, w: int, h: int) -> ROI:
    x, y, rw, rh = roi
    nx = max(0, x - expand_px)
    ny = max(0, y - expand_px)
    nrw = min(rw + 2 * expand_px, w - nx)
    nrh = min(rh + 2 * expand_px, h - ny)
    return (int(nx), int(ny), int(nrw), int(nrh))


def run(cfg: Config) -> None:
    import os
    import cv2

    # --- Counters / debug ---
    total_frames = 0
    det_frames = 0
    raw_misses = 0
    miss_streak = 0
    max_miss_streak = 0

    recovered_on_retry = 0
    rejected_retry = 0
    retry_level_hits = [0, 0, 0]

    # continuity metrics
    track_visible_frames = 0
    color_stable_frames = 0

    debug_dir = "debug_misses"
    os.makedirs(debug_dir, exist_ok=True)

    cap = open_video(cfg.video_path)
    fps = get_fps(cap, cfg.fps_assumed)
    frame_size = get_frame_size(cap)  # (w, h)
    w, h = frame_size

    writer = make_writer(cfg.output_path, fps, frame_size)

    # -----------------------------
    # SIGN DEMO: Start/Stop window
    # -----------------------------
    # IMPORTANT: Because the sign appears briefly, we evaluate only the visibility window.
    # If your Config doesn't have these fields, getattr() safely defaults.
    start_seconds = float(getattr(cfg, "start_seconds", 6.0))  # set to the time sign first appears
    stop_seconds = float(getattr(cfg, "stop_seconds", 12.0))   # set to a time after sign disappears

    start_frame = int(start_seconds * fps)
    stop_frame = int(stop_seconds * fps)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print("Starting at frame:", start_frame, "(seconds:", start_seconds, ")")
    else:
        print("Starting at frame: 0 (seconds: 0.0)")

    # --- Kalman Filter ---
    dt = 1.0 / fps
    params = KFParams(
        meas_std=float(getattr(cfg, "meas_std", 2.0)),
        accel_std=float(getattr(cfg, "accel_std", 3.0)),
        gate_px=float(getattr(cfg, "gate_px", 80.0)),
    )
    kf = KalmanTracker(dt, params)

    # Temporal voting: for sign demo this stabilizes label presence ("GIVE_WAY" vs UNKNOWN)
    voter = TemporalColorVoter(ColorVoteConfig(window=8, majority=0.60, margin=1, max_consecutive_misses=8))

    # Retry settings: expand ROI when detector misses (still within fixed ROI concept)
    retry_expansions: List[int] = list(getattr(cfg, "retry_expansions", [25, 45, 65]))

    metrics = Metrics()
    det_prev: Optional[Point] = None
    kf_prev: Optional[Point] = None

    last_bbox: Optional[BBOX] = None
    last_kf_pt: Optional[Point] = None

    # -------------------------------------------------------
    # KEY FIX: For moving-car sign video, DO NOT template-track ROI.
    # We keep ROI fixed and only run within [start_seconds, stop_seconds].
    # -------------------------------------------------------
    fixed_roi: ROI = cfg.roi

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Stop after visibility window (so metrics make sense)
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if stop_frame > start_frame and frame_idx >= stop_frame:
            break

        total_frames += 1

        # Ensure writer always receives 3-channel BGR frames
        if frame is not None and len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame is not None and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Fixed ROI (no template tracker)
        roi_now: ROI = _clamp_roi(fixed_roi, w, h)

        # KF predict early (optional, helps gating inside KF)
        if kf.initialized:
            _ = kf.predict()

        # -------- First pass sign detection --------
        # Debug only for the first ~5 frames to inspect mask/contours
        debug_now = (total_frames <= 5)
        det = detect_traffic_sign(frame, roi_now, debug=debug_now, debug_prefix=f"debug_sign_{total_frames:03d}")

        det_from_retry = False

        # -------- Retry (expanded ROI) if missing --------
        if det is None:
            for i, ex in enumerate(retry_expansions):
                roi_retry = _expand_roi(roi_now, ex, w, h)
                det2 = detect_traffic_sign(frame, roi_retry)
                if det2 is not None:
                    det = det2
                    roi_now = roi_retry
                    recovered_on_retry += 1
                    retry_level_hits[min(i, 2)] += 1
                    det_from_retry = True
                    break

        det_raw_pt: Optional[Point] = (det.cx, det.cy) if det is not None else None

        # Miss accounting + debug dumps
        if det_raw_pt is None:
            raw_misses += 1
            miss_streak += 1
            max_miss_streak = max(max_miss_streak, miss_streak)

            if (total_frames % 30) == 0:
                dbg = frame.copy()
                draw_roi(dbg, roi_now)
                cv2.imwrite(os.path.join(debug_dir, f"miss_{total_frames:06d}.png"), dbg)
        else:
            det_frames += 1
            miss_streak = 0

        # Initialize KF on first valid detection
        if (not kf.initialized) and det_raw_pt is not None:
            kf.init(det_raw_pt[0], det_raw_pt[1])
            last_kf_pt = det_raw_pt

        # KF update (gating occurs inside KF)
        kf_now: Optional[Point] = None
        if kf.initialized:
            kf_now = kf.update(det_raw_pt)

        # --- Visualization ---
        draw_roi(frame, roi_now)

        if det is not None:
            stable_label = voter.update(det.color)  # detector_sign provides .color = "GIVE_WAY"
        else:
            stable_label = voter.update(None)

        label = stable_label if stable_label is not None else "UNKNOWN"
        if stable_label is not None:
            color_stable_frames += 1

        if det is not None:
            draw_detection(frame, det.bbox, label=f"DETECTION {label}")
            last_bbox = det.bbox

        if kf_now is not None:
            draw_track_point(frame, kf_now[0], kf_now[1], label="KALMAN")
            track_visible_frames += 1
            last_kf_pt = (kf_now[0], kf_now[1])

        writer.write(frame)

        # Metrics: for sign demo, we avoid counting retry detections as detector jitter
        det_for_metrics: Optional[Point]
        if det_raw_pt is not None and (not det_from_retry):
            det_for_metrics = det_raw_pt
        else:
            det_for_metrics = None

        update_metrics(metrics, det_prev, det_for_metrics, kf_prev, kf_now)
        det_prev = det_for_metrics
        kf_prev = kf_now

    cap.release()
    writer.release()

    det_avg_step = metrics.det_jitter_sum / metrics.det_steps if metrics.det_steps else 0.0
    kf_avg_step = metrics.kf_jitter_sum / metrics.kf_steps if metrics.kf_steps else 0.0
    det_rate = (det_frames / total_frames) if total_frames else 0.0
    track_rate = (track_visible_frames / total_frames) if total_frames else 0.0
    stable_rate = (color_stable_frames / total_frames) if total_frames else 0.0

    print("Frames:", total_frames)
    print("Detection frames:", det_frames)
    print("Detection rate:", det_rate)
    print("Raw misses (detector returned None):", raw_misses)
    print("Max consecutive misses:", max_miss_streak)

    print("Recovered on retry (expanded ROI):", recovered_on_retry)
    print("Retry level hits (25/45/65):", retry_level_hits)
    print("Rejected retry detections (too far from pred):", rejected_retry)

    print("TRACK visible frames:", track_visible_frames)
    print("TRACK visible rate:", track_rate)
    print("Stable label frames:", color_stable_frames)
    print("Stable label rate:", stable_rate)

    print("Missed detections:", metrics.missed_detections)
    print("Avg detection step (pixels) [first-pass only]:", det_avg_step)
    print("Avg KF step (pixels):", kf_avg_step)


if __name__ == "__main__":
    cfg = Config(
        video_path="videos/traffic_sign_day_cv.mp4",
        output_path="outputs/annotated_sign.mp4",
        roi=(1443, 500, 97, 63),   # ROI you picked at ~6 seconds
        min_contour_area=20,
        max_contour_area=8000,
        fps_assumed=30.0,
        # IMPORTANT: Your Config dataclass probably doesn't include these.
        # So DON'T pass start_seconds/stop_seconds here.
        # They are handled via getattr() defaults inside run().
    )
    run(cfg)
