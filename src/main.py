# src/main.py
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import ceil
from typing import Deque, Optional, Tuple, List

from config import Config
from detector import detect_traffic_light
from metrics import Metrics, update_metrics
from roi_tracker import RoiTemplateTracker
from tracker_kf import KalmanTracker, KFParams
from video_io import open_video, get_fps, get_frame_size, make_writer
from visualize import draw_roi, draw_detection, draw_track_point

Point = Tuple[float, float]
ROI = Tuple[int, int, int, int]  # (x, y, w, h)
BBOX = Tuple[int, int, int, int] # (x, y, w, h) global coords


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
            # we could keep the last label indefinitely; but UNKNOWN is safer academically
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


def _center_roi_on_point(roi: ROI, pt: Point, w: int, h: int) -> ROI:
    _, _, rw, rh = roi
    cx, cy = pt
    nx = int(round(cx - rw / 2))
    ny = int(round(cy - rh / 2))
    return _clamp_roi((nx, ny, rw, rh), w, h)


def _shift_bbox(bbox: BBOX, dx: float, dy: float, w: int, h: int) -> BBOX:
    x, y, bw, bh = bbox
    nx = int(round(x + dx))
    ny = int(round(y + dy))
    nx = max(0, min(nx, w - bw))
    ny = max(0, min(ny, h - bh))
    return (nx, ny, bw, bh)


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

    # continuity metrics (what you *actually* want to show/defend)
    track_visible_frames = 0         # frames where we show a KF-based track output
    color_stable_frames = 0          # frames where stable color is known (not UNKNOWN)

    debug_dir = "debug_misses"
    os.makedirs(debug_dir, exist_ok=True)

    cap = open_video(cfg.video_path)
    fps = get_fps(cap, cfg.fps_assumed)
    frame_size = get_frame_size(cap)  # (w, h)
    w, h = frame_size
    writer = make_writer(cfg.output_path, fps, frame_size)

    dt = 1.0 / fps

# --- Kalman Filter parameters ---
# You can tune these; these defaults are good starting points.
# Day: meas_std ~ 0.5, accel_std ~ 1.0, gate_px ~ 50
# Night: meas_std ~ 2.0, accel_std ~ 3.0, gate_px ~ 80
    params = KFParams(
    meas_std=float(getattr(cfg, "meas_std", 2.0)),
    accel_std=float(getattr(cfg, "accel_std", 3.0)),
    gate_px=float(getattr(cfg, "gate_px", 80.0)),
    )

    kf = KalmanTracker(dt, params)


    voter = TemporalColorVoter(ColorVoteConfig())

    # --- thresholds ---
    gate_dist_px = float(getattr(cfg, "gate_dist_px", 110.0))
    roi_recenter_after = int(getattr(cfg, "roi_recenter_after", 30))
    retry_expansions: List[int] = list(getattr(cfg, "retry_expansions", [25, 45, 65]))
    retry_accept_px = float(getattr(cfg, "retry_accept_px", gate_dist_px * 1.35))

    metrics = Metrics()
    det_prev: Optional[Point] = None
    kf_prev: Optional[Point] = None

    roi_tracker: Optional[RoiTemplateTracker] = None
    roi_current: ROI = cfg.roi
    miss_streak_for_roi = 0

    # last known bbox + last KF point for bbox shifting
    last_bbox: Optional[BBOX] = None
    last_kf_pt: Optional[Point] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1

        # Ensure writer always receives 3-channel BGR frames
        if frame is not None and len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame is not None and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Initialize ROI tracker
        if roi_tracker is None:
            roi_tracker = RoiTemplateTracker(frame, cfg.roi)
            roi_current = cfg.roi

        # Update ROI via template tracker
        roi_current = roi_tracker.update(frame)

        # KF predict early
        pred_pt: Optional[Point] = None
        if kf.initialized:
            pred_pt = kf.predict()

        # Base ROI
        roi_now: ROI = roi_current
        if miss_streak_for_roi >= roi_recenter_after and pred_pt is not None:
            roi_now = _center_roi_on_point(roi_current, pred_pt, w, h)

        # First pass detection
        det = detect_traffic_light(
            frame=frame,
            roi=roi_now,
            min_area=cfg.min_contour_area,
            max_area=cfg.max_contour_area,
        )

        # Escalating retries only if missing
        if det is None:
            for i, ex in enumerate(retry_expansions):
                if pred_pt is not None:
                    roi_retry = _center_roi_on_point(roi_now, pred_pt, w, h)
                    roi_retry = _expand_roi(roi_retry, ex, w, h)
                else:
                    roi_retry = _expand_roi(roi_now, ex, w, h)

                det2 = detect_traffic_light(
                    frame=frame,
                    roi=roi_retry,
                    min_area=cfg.min_contour_area,
                    max_area=cfg.max_contour_area,
                )
                if det2 is None:
                    continue

                if pred_pt is not None:
                    d = _dist((det2.cx, det2.cy), pred_pt)
                    if d <= retry_accept_px:
                        det = det2
                        roi_now = roi_retry
                        recovered_on_retry += 1
                        retry_level_hits[min(i, 2)] += 1
                        break
                    else:
                        rejected_retry += 1
                        continue
                else:
                    det = det2
                    roi_now = roi_retry
                    recovered_on_retry += 1
                    retry_level_hits[min(i, 2)] += 1
                    break

        det_raw_pt: Optional[Point] = (det.cx, det.cy) if det is not None else None

        # Miss accounting + debug dumps
        if det_raw_pt is None:
            raw_misses += 1
            miss_streak += 1
            max_miss_streak = max(max_miss_streak, miss_streak)
            miss_streak_for_roi += 1

            if (total_frames % 30) == 0:
                dbg = frame.copy()
                draw_roi(dbg, roi_now)
                cv2.imwrite(os.path.join(debug_dir, f"miss_{total_frames:06d}.png"), dbg)
        else:
            det_frames += 1
            miss_streak = 0
            miss_streak_for_roi = 0

        # Initialize KF on first valid detection
        if (not kf.initialized) and det_raw_pt is not None:
            kf.init(det_raw_pt[0], det_raw_pt[1])

        # KF update with gating
# KF update (gating happens inside the KF via params.gate_px)
    kf_now: Optional[Point] = None
    if kf.initialized:
        kf_now = kf.update(det_raw_pt)
 

        # --- Visualization & continuity outputs ---

        # ROI box
        draw_roi(frame, roi_now)

        # Color voting: hold stable label (even during misses we keep last stable)
        if det is not None:
            stable_color = voter.update(det.color)
        else:
            stable_color = voter.update(None)

        label = stable_color if stable_color is not None else "UNKNOWN"
        if stable_color is not None:
            color_stable_frames += 1

        # Draw detection bbox when present
        if det is not None:
            draw_detection(frame, det.bbox, label=f"DETECTION {label}")
            last_bbox = det.bbox

        # Always draw KF point when available
        if kf_now is not None:
            draw_track_point(frame, kf_now[0], kf_now[1], label="KALMAN")
            track_visible_frames += 1

            # If no detection bbox this frame, draw a "TRACK" bbox by shifting last bbox with KF motion
            if det is None and last_bbox is not None and last_kf_pt is not None:
                dx = kf_now[0] - last_kf_pt[0]
                dy = kf_now[1] - last_kf_pt[1]
                track_bbox = _shift_bbox(last_bbox, dx, dy, w, h)
                draw_detection(frame, track_bbox, label=f"TRACK {label}")

            last_kf_pt = (kf_now[0], kf_now[1])

        writer.write(frame)

        # Metrics: use RAW detection point for detector-based misses/jitter
        update_metrics(metrics, det_prev, det_raw_pt, kf_prev, kf_now)
        det_prev = det_raw_pt
        kf_prev = kf_now

    cap.release()
    writer.release()

    det_avg_step = metrics.det_jitter_sum / metrics.det_steps if metrics.det_steps else 0.0
    kf_avg_step = metrics.kf_jitter_sum / metrics.kf_steps if metrics.kf_steps else 0.0
    det_rate = (det_frames / total_frames) if total_frames else 0.0
    track_rate = (track_visible_frames / total_frames) if total_frames else 0.0
    stable_color_rate = (color_stable_frames / total_frames) if total_frames else 0.0

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
    print("Stable color frames:", color_stable_frames)
    print("Stable color rate:", stable_color_rate)

    print("Missed detections:", metrics.missed_detections)
    print("Avg detection step (pixels):", det_avg_step)
    print("Avg KF step (pixels):", kf_avg_step)


if __name__ == "__main__":
    cfg = Config(
        video_path="videos/night.mov",
        output_path="outputs/annotated_night.mp4",
        roi=(260, 0, 220, 300),
        min_contour_area=20,
        max_contour_area=8000,
        fps_assumed=30.0,
        # Optional knobs if your Config supports them:
        # gate_dist_px=110.0,
        # roi_recenter_after=30,
        # retry_accept_px=150.0,
        # retry_expansions=[25, 45, 65],
    )
    run(cfg)
