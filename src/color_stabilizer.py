# color_stabilizer.py
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import ceil
from typing import Deque, Optional


@dataclass
class ColorVoteConfig:
    """Tunable parameters for temporal color voting."""
    window: int = 12              # number of recent frames
    majority: float = 0.60        # votes needed for RED/YELLOW
    majority_green: float = 0.70  # stricter for GREEN (night is harder)
    margin: int = 2               # hysteresis vs switching
    max_consecutive_misses: int = 12  # drop label to None after long miss streak


class TemporalColorVoter:
    """Stabilize per-frame labels via majority voting + hysteresis."""
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

        # optional: after many misses, go unknown
        if self._miss_streak >= self.cfg.max_consecutive_misses:
            self._stable = None
            return self._stable

        counts = Counter([c for c in self._hist if c != "NONE"])
        if not counts:
            return self._stable  # keep last known state

        winner, winner_votes = counts.most_common(1)[0]

        maj = self.cfg.majority_green if winner == "GREEN" else self.cfg.majority
        needed = ceil(self.cfg.window * maj)
        if winner_votes < needed:
            return self._stable

        # hysteresis: only switch if winner clearly beats current stable
        if self._stable is not None and winner != self._stable:
            stable_votes = counts.get(self._stable, 0)
            if (winner_votes - stable_votes) < self.cfg.margin:
                return self._stable

        self._stable = winner
        return self._stable
