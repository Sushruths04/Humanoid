"""Pull-speed curriculum (PLAN.md §6.1) — pure Python/PyTorch, CPU-testable.

Ramps the target pull speed 10 -> 15 -> 20 -> 25 -> 30 km/h. Advance a level when the
rolling success rate over `window` updates clears `promote_success_rate`.
"""
from __future__ import annotations

from collections import deque

from .rope_model import kmh_to_ms


class PullSpeedCurriculum:
    def __init__(
        self,
        levels_kmh: list[float],
        promote_success_rate: float = 0.60,
        window: int = 200,
        enabled: bool = True,
    ):
        self.levels_kmh = list(levels_kmh)
        self.promote = promote_success_rate
        self.enabled = enabled
        self.level = 0
        self._succ = deque(maxlen=window)

    @property
    def current_kmh(self) -> float:
        return self.levels_kmh[self.level]

    @property
    def current_ms(self) -> float:
        return kmh_to_ms(self.current_kmh)

    @property
    def at_max(self) -> bool:
        return self.level >= len(self.levels_kmh) - 1

    def update(self, batch_success_rate: float) -> bool:
        """Feed the latest eval/rollout success rate. Returns True if level advanced."""
        if not self.enabled or self.at_max:
            return False
        self._succ.append(batch_success_rate)
        if len(self._succ) >= self._succ.maxlen:
            avg = sum(self._succ) / len(self._succ)
            if avg >= self.promote:
                self.level += 1
                self._succ.clear()
                return True
        return False

    def state(self) -> dict:
        return {"level": self.level, "v_pull_kmh": self.current_kmh, "at_max": self.at_max}
