"""Reusable video recorder for all eval tasks (T1, T2, T3, P3, C5).

Usage:
    recorder = EpisodeRecorder(out_dir="programs/videos/t1_groot", fps=10)

    # inside episode loop:
    recorder.start_episode("task_name", episode_idx)
    # ... env step ...
    recorder.add_frame(obs_dict["agentview_image"])   # H×W×3 uint8
    recorder.finish_episode(success=True)              # saves only if success or force

    recorder.close()   # flush any pending writes
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path


class EpisodeRecorder:
    """Records one episode at a time; saves as MP4 via imageio-ffmpeg.

    Args:
        out_dir:         Directory for output MP4 files.
        fps:             Playback frame rate (default 10 — matches ~10 Hz eval loop).
        max_per_task:    How many episodes to save per task (saves first N successes,
                         then first N failures if not enough successes).
        record_failures: Also save failure episodes (up to max_per_task total).
    """

    def __init__(
        self,
        out_dir: str,
        fps: int = 10,
        max_per_task: int = 1,
        record_failures: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.fps = fps
        self.max_per_task = max_per_task
        self.record_failures = record_failures

        self._frames: list[np.ndarray] = []
        self._task: str = ""
        self._ep: int = 0
        self._task_saved: dict[str, int] = {}  # task → count saved
        self._active = False

    def start_episode(self, task_name: str, episode_idx: int) -> None:
        self._frames = []
        self._task = task_name
        self._ep = episode_idx
        self._active = True

    def add_frame(self, frame: np.ndarray) -> None:
        if self._active:
            self._frames.append(np.asarray(frame, dtype=np.uint8))

    def finish_episode(self, success: bool) -> str | None:
        """Save episode video if quota allows. Returns saved path or None."""
        if not self._active or not self._frames:
            self._active = False
            return None

        saved = self._task_saved.get(self._task, 0)
        should_save = success or (self.record_failures and saved < self.max_per_task)

        self._active = False
        if not should_save or saved >= self.max_per_task:
            self._frames = []
            return None

        label = "success" if success else "fail"
        fname = f"{self._task[:50]}_{label}_ep{self._ep:02d}.mp4"
        out_path = self.out_dir / fname
        self.out_dir.mkdir(parents=True, exist_ok=True)

        _write_mp4(out_path, self._frames, self.fps)
        self._frames = []
        self._task_saved[self._task] = saved + 1
        print(f"[video] saved → {out_path}")
        return str(out_path)

    def close(self) -> None:
        self._frames = []
        self._active = False


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    try:
        import imageio
        imageio.mimwrite(str(path), frames, fps=fps, macro_block_size=1)
    except Exception as e:
        print(f"[video] imageio failed ({e}), trying cv2...")
        try:
            import cv2
            h, w = frames[0].shape[:2]
            out = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
        except Exception as e2:
            print(f"[video] cv2 also failed ({e2}). No video saved.")
