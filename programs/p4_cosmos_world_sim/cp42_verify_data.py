"""CP4.2 Verify Bridge-format dataset integrity.

Checks that the datasets/g1_nav directory has valid MP4 videos + JSON annotations
in Bridge format. Prints per-episode stats.

Usage:
    python programs/p4_cosmos_world_sim/cp42_verify_data.py --data datasets/g1_nav
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def verify(data_root: str) -> None:
    data_root = Path(data_root)
    vid_root = data_root / "videos" / "train"
    ann_root = data_root / "annotation" / "train"

    if not vid_root.exists():
        print(f"[FAIL] Video dir not found: {vid_root}")
        sys.exit(1)
    if not ann_root.exists():
        print(f"[FAIL] Annotation dir not found: {ann_root}")
        sys.exit(1)

    episodes = sorted([d.name for d in vid_root.iterdir() if d.is_dir()])
    print(f"Found {len(episodes)} episodes in {vid_root}")

    if len(episodes) == 0:
        print("[FAIL] No episodes found!")
        sys.exit(1)

    errors = []
    stats = {"total_frames": 0, "total_episodes": 0, "action_range": []}

    for ep in episodes:
        vid_path = vid_root / ep / "0" / "rgb.mp4"
        ann_path = ann_root / f"{ep}.json"

        if not vid_path.exists():
            errors.append(f"Episode {ep}: missing video {vid_path}")
            continue
        if not ann_path.exists():
            errors.append(f"Episode {ep}: missing annotation {ann_path}")
            continue

        # Load annotation
        with open(ann_path) as f:
            ann = json.load(f)

        actions = ann.get("action", [])
        gripper = ann.get("continuous_gripper_state", [])
        state = ann.get("state", [])
        T = len(actions)

        if T == 0:
            errors.append(f"Episode {ep}: empty actions")
            continue
        if len(gripper) != T + 1:
            errors.append(f"Episode {ep}: gripper_state len={len(gripper)} expected {T+1}")
        if len(state) != T:
            errors.append(f"Episode {ep}: state len={len(state)} expected {T}")

        # Validate action dims
        if len(actions[0]) != 6:
            errors.append(f"Episode {ep}: action dim={len(actions[0])} expected 6")

        # Check video
        try:
            import imageio
            reader = imageio.get_reader(str(vid_path))
            n_frames = reader.count_frames()
            meta = reader.get_meta_data()
            reader.close()
            if abs(n_frames - T) > 2:  # allow ±2 frame rounding
                errors.append(f"Episode {ep}: video {n_frames} frames vs annotation {T}")
        except Exception as e:
            errors.append(f"Episode {ep}: video read error: {e}")
            n_frames = T

        stats["total_frames"] += T
        stats["total_episodes"] += 1
        arr = np.array(actions)
        stats["action_range"].append((arr.min(), arr.max()))

    print(f"Episodes verified: {stats['total_episodes']}")
    print(f"Total frames:      {stats['total_frames']}")
    if stats["action_range"]:
        mins = min(x[0] for x in stats["action_range"])
        maxs = max(x[1] for x in stats["action_range"])
        print(f"Action range:      [{mins:.4f}, {maxs:.4f}] (Bridge /20 scale)")

    if errors:
        print(f"\n[WARN] {len(errors)} issues:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more")
    else:
        print("\nCP4.2 VERIFY OK — dataset looks clean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/g1_nav")
    args = parser.parse_args()
    verify(args.data)


if __name__ == "__main__":
    main()
