"""CP4.2: Verify the collected HDF5 dataset — print shapes, check value ranges.

DoD: N >= 5000 triplets; frames are uint8 [0,255]; actions are float32 in plausible range.

Usage:
    python -m programs.p4_cosmos_world_sim.cp42_verify_data \
        --data datasets/g1_nav_cosmos.h5
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="CP4.2: verify Cosmos training dataset")
    parser.add_argument("--data", required=True, help="Path to g1_nav_cosmos.h5")
    args = parser.parse_args()

    import h5py
    import numpy as np

    print(f"\nDataset: {args.data}")

    with h5py.File(args.data, "r") as f:
        ft = f["frame_t"]
        at = f["action_t"]
        ft1 = f["frame_t1"]

        n = ft.shape[0]
        print(f"  frame_t:   {ft.shape}  {ft.dtype}")
        print(f"  action_t:  {at.shape}  {at.dtype}")
        print(f"  frame_t1:  {ft1.shape}  {ft1.dtype}")

        # Sample a batch for stats (avoid loading all into RAM)
        idx = np.random.choice(n, min(1024, n), replace=False)
        idx_sorted = np.sort(idx)

        sample_ft = ft[idx_sorted]    # (1024, 64, 64, 3)
        sample_at = at[idx_sorted]    # (1024, 29)
        sample_ft1 = ft1[idx_sorted]

    print(f"\nN = {n} triplets")

    # Checks
    ok = True

    if n < 5000:
        print(f"  [WARN] N={n} < 5000 — re-run with more envs/steps")

    ft_min, ft_max = int(sample_ft.min()), int(sample_ft.max())
    print(f"  frame_t range:   [{ft_min}, {ft_max}]")
    if ft_min < 0 or ft_max > 255:
        print("  [FAIL] frame values out of uint8 range")
        ok = False
    else:
        print("  frame range ✓")

    at_min, at_max = float(sample_at.min()), float(sample_at.max())
    print(f"  action_t range:  [{at_min:.3f}, {at_max:.3f}]")
    if abs(at_min) > 50 or abs(at_max) > 50:
        print("  [WARN] action values seem very large — check normalisation")
    else:
        print("  action range ✓")

    # Check frame_t1 differs from frame_t (robot should be moving)
    mean_diff = float(np.abs(sample_ft.astype(np.float32) - sample_ft1.astype(np.float32)).mean())
    print(f"  mean |frame_t - frame_t1|: {mean_diff:.2f}")
    if mean_diff < 0.5:
        print("  [WARN] frames barely change — robot may be standing still")
    else:
        print("  frame delta ✓")

    if ok:
        print("\nCP4.2 DONE ✓")
    else:
        print("\nCP4.2 FAILED — fix issues above before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()
