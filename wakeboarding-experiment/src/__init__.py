"""Wakeboarding-start humanoid RL — source package.

CPU-importable: heavy Isaac Lab imports are guarded so this package can be
syntax/import-checked on a machine without Isaac Sim (mirrors the pattern in
my-humanoid-project). Reward math, rope model, curriculum, and AMP are pure
PyTorch and import everywhere.
"""
