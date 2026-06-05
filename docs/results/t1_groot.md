# T1 GR00T N1.7 Eval: LIBERO Spatial

Checkpoint: `programs/checkpoints/groot_n17/libero_spatial/libero_spatial`  
Embodiment: `LIBERO_PANDA`  
Episodes per task: 20  

## Per-Task Results

| Task | Success Rate | Grasp Rate |
|---|---|---|
| pick up the black bowl between the plate and the ramekin and place it on the plate | **1.000** | 0.650 |
| pick up the black bowl from table center and place it on the plate | **1.000** | 1.000 |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | **1.000** | 1.000 |
| pick up the black bowl next to the cookie box and place it on the plate | **1.000** | 1.000 |
| pick up the black bowl next to the plate and place it on the plate | **0.950** | 0.900 |
| pick up the black bowl next to the ramekin and place it on the plate | **1.000** | 1.000 |
| pick up the black bowl on the cookie box and place it on the plate | **1.000** | 0.000 |
| pick up the black bowl on the ramekin and place it on the plate | **0.900** | 0.000 |
| pick up the black bowl on the stove and place it on the plate | **0.900** | 1.000 |
| pick up the black bowl on the wooden cabinet and place it on the plate | **0.950** | 0.300 |

## Summary

| Metric | Value |
|---|---|
| mean_task_success | **0.970** |
| T0 BC baseline | 0.500 (single task) |
| GR00T N1.7 (official NVIDIA) | ~0.977 (200-ep benchmark) |
