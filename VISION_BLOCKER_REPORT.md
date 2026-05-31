# Vision VLA Diagnostics: Vulkan/Graphics Blocker

While attempting to initiate Step 30 (Vision VLA), we encountered a persistent system-level blocker on the Lightning AI L40S instance.

## Error Details
The G1 simulation requires Vulkan rendering for its camera sensors. The current error is:
`ERROR: [Loader Message] Code 0 : libGLX_nvidia.so.0: cannot open shared object file: No such file or directory`

## Troubleshooting Steps Taken
1.  **Docker Config Update**: Modified `docker-compose.yaml` to explicitly request `capabilities: [gpu, graphics, compute, utility, video]`.
2.  **Environment Variables**: Set `NVIDIA_DRIVER_CAPABILITIES=all` and `NVIDIA_VISIBLE_DEVICES=all`.
3.  **Driver Verification**: Confirmed `nvidia-smi` works inside the container (Compute/CUDA is functional).
4.  **Path Forcing**: Manually pointed to NVIDIA EGL and Vulkan ICD files.

## Conclusion
The host machine's NVIDIA container runtime is not mounting the required `.so` libraries (like `libGLX_nvidia.so.0`) into the container. This prevents the robot from "seeing" pixels. 

**Recommendation**: 
To continue with Vision research, this workspace may need to be moved to a cloud provider or local machine where the `nvidia-container-toolkit` is configured with full `graphics` support enabled in `/etc/nvidia-container-runtime/config.toml`.

*Note: Phase 1 (GR00T) and Phase 2 (G1 Navigation) were unaffected as they only use Compute/CUDA, not Graphics/Vulkan.*
