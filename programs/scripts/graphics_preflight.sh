#!/usr/bin/env bash
# P3 gate: verify Vulkan/graphics inside the Isaac Lab container before any
# camera/vision work. Must print a GPU under "Devices" for rendering to work.
set -e
docker exec isaac-lab-base bash -lc "vulkaninfo --summary 2>&1 | head -25 || echo VULKAN_NOT_AVAILABLE"
