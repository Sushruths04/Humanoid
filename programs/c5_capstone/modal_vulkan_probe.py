r"""Minimal Modal L40S Vulkan probe.

The full Isaac compatibility gate failed at `vkCreateInstance` with
`ERROR_INCOMPATIBLE_DRIVER` because `/etc/vulkan/icd.d/nvidia_icd.json` did not
exist. That JSON is normally installed by the NVIDIA driver package, but Modal
mounts the driver at *runtime*, not at image-build time, so no ICD manifest is
present.

This probe answers the single decisive question cheaply (small image, ~1 min
run): does Modal's runtime-mounted L40S driver actually ship the NVIDIA Vulkan
loader library (libGLX_nvidia.so.0)? If yes, we synthesize the ICD manifest at
runtime and Vulkan should initialize. If the library is absent, Modal cannot
host Isaac Sim's RTX renderer and the question is settled.

Run:
    .\.venv-modal\Scripts\modal.exe run programs\c5_capstone\modal_vulkan_probe.py
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time

import modal

APP_NAME = "c5-vulkan-probe"
BASE_IMAGE = "nvidia/cuda:12.8.1-runtime-ubuntu24.04"

app = modal.App(APP_NAME)

# Tiny image: just the Vulkan loader + tools. Builds in seconds, not minutes.
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .apt_install(
        "vulkan-tools",
        "libvulkan1",
        "pciutils",
        "binutils",
        # libGLX_nvidia.so.0 links against these X11 libs at dlopen time.
        "libxext6",
        "libx11-6",
        "libxrandr2",
        "libxrender1",
        "libxcb1",
        "libxau6",
        "libxdmcp6",
    )
)


def _sh(name: str, command: str, timeout_s: int = 60) -> dict:
    print(f"\n=== {name} ===", flush=True)
    print(f"$ {command}", flush=True)
    started = time.monotonic()
    proc = subprocess.run(
        command,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    out = proc.stdout or ""
    print(out, flush=True)
    return {
        "name": name,
        "returncode": proc.returncode,
        "duration_s": round(time.monotonic() - started, 2),
        "output_tail": out[-6000:],
    }


@app.function(image=image, gpu="L40S", timeout=10 * 60)
def vulkan_probe() -> dict:
    checks: list[dict] = []

    # 0. Identify the sandbox. gVisor's nvproxy serves CUDA-compute ioctls but
    #    not the graphics/Vulkan ioctl surface — if this is gVisor, Vulkan failure
    #    is a fundamental runtime limitation, not a fixable config gap.
    checks.append(_sh(
        "sandbox identity",
        "echo '--- uname ---'; uname -a; "
        "echo '--- /proc/version ---'; cat /proc/version; "
        "echo '--- gVisor markers ---'; (dmesg 2>&1 | grep -i gvisor | head) ; "
        "grep -qi gvisor /proc/version && echo 'GVISOR DETECTED in /proc/version' || echo 'no gvisor string in /proc/version'; "
        "echo '--- nvidia device nodes ---'; ls -la /dev/nvidia* 2>&1; "
        "echo '--- runsc/sentry hint ---'; cat /proc/1/cmdline 2>/dev/null | tr '\\0' ' '; echo",
    ))

    # 1. Confirm the GPU is visible (CUDA path).
    checks.append(_sh("nvidia-smi", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"))

    # 2. Find every NVIDIA-related shared library Modal mounted at runtime.
    #    The Vulkan ICD library we need is libGLX_nvidia.so.* — its presence is
    #    the make-or-break fact.
    search_roots = ["/usr/lib", "/usr/lib/x86_64-linux-gnu", "/lib", "/usr/local/lib", "/run", "/usr/lib64"]
    found_libs: list[str] = []
    for root in search_roots:
        for pat in ("libGLX_nvidia.so*", "libnvidia-glcore.so*", "libGLX_nvidia*", "libnvidia-vulkan*", "libnvidia-gpucomp*"):
            found_libs.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
    found_libs = sorted(set(found_libs))
    print("\n=== NVIDIA GL/Vulkan libraries found ===", flush=True)
    print("\n".join(found_libs) if found_libs else "(none)", flush=True)
    checks.append({
        "name": "nvidia vulkan libs",
        "returncode": 0 if any("libGLX_nvidia" in p for p in found_libs) else 1,
        "found_libs": found_libs,
        "output_tail": "\n".join(found_libs) if found_libs else "(none found)",
    })

    # Also dump the broader nvidia driver lib mount so we know what *is* present.
    checks.append(_sh("ldconfig nvidia libs", "ldconfig -p | grep -i nvidia || echo 'no nvidia libs in ldconfig cache'"))

    # 3. Synthesize the ICD manifest pointing at the (hopefully present) lib.
    icd_lib = "libGLX_nvidia.so.0"
    for p in found_libs:
        if "libGLX_nvidia.so" in p:
            icd_lib = os.path.basename(p)
            break
    icd_dir = "/etc/vulkan/icd.d"
    os.makedirs(icd_dir, exist_ok=True)
    icd_path = os.path.join(icd_dir, "nvidia_icd.json")
    with open(icd_path, "w") as f:
        json.dump({"file_format_version": "1.0.0", "ICD": {"library_path": icd_lib, "api_version": "1.3.277"}}, f, indent=2)
    print(f"\nWrote ICD manifest -> {icd_path} (library_path={icd_lib})", flush=True)

    # 3b. Did NVIDIA's own container mount drop an ICD anywhere? (It does so only
    #     when NVIDIA_DRIVER_CAPABILITIES includes "graphics".)
    checks.append(_sh(
        "search mounted ICDs + caps",
        "echo NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES; "
        "echo '--- existing icd json files ---'; "
        "find /usr /etc /run -name '*icd*.json' 2>/dev/null; "
        "echo '--- libGLX_nvidia exported vk symbols ---'; "
        "nm -D /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>/dev/null | grep -i 'vk_icd\\|vkCreateInstance' || echo '(nm unavailable or no symbols)'",
    ))

    # 4. The decisive test: can Vulkan now enumerate the NVIDIA GPU?
    #    Run with full loader debug to see exactly where it fails.
    env_prefix = f"VK_LOADER_DEBUG=all VK_ICD_FILENAMES={icd_path} VK_DRIVER_FILES={icd_path}"
    vk = _sh("vulkaninfo --summary", f"{env_prefix} vulkaninfo --summary")
    checks.append(vk)

    vk_ok = vk["returncode"] == 0 and "NVIDIA" in (vk.get("output_tail") or "")

    report = {
        "vulkan_works": vk_ok,
        "icd_library": icd_lib,
        "checks": checks,
    }
    print("\n=== VULKAN PROBE RESULT ===", flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2), flush=True)
    print(f"VULKAN ON MODAL L40S: {'WORKS' if vk_ok else 'DOES NOT WORK'}", flush=True)
    return report


@app.local_entrypoint()
def main() -> None:
    report = vulkan_probe.remote()
    if report["vulkan_works"]:
        print("\nSUCCESS: Vulkan initialized on Modal L40S. Isaac Sim rendering is feasible — proceed to fix the full gate.")
    else:
        print("\nCONFIRMED: Vulkan does NOT work on Modal L40S. Isaac rendering must stay on Lightning AI.")
