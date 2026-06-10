"""Modal L40S compatibility gate for Isaac Sim and Isaac Lab.

This intentionally does not run C5. It only verifies that the existing public
Isaac Lab image can see the GPU, initialize Vulkan, and boot a headless scene.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "c5-isaac-compatibility"
BASE_IMAGE = "nvidia/cuda:12.8.1-runtime-ubuntu22.04"
ISAAC_SIM_VERSION = "5.1.0"
ISAAC_LAB_VERSION = "v2.3.2"

app = modal.App(APP_NAME)

# Install through NVIDIA's supported Python packages instead of importing the
# existing GHCR image, whose 6.99 GB monolithic layer stalls Modal's importer.
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .env(
        {
            "ACCEPT_EULA": "Y",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PRIVACY_CONSENT": "Y",
        }
    )
    .apt_install(
        "build-essential",
        "cmake",
        "git",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libx11-6",
        "libxcursor1",
        "libxi6",
        "libxinerama1",
        "libxrandr2",
        "vulkan-tools",
    )
    .run_commands(
        "python -m pip install --upgrade pip",
        (
            "python -m pip install "
            f"'isaacsim[all,extscache]=={ISAAC_SIM_VERSION}' "
            "--extra-index-url https://pypi.nvidia.com"
        ),
        (
            "git clone --depth 1 "
            f"--branch {ISAAC_LAB_VERSION} "
            "https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab"
        ),
        "cd /opt/IsaacLab && ./isaaclab.sh --install rsl_rl",
    )
)


def _run(name: str, command: list[str], timeout_seconds: int) -> dict:
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        output = result.stdout or ""
        print(output, flush=True)
        return {
            "name": name,
            "command": command,
            "returncode": result.returncode,
            "duration_seconds": round(time.monotonic() - started, 2),
            "output_tail": output[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        print(output, flush=True)
        print(f"{name} timed out after {timeout_seconds}s", flush=True)
        return {
            "name": name,
            "command": command,
            "returncode": 124,
            "duration_seconds": round(time.monotonic() - started, 2),
            "output_tail": output[-8000:],
        }


def _find_first(paths: list[str]) -> str | None:
    for candidate in paths:
        if Path(candidate).exists():
            return candidate
    return None


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    # Modal requires at least 512 GiB for this large Isaac Sim image.
    ephemeral_disk=512 * 1024,
    timeout=20 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    single_use_containers=True,
    env={
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
    },
)
def isaac_smoke() -> dict:
    checks: list[dict] = []

    checks.append(
        _run(
            "GPU visibility",
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            timeout_seconds=30,
        )
    )

    vulkaninfo = _find_first(
        ["/usr/bin/vulkaninfo", "/usr/local/bin/vulkaninfo"]
    )
    if vulkaninfo:
        checks.append(
            _run(
                "Vulkan summary",
                [vulkaninfo, "--summary"],
                timeout_seconds=60,
            )
        )
    else:
        checks.append(
            {
                "name": "Vulkan summary",
                "returncode": 0,
                "duration_seconds": 0,
                "output_tail": "vulkaninfo is not installed; Isaac boot remains the renderer gate.",
            }
        )

    compatibility = _find_first(
        [
            "/isaac-sim/isaac-sim.compatibility_check.sh",
            "/workspace/isaac-sim/isaac-sim.compatibility_check.sh",
        ]
    )
    if compatibility:
        checks.append(
            _run(
                "Isaac Sim compatibility checker",
                [compatibility, "--no-window", "--/app/quitAfter=10"],
                timeout_seconds=5 * 60,
            )
        )

    isaaclab_root = _find_first(
        [
            "/opt/IsaacLab",
            "/workspace/isaaclab",
            "/workspace/IsaacLab",
            "/IsaacLab",
            "/root/IsaacLab",
        ]
    )
    if not isaaclab_root:
        checks.append(
            {
                "name": "Isaac Lab root",
                "returncode": 2,
                "duration_seconds": 0,
                "output_tail": "Could not find the Isaac Lab checkout in the image.",
            }
        )
    else:
        tutorial = _find_first(
            [
                f"{isaaclab_root}/scripts/tutorials/00_sim/create_empty.py",
                f"{isaaclab_root}/source/standalone/tutorials/00_sim/create_empty.py",
            ]
        )
        launcher = f"{isaaclab_root}/isaaclab.sh"
        if not tutorial or not Path(launcher).exists():
            checks.append(
                {
                    "name": "Isaac Lab files",
                    "returncode": 2,
                    "duration_seconds": 0,
                    "output_tail": (
                        f"Missing launcher or tutorial under {isaaclab_root}."
                    ),
                }
            )
        else:
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                f"/workspace:/workspace/my-humanoid-project:"
                f"{isaaclab_root}/source:{env.get('PYTHONPATH', '')}"
            )
            command = [launcher, "-p", tutorial, "--headless"]
            print(f"Using Isaac Lab root: {isaaclab_root}", flush=True)
            started = time.monotonic()
            try:
                result = subprocess.run(
                    command,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=10 * 60,
                    check=False,
                )
                output = result.stdout or ""
                print("\n=== Isaac Lab headless boot ===", flush=True)
                print(output, flush=True)
                checks.append(
                    {
                        "name": "Isaac Lab headless boot",
                        "command": command,
                        "returncode": result.returncode,
                        "duration_seconds": round(time.monotonic() - started, 2),
                        "output_tail": output[-12000:],
                    }
                )
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ""
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                print(output, flush=True)
                checks.append(
                    {
                        "name": "Isaac Lab headless boot",
                        "command": command,
                        "returncode": 124,
                        "duration_seconds": round(time.monotonic() - started, 2),
                        "output_tail": output[-12000:],
                    }
                )

    passed = all(check["returncode"] == 0 for check in checks)
    report = {
        "passed": passed,
        "base_image": BASE_IMAGE,
        "isaac_sim_version": ISAAC_SIM_VERSION,
        "isaac_lab_version": ISAAC_LAB_VERSION,
        "checks": checks,
    }
    print("\n=== Modal Isaac compatibility result ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.local_entrypoint()
def main() -> None:
    report = isaac_smoke.remote()
    if not report["passed"]:
        failed = [
            check["name"]
            for check in report["checks"]
            if check["returncode"] != 0
        ]
        raise SystemExit(f"Compatibility gate failed: {', '.join(failed)}")
    print("Compatibility gate passed. Modal can proceed to the C5 baseline.")
