"""
System monitoring for Boltz prediction runs.

On macOS:
  - Launches asitop in a new Terminal window for live CPU/GPU visualisation.
  - Runs powermetrics in the background and generates a power-usage plot.

On Linux:
  - Polls /proc/stat for CPU usage and nvidia-smi for GPU usage (if available).
  - Generates the same style of usage plot after the run.

On unsupported platforms monitoring is skipped gracefully.
"""

import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


# ---------------------------------------------------------------------------
# asitop — live terminal dashboard (macOS only)
# ---------------------------------------------------------------------------

def launch_asitop() -> None:
    if not IS_MACOS:
        print(f"Note: live monitoring (asitop) is macOS-only — skipping on {platform.system()}.")
        return

    asitop_bin = Path(sys.executable).parent / "asitop"
    if not asitop_bin.exists():
        print("Warning: asitop not found in venv — skipping live monitor.")
        return

    script = f'tell application "Terminal" to do script "sudo {asitop_bin}"'
    subprocess.Popen(["osascript", "-e", script])
    print("Launched asitop in a new Terminal window.")


# ---------------------------------------------------------------------------
# macOS monitor — powermetrics
# ---------------------------------------------------------------------------

class MacOSMonitor:
    """
    Runs 'sudo powermetrics' in the background (macOS only).
    Requires cached sudo credentials — run 'sudo -v' before starting.
    """

    INTERVAL_MS = 1000

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._proc = None
        self._log_fh = None

    def start(self) -> bool:
        can_sudo = subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
        if not can_sudo:
            print(
                "Warning: powermetrics needs sudo — run 'sudo -v' first to cache "
                "credentials. Skipping usage logging."
            )
            return False

        self._log_fh = open(self.log_path, "w")
        self._proc = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "cpu_power,gpu_power",
             "-i", str(self.INTERVAL_MS)],
            stdout=self._log_fh,
            stderr=subprocess.DEVNULL,
        )
        print(f"powermetrics logging started → {self.log_path}")
        return True

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._log_fh:
            self._log_fh.close()

    def plot(self, output_path: Path) -> None:
        text = self.log_path.read_text(errors="replace")
        cpu_pct = [int(x) for x in re.findall(r"CPU Power:\s+(\d+) mW", text)]
        gpu_pct = [int(x) for x in re.findall(r"GPU Power:\s+(\d+) mW", text)]
        interval_s = self.INTERVAL_MS / 1000
        _save_plot(cpu_pct, gpu_pct, interval_s, output_path,
                   cpu_label="CPU", gpu_label="GPU (MPS)",
                   ylabel="Power (W)", scale=1/1000)


# ---------------------------------------------------------------------------
# Linux monitor — /proc/stat + nvidia-smi
# ---------------------------------------------------------------------------

class LinuxMonitor:
    """
    Polls CPU usage from /proc/stat and GPU usage from nvidia-smi every second.
    Falls back gracefully if nvidia-smi is unavailable.
    """

    INTERVAL_S = 1

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._stop = threading.Event()
        self._thread = None
        self._cpu_samples: list[float] = []
        self._gpu_samples: list[float] = []
        self._has_nvidia = self._check_nvidia()

    @staticmethod
    def _check_nvidia() -> bool:
        return subprocess.run(
            ["nvidia-smi"], capture_output=True
        ).returncode == 0

    @staticmethod
    def _cpu_usage() -> float:
        """Read instantaneous CPU usage % from /proc/stat."""
        try:
            lines = Path("/proc/stat").read_text().splitlines()
            vals = list(map(int, lines[0].split()[1:]))
            idle = vals[3]
            total = sum(vals)
            return round((1 - idle / total) * 100, 1)
        except Exception:
            return 0.0

    def _gpu_usage(self) -> float:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
            )
            return float(out.decode().strip().split("\n")[0])
        except Exception:
            return 0.0

    def _poll(self) -> None:
        prev_vals = None
        while not self._stop.is_set():
            # Delta-based CPU so first sample is accurate
            try:
                vals = list(map(int, Path("/proc/stat").read_text().split("\n")[0].split()[1:]))
                if prev_vals:
                    idle_delta = vals[3] - prev_vals[3]
                    total_delta = sum(v - p for v, p in zip(vals, prev_vals))
                    cpu = round((1 - idle_delta / max(total_delta, 1)) * 100, 1)
                    self._cpu_samples.append(cpu)
                prev_vals = vals
            except Exception:
                pass

            if self._has_nvidia:
                self._gpu_samples.append(self._gpu_usage())

            time.sleep(self.INTERVAL_S)

    def start(self) -> bool:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        gpu_info = "nvidia-smi GPU" if self._has_nvidia else "no GPU monitor (nvidia-smi not found)"
        print(f"Linux monitor started — CPU + {gpu_info}")
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def plot(self, output_path: Path) -> None:
        _save_plot(
            self._cpu_samples, self._gpu_samples, self.INTERVAL_S, output_path,
            cpu_label="CPU", gpu_label="GPU (NVIDIA)",
            ylabel="Utilisation (%)", scale=1,
        )


# ---------------------------------------------------------------------------
# Shared plot helper
# ---------------------------------------------------------------------------

def _save_plot(
    cpu_samples: list,
    gpu_samples: list,
    interval_s: float,
    output_path: Path,
    cpu_label: str,
    gpu_label: str,
    ylabel: str,
    scale: float,
) -> None:
    import matplotlib.pyplot as plt

    if not cpu_samples and not gpu_samples:
        print("Warning: no usage data collected — skipping plot.")
        return

    n = max(len(cpu_samples), len(gpu_samples))
    t = [i * interval_s for i in range(n)]

    fig, ax = plt.subplots(figsize=(12, 4))
    if cpu_samples:
        ax.plot(t[:len(cpu_samples)], [v * scale for v in cpu_samples],
                label=cpu_label, color="steelblue", linewidth=1.5)
    if gpu_samples:
        ax.plot(t[:len(gpu_samples)], [v * scale for v in gpu_samples],
                label=gpu_label, color="darkorange", linewidth=1.5)

    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title("CPU & GPU Usage — Boltz2 Prediction Run")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Usage plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Public factory — returns the right monitor for the current platform
# ---------------------------------------------------------------------------

def PowerMetricsMonitor(log_path: Path):
    """Return the appropriate monitor for the current platform."""
    if IS_MACOS:
        return MacOSMonitor(log_path)
    elif IS_LINUX:
        return LinuxMonitor(log_path)
    else:
        return _NoOpMonitor()


class _NoOpMonitor:
    """Fallback for unsupported platforms — does nothing silently."""
    def start(self) -> bool:
        print(f"Note: usage monitoring not supported on {platform.system()} — skipping.")
        return False
    def stop(self) -> None:
        pass
    def plot(self, output_path: Path) -> None:
        pass
