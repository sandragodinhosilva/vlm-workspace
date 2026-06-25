#!/usr/bin/env python3
"""
Interactive cache cleanup script.

Scans known cache and venv directories, shows sizes, and asks
per-directory whether to delete. Dry-run by default.

Usage:
    # Dry run — just show what can be cleaned
    python cleanup_caches.py

    # Interactive deletion — asks per directory
    python cleanup_caches.py --delete

    # Include venvs in scan
    python cleanup_caches.py --include-venvs

    # Include venvs + model checkpoints
    python cleanup_caches.py --include-venvs --include-models
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ── Directories to scan ──────────────────────────────────────────────────────

HOME = Path.home()
DATA = Path("/mnt/data/sgsilva")

CACHE_DIRS = [
    (HOME / ".cache" / "uv", "UV package manager cache"),
    (HOME / ".cache" / "pypoetry", "Poetry dependency cache"),
    (HOME / ".cache" / "vllm", "vLLM compiled kernels / downloaded models"),
    (HOME / ".cache" / "pip", "pip download cache"),
    (HOME / ".cache" / "flashinfer", "FlashInfer compiled kernels"),
    (HOME / ".cache" / "cloud-code", "Cloud Code IDE cache"),
    (HOME / ".cache" / "torch_extensions", "PyTorch JIT compiled extensions"),
    (HOME / ".cache" / "gradio_temp", "Gradio temporary files"),
    (HOME / "ray_tmp", "Ray session temp data"),
]

VENV_DIRS = [
    (DATA / "VLMEvalKit" / ".venv", "VLMEvalKit virtual environment"),
    # NOTE: SIBench-VSR, nvidia-rl, data-tuning venvs removed — no longer present on disk
]

MODEL_DIRS = [
    (DATA / "vlm-evaluation" / "results" / "archive", "Archived evaluation results"),
    (DATA / "vlm-evaluation" / "results" / "backups", "Evaluation result backups"),
    # NOTE: vlm-evaluation/kimi-serving removed — no longer present on disk
]


def dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    if not path.exists():
        return 0
    try:
        result = subprocess.run(
            ["du", "-sb", str(path)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return int(result.stdout.split()[0])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    # Fallback: walk manually
    total = 0
    try:
        for f in path.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def fmt(size_bytes: int) -> str:
    """Human-readable size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def ask_yes_no(prompt: str) -> bool:
    """Ask user y/n. Default no."""
    while True:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no", ""):
            return False


def scan_and_clean(dirs: list[tuple[Path, str]], *, delete: bool, category: str):
    """Scan a list of (path, description) tuples, show sizes, optionally delete."""
    print(f"\n{'='*70}")
    print(f"  {category}")
    print(f"{'='*70}\n")

    found = []
    for path, desc in dirs:
        if not path.exists():
            continue
        size = dir_size(path)
        if size < 1_000_000:  # skip <1MB
            continue
        found.append((path, desc, size))

    if not found:
        print("  Nothing found.\n")
        return 0

    # Sort largest first
    found.sort(key=lambda x: x[2], reverse=True)

    total_freed = 0

    for path, desc, size in found:
        print(f"  {fmt(size):>10}  {path}")
        print(f"             {desc}")

        if delete:
            if ask_yes_no("             Delete?"):
                try:
                    shutil.rmtree(path)
                    total_freed += size
                    print(f"             -> Deleted. Freed {fmt(size)}\n")
                except Exception as e:
                    print(f"             -> ERROR: {e}\n")
            else:
                print(f"             -> Skipped\n")
        else:
            print()

    subtotal = sum(s for _, _, s in found)
    print(f"  {'─'*50}")
    if delete:
        print(f"  Category total: {fmt(subtotal)} | Freed: {fmt(total_freed)}")
    else:
        print(f"  Category total: {fmt(subtotal)}")
    print()

    return total_freed


def main():
    parser = argparse.ArgumentParser(
        description="Interactive cache cleanup — dry-run by default",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Enable interactive deletion (asks per directory). Without this flag, only shows sizes.",
    )
    parser.add_argument(
        "--include-venvs", action="store_true",
        help="Also scan virtual environments (VLMEvalKit, SIBench, nvidia-rl, data-tuning)",
    )
    parser.add_argument(
        "--include-models", action="store_true",
        help="Also scan model/eval artifact directories",
    )
    args = parser.parse_args()

    mode = "DELETE (interactive)" if args.delete else "DRY RUN (read-only)"
    print(f"\n{'#'*70}")
    print(f"  Cache Cleanup — {mode}")
    print(f"{'#'*70}")

    total = 0

    total += scan_and_clean(CACHE_DIRS, delete=args.delete, category="System Caches (~/.cache + misc)")

    if args.include_venvs:
        total += scan_and_clean(VENV_DIRS, delete=args.delete, category="Virtual Environments")

    if args.include_models:
        total += scan_and_clean(MODEL_DIRS, delete=args.delete, category="Model / Eval Artifacts")

    # Summary
    print(f"{'='*70}")
    if args.delete:
        print(f"  Total freed: {fmt(total)}")
    else:
        print(f"  DRY RUN complete. Re-run with --delete to interactively clean up.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
