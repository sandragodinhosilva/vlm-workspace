#!/usr/bin/env python3
"""
Cleanup Qwen checkpoints - keep only the final checkpoint per training run.

Usage:
    # Dry run
    python cleanup_qwen_checkpoints.py --dry-run

    # Actually delete
    python cleanup_qwen_checkpoints.py
"""

import argparse
import shutil
from pathlib import Path


def cleanup_qwen_checkpoints(qwen_dir: Path, dry_run: bool = True):
    """Keep only the final checkpoint for each Qwen training run."""

    stats = {
        "deleted_dirs": [],
        "kept_dirs": [],
        "space_freed": 0,
        "errors": []
    }

    if not qwen_dir.exists():
        print(f"⚠️  Qwen directory not found: {qwen_dir}")
        return stats

    print(f"🔍 Scanning Qwen directory: {qwen_dir}")
    print()

    # Find all training run subdirectories
    for run_dir in sorted(qwen_dir.glob("*")):
        if not run_dir.is_dir():
            continue

        print(f"📁 {run_dir.name}")

        # Find all step checkpoints
        step_dirs = sorted(run_dir.glob("step_*"), key=lambda p: int(p.name.split('_')[1]))

        if not step_dirs:
            print(f"   ⚠️  No checkpoints found")
            print()
            continue

        print(f"   Total checkpoints: {len(step_dirs)}")

        # Keep the final (highest step number) checkpoint
        final_checkpoint = step_dirs[-1]
        print(f"   Final checkpoint: {final_checkpoint.name}")

        for step_dir in step_dirs:
            # Calculate size
            size = sum(f.stat().st_size for f in step_dir.rglob('*') if f.is_file())
            size_gb = size / (1024**3)

            if step_dir == final_checkpoint:
                print(f"   ✓ Keeping: {step_dir.name} ({size_gb:.1f} GB)")
                stats["kept_dirs"].append(str(step_dir))
            else:
                if dry_run:
                    print(f"   🗑️  Would delete: {step_dir.name} ({size_gb:.1f} GB)")
                    stats["space_freed"] += size
                else:
                    print(f"   🗑️  Deleting: {step_dir.name} ({size_gb:.1f} GB)")
                    try:
                        shutil.rmtree(step_dir)
                        stats["deleted_dirs"].append(str(step_dir))
                        stats["space_freed"] += size
                    except Exception as e:
                        error_msg = f"Failed to delete {step_dir}: {e}"
                        print(f"   ❌ {error_msg}")
                        stats["errors"].append(error_msg)

        print()

    return stats


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup Qwen checkpoints to keep only final checkpoint per run"
    )

    parser.add_argument(
        '--qwen-dir',
        type=Path,
        default=Path('/mnt/data/sgsilva/checkpoints'),
        help='Path to Qwen checkpoint directory (default: /mnt/data/sgsilva/checkpoints)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what will be deleted without actually deleting'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("QWEN CHECKPOINT CLEANUP SCRIPT")
    print("=" * 80)
    print()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
    else:
        print("⚠️  LIVE MODE - Files will be permanently deleted!")
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return

    print()

    stats = cleanup_qwen_checkpoints(args.qwen_dir, dry_run=args.dry_run)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Directories deleted: {len(stats['deleted_dirs'])}")
    print(f"Directories kept: {len(stats['kept_dirs'])}")
    print(f"Space freed: {format_size(stats['space_freed'])}")

    if stats["errors"]:
        print()
        print(f"⚠️  Errors: {len(stats['errors'])}")
        for error in stats["errors"]:
            print(f"  - {error}")

    print()

    if args.dry_run:
        print("🔍 This was a dry run. Use without --dry-run to actually delete files.")
    else:
        print("✅ Cleanup complete!")

    print()


if __name__ == "__main__":
    main()
