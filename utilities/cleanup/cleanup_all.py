#!/usr/bin/env python3
"""
Comprehensive cleanup script for freeing disk space.

This script can:
1. Clean up old log files (keep only recent ones)
2. Clean up Qwen checkpoints (keep only final checkpoint)
3. Clear cache directories
4. Provide detailed space analysis

Usage:
    # Dry run (see what would be deleted)
    python cleanup_all.py --dry-run

    # Clean up logs older than 7 days
    python cleanup_all.py --cleanup-logs --keep-days 7

    # Clean up Qwen checkpoints
    python cleanup_all.py --cleanup-qwen

    # Clean up cache
    python cleanup_all.py --cleanup-cache

    # Do everything
    python cleanup_all.py --cleanup-logs --cleanup-qwen --cleanup-cache --keep-days 7
"""

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def cleanup_old_logs(logs_dir: Path, keep_days: int = 7, dry_run: bool = True):
    """Delete log directories older than keep_days."""

    stats = {
        "deleted_dirs": [],
        "kept_dirs": [],
        "space_freed": 0,
        "errors": []
    }

    if not logs_dir.exists():
        print(f"⚠️  Logs directory not found: {logs_dir}")
        return stats

    print(f"🔍 Scanning logs directory: {logs_dir}")
    print(f"   Keeping logs from last {keep_days} days")
    print()

    cutoff_date = datetime.now() - timedelta(days=keep_days)

    # Find all date-named directories
    for log_dir in sorted(logs_dir.glob("*")):
        if not log_dir.is_dir():
            continue

        # Try to parse directory name as date (YYYYMMDD)
        try:
            dir_date = datetime.strptime(log_dir.name, "%Y%m%d")
        except ValueError:
            # Not a date directory, skip
            print(f"   ? Skipping (not a date): {log_dir.name}")
            stats["kept_dirs"].append(str(log_dir))
            continue

        # Calculate size
        size = sum(f.stat().st_size for f in log_dir.rglob('*') if f.is_file())
        size_gb = size / (1024**3)

        if dir_date < cutoff_date:
            # Old log, delete
            if dry_run:
                print(f"   🗑️  Would delete: {log_dir.name} ({size_gb:.1f} GB)")
                stats["space_freed"] += size
            else:
                print(f"   🗑️  Deleting: {log_dir.name} ({size_gb:.1f} GB)")
                try:
                    shutil.rmtree(log_dir)
                    stats["deleted_dirs"].append(str(log_dir))
                    stats["space_freed"] += size
                except Exception as e:
                    error_msg = f"Failed to delete {log_dir}: {e}"
                    print(f"   ❌ {error_msg}")
                    stats["errors"].append(error_msg)
        else:
            # Recent log, keep
            days_old = (datetime.now() - dir_date).days
            print(f"   ✓ Keeping: {log_dir.name} ({days_old} days old, {size_gb:.1f} GB)")
            stats["kept_dirs"].append(str(log_dir))

    print()
    return stats


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


def cleanup_cache(cache_dir: Path, dry_run: bool = True):
    """Clear cache directory."""

    stats = {
        "deleted_dirs": [],
        "space_freed": 0,
        "errors": []
    }

    if not cache_dir.exists():
        print(f"⚠️  Cache directory not found: {cache_dir}")
        return stats

    # Calculate size
    size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    size_gb = size / (1024**3)

    print(f"🔍 Cache directory: {cache_dir}")
    print(f"   Size: {size_gb:.1f} GB")
    print()

    if dry_run:
        print(f"   🗑️  Would delete entire cache ({size_gb:.1f} GB)")
        stats["space_freed"] = size
    else:
        print(f"   🗑️  Deleting cache ({size_gb:.1f} GB)")
        try:
            shutil.rmtree(cache_dir)
            stats["deleted_dirs"].append(str(cache_dir))
            stats["space_freed"] = size
            print(f"   ✓ Cache cleared")
        except Exception as e:
            error_msg = f"Failed to delete {cache_dir}: {e}"
            print(f"   ❌ {error_msg}")
            stats["errors"].append(error_msg)

    print()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive cleanup script for disk space management",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--logs-dir',
        type=Path,
        default=Path('/mnt/data/sgsilva/logs'),
        help='Path to logs directory'
    )

    parser.add_argument(
        '--qwen-dir',
        type=Path,
        default=Path('/mnt/data/sgsilva/checkpoints'),
        help='Path to Qwen checkpoint directory (default: /mnt/data/sgsilva/checkpoints)'
    )

    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('/mnt/data/sgsilva/.cache/nemo_rl'),
        help='Path to cache directory'
    )

    parser.add_argument(
        '--cleanup-logs',
        action='store_true',
        help='Clean up old log files'
    )

    parser.add_argument(
        '--cleanup-qwen',
        action='store_true',
        help='Clean up Qwen checkpoints'
    )

    parser.add_argument(
        '--cleanup-cache',
        action='store_true',
        help='Clear cache directory'
    )

    parser.add_argument(
        '--keep-days',
        type=int,
        default=7,
        help='Keep logs from last N days (default: 7)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what will be deleted without actually deleting'
    )

    args = parser.parse_args()

    # Check if any cleanup option is selected
    if not (args.cleanup_logs or args.cleanup_qwen or args.cleanup_cache):
        parser.error("At least one cleanup option must be specified: --cleanup-logs, --cleanup-qwen, or --cleanup-cache")

    print("=" * 80)
    print("COMPREHENSIVE CLEANUP SCRIPT")
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

    total_space_freed = 0
    all_errors = []

    # Cleanup logs
    if args.cleanup_logs:
        print("=" * 80)
        print("CLEANING UP OLD LOGS")
        print("=" * 80)
        print()

        logs_stats = cleanup_old_logs(args.logs_dir, args.keep_days, args.dry_run)
        total_space_freed += logs_stats["space_freed"]
        all_errors.extend(logs_stats["errors"])

    # Cleanup Qwen checkpoints
    if args.cleanup_qwen:
        print("=" * 80)
        print("CLEANING UP QWEN CHECKPOINTS")
        print("=" * 80)
        print()

        qwen_stats = cleanup_qwen_checkpoints(args.qwen_dir, args.dry_run)
        total_space_freed += qwen_stats["space_freed"]
        all_errors.extend(qwen_stats["errors"])

    # Cleanup cache
    if args.cleanup_cache:
        print("=" * 80)
        print("CLEANING UP CACHE")
        print("=" * 80)
        print()

        cache_stats = cleanup_cache(args.cache_dir, args.dry_run)
        total_space_freed += cache_stats["space_freed"]
        all_errors.extend(cache_stats["errors"])

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total space that will be freed: {format_size(total_space_freed)}")

    if all_errors:
        print()
        print(f"⚠️  Errors: {len(all_errors)}")
        for error in all_errors:
            print(f"  - {error}")

    print()

    if args.dry_run:
        print("🔍 This was a dry run. Use without --dry-run to actually delete files.")
    else:
        print("✅ Cleanup complete!")

    print()


if __name__ == "__main__":
    main()
