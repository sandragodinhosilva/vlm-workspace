#!/usr/bin/env python3
"""
Log cleanup — prune old date-dirs from /mnt/data/sgsilva/logs.

Layout: logs/<category>/YYYY-MM-DD/  (category dirs: eval, misc, dataset, oracle, serve, export)
Nested subdirs (e.g. eval/serve/YYYY-MM-DD, eval/slurm/YYYY-MM-DD) are handled too.
logs/_archive/ is never auto-deleted — reported with a manual rm hint.

Usage:
    # Dry run — show what would be deleted (default: keep last 30 days)
    python cleanup_all.py --cleanup-logs --keep-days 30 --dry-run

    # Live run
    python cleanup_all.py --cleanup-logs --keep-days 30
"""

import argparse
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


def format_size(size_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def _dir_size(path: Path) -> int:
    try:
        result = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return int(result.stdout.split()[0])
    except Exception:
        pass
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _try_parse_date(name: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(name, fmt)
        except ValueError:
            continue
    return None


def cleanup_old_logs(logs_dir: Path, keep_days: int = 30, dry_run: bool = True) -> dict:
    stats = {"deleted_dirs": [], "kept_dirs": [], "space_freed": 0, "errors": []}

    if not logs_dir.exists():
        print(f"⚠️  Logs directory not found: {logs_dir}")
        return stats

    print(f"🔍 Scanning logs directory: {logs_dir}")
    print(f"   Keeping logs from last {keep_days} days")
    print()

    cutoff_date = datetime.now() - timedelta(days=keep_days)

    def _prune_or_keep(date_dir: Path, category: str):
        d = _try_parse_date(date_dir.name)
        if d is None:
            return
        size = _dir_size(date_dir)
        size_gb = size / (1024 ** 3)
        days_old = (datetime.now() - d).days
        if d < cutoff_date:
            if dry_run:
                print(f"   🗑️  Would delete: {category}/{date_dir.name} ({days_old}d old, {size_gb:.2f} GB)")
                stats["space_freed"] += size
            else:
                print(f"   🗑️  Deleting: {category}/{date_dir.name} ({size_gb:.2f} GB)")
                try:
                    shutil.rmtree(date_dir)
                    stats["deleted_dirs"].append(str(date_dir))
                    stats["space_freed"] += size
                except Exception as e:
                    msg = f"Failed to delete {date_dir}: {e}"
                    print(f"   ❌ {msg}")
                    stats["errors"].append(msg)
        else:
            print(f"   ✓ Keeping: {category}/{date_dir.name} ({days_old}d old, {size_gb:.2f} GB)")
            stats["kept_dirs"].append(str(date_dir))

    for cat_dir in sorted(logs_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("_"):
            continue
        for sub in sorted(cat_dir.iterdir()):
            if not sub.is_dir():
                continue
            if _try_parse_date(sub.name):
                _prune_or_keep(sub, cat_dir.name)
            else:
                for subsub in sorted(sub.iterdir()):
                    if subsub.is_dir():
                        _prune_or_keep(subsub, f"{cat_dir.name}/{sub.name}")

    archive_dir = logs_dir / "_archive"
    if archive_dir.exists():
        size = _dir_size(archive_dir)
        print()
        print(f"   📦 _archive/ ({format_size(size)}) — cold-archived pre-reorganization logs.")
        print(f"      Delete manually if no longer needed: rm -rf {archive_dir}")
        stats["kept_dirs"].append(str(archive_dir))

    print()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prune old date-dirs from /mnt/data/sgsilva/logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("/mnt/data/sgsilva/logs"))
    parser.add_argument("--cleanup-logs", action="store_true", help="Prune old log date-dirs")
    parser.add_argument("--keep-days", type=int, default=30, help="Retain logs newer than N days (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    args = parser.parse_args()

    if not args.cleanup_logs:
        parser.error("Specify --cleanup-logs (the only supported mode)")

    print("=" * 80)
    print("LOG CLEANUP")
    print("=" * 80)
    print()

    if args.dry_run:
        print("🔍 DRY RUN — no files will be deleted")
    else:
        print("⚠️  LIVE MODE — files will be permanently deleted!")
        if input("Continue? (yes/no): ").lower() != "yes":
            print("Aborted.")
            return
    print()

    stats = cleanup_old_logs(args.logs_dir, args.keep_days, args.dry_run)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Space freed: {format_size(stats['space_freed'])}")
    if stats["errors"]:
        print(f"⚠️  Errors: {len(stats['errors'])}")
        for e in stats["errors"]:
            print(f"  - {e}")
    print()
    if args.dry_run:
        print("🔍 Dry run complete. Re-run without --dry-run to apply.")
    else:
        print("✅ Done.")
    print()


if __name__ == "__main__":
    main()
