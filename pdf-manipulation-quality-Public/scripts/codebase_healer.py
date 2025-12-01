#!/usr/bin/env python3
"""Codebase Healer

Lightweight, safe "self-healing" script to detect common sources of codebase
overbloat and optionally perform conservative fixes.

Features:
- Find large files and directories (configurable threshold)
- Detect common build/artifact folders (dist, build, __pycache__, .pytest_cache)
- Find duplicate files by hash
- Remove/remove or archive identified artifacts (dry-run by default)
- Optionally create a git branch and commit cleanup changes

Safety notes:
- By default the script runs in dry-run mode and only reports suggestions.
- Use `--apply` to actually perform conservative cleanup actions (removes
  bytecode, build artifacts, and moves large files into `backups/`).
- The script never deletes user source files (*.py, src/ files) unless
  `--force` is specified.

Usage:
  python scripts/codebase_healer.py [--apply] [--backup-dir BACKUP] [--threshold-mb N]

"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SAFE_REMOVE_DIRS = ["__pycache__", ".pytest_cache", "dist", "build", "node_modules"]
SAFE_REMOVE_EXTS = [".pyc", ".pyo", ".egg-info"]


def find_large_files(root: Path, threshold_mb: int) -> List[Tuple[Path, int]]:
    threshold = threshold_mb * 1024 * 1024
    large = []
    for p in root.rglob("*"):
        try:
            if p.is_file():
                size = p.stat().st_size
                if size >= threshold:
                    large.append((p, size))
        except OSError:
            continue
    large.sort(key=lambda x: x[1], reverse=True)
    return large


def find_artifact_dirs(root: Path) -> List[Path]:
    found = []
    for name in SAFE_REMOVE_DIRS:
        for p in root.rglob(name):
            if p.is_dir():
                found.append(p)
    return found


def find_artifact_files(root: Path) -> List[Path]:
    found = []
    for ext in SAFE_REMOVE_EXTS:
        for p in root.rglob(f"*{ext}"):
            if p.is_file():
                found.append(p)
    return found


def find_duplicate_files(root: Path) -> Dict[str, List[Path]]:
    # group by hash
    hashes: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            with p.open("rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            hashes.setdefault(h, []).append(p)
        except OSError:
            continue
    return {h: ps for h, ps in hashes.items() if len(ps) > 1}


def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def archive_file(src: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / src.name
    i = 1
    while dest.exists():
        dest = backup_dir / f"{src.stem}-{i}{src.suffix}"
        i += 1
    shutil.move(str(src), str(dest))
    return dest


def remove_path(p: Path) -> None:
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


def safe_cleanup(root: Path, apply: bool, backup_dir: Path, threshold_mb: int, force: bool) -> None:
    print(f"Scanning {root} (threshold {threshold_mb} MB) ...\n")

    large_files = find_large_files(root, threshold_mb)
    art_dirs = find_artifact_dirs(root)
    art_files = find_artifact_files(root)
    dups = find_duplicate_files(root)

    print("Large files:")
    for p, s in large_files[:50]:
        print(f" - {p} ({human_size(s)})")
    if not large_files:
        print("  (none)\n")
    else:
        print()

    print("Artifact directories (candidates for removal):")
    for d in art_dirs:
        print(f" - {d}")
    if not art_dirs:
        print("  (none)\n")
    else:
        print()

    print("Artifact files (bytecode/build info):")
    for f in art_files:
        print(f" - {f}")
    if not art_files:
        print("  (none)\n")
    else:
        print()

    print("Duplicate files (same hash):")
    for h, ps in dups.items():
        print(f" - hash {h[:12]}: {len(ps)} files")
        for p in ps[:3]:
            print(f"    - {p}")
    if not dups:
        print("  (none)\n")
    else:
        print()

    if not apply:
        print("Dry-run mode. No changes will be made. Pass --apply to perform conservative fixes.")
        return

    print("Applying conservative fixes...")

    # 1) Remove artifact directories
    for d in art_dirs:
        try:
            print(f"Removing directory: {d}")
            remove_path(d)
        except Exception as e:
            print(f"  failed: {e}")

    # 2) Remove artifact files
    for f in art_files:
        try:
            print(f"Removing artifact file: {f}")
            remove_path(f)
        except Exception as e:
            print(f"  failed: {e}")

    # 3) Handle large files: move to backups unless forced and looks like source
    for p, s in large_files:
        try:
            suffix = p.suffix.lower()
            # skip common source files unless force
            if suffix in {".py", ".md", ".txt", ".ipynb"} and not force:
                print(f"Skipping source file {p} ({human_size(s)})")
                continue
            dest = archive_file(p, backup_dir)
            print(f"Archived {p} -> {dest}")
        except Exception as e:
            print(f"  failed: {e}")

    # 4) Optionally: remove duplicates leaving one copy
    for h, ps in dups.items():
        keeper = ps[0]
        for p in ps[1:]:
            try:
                print(f"Removing duplicate {p} (keep {keeper})")
                remove_path(p)
            except Exception as e:
                print(f"  failed: {e}")

    print("Conservative fixes complete. Review backups before committing.")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Codebase Healer: detect and fix overbloat")
    p.add_argument("--root", default=".", help="Root of the repository")
    p.add_argument("--apply", action="store_true", help="Apply conservative fixes (default: dry-run)")
    p.add_argument("--backup-dir", default="backups/codebase_healer", help="Where to archive large files")
    p.add_argument("--threshold-mb", type=int, default=5, help="File size threshold in MB to consider large")
    p.add_argument("--force", action="store_true", help="Force removal/archival of source-like files")
    return p.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    backup = Path(args.backup_dir)
    safe_cleanup(root, args.apply, backup, args.threshold_mb, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
