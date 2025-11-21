#!/usr/bin/env python3
r"""
prepare_upload_bundle.py

Create a cleaned dataset bundle for upload to Google Drive by copying only
relevant data files from the workspace into upload_bundle/adr_datasets,
excluding PDFs, images, caches, checkpoints, and other non-essential artifacts.

Usage (from repo root on Windows):
    py -3 colab_pipeline\prepare_upload_bundle.py \
      --source-root . \
      --out upload_bundle/adr_datasets

Recommended: Run once, then compress the folder and upload to Drive.
  PowerShell:
    Compress-Archive -Path upload_bundle/adr_datasets -DestinationPath adr_datasets_clean.zip -Force

This script preserves relative paths per dataset directory so your Colab run can
use --data_root pointing to the uploaded folder.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
import zipfile


DEFAULT_INCLUDE_DIRS = [
    # Core dataset repos/folders commonly present in this workspace
    "ADRtarget-master",
    "CT-ADE-main-dataset",
    "DrugMeNot-main",
    "Hybrid-Adverse-Drug-Reactions-Predictor-main",
    "RecSys23-ADRnet-main",
    "SIDER4-master-dataset",
    "siderData",
    "faersData",
    "pubChemData",
    "chemBlData",
    # Some datasets may have spaces in names
    "ML Clinical Trials - Galeano and Paccanaro-dataset",
]


# Keep: typical tabular/text data and model-related binary artifacts
WHITELIST_EXTS = {
    ".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson",
    ".parquet", ".feather", ".pkl", ".pickle",
    ".gz", ".bz2", ".xz",  # compressed data (e.g., .tsv.gz)
    ".zip",  # sometimes datasets include zipped splits
    ".smi", ".sdf", ".mol", ".mol2",  # chemistry file formats
    ".fps",  # ChEMBL fingerprints
}


# Skip: docs, images, notebooks, code caches, build artifacts, etc.
BLACKLIST_EXTS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg",
    ".tif", ".tiff", ".webp",
    ".ppt", ".pptx", ".doc", ".docx", ".rtf",
    ".md", ".rst", ".html", ".htm",
    ".ipynb", ".log",
}

BLACKLIST_DIR_NAMES = {
    ".git", "__pycache__", ".ipynb_checkpoints",
    "figs", "images", "image", "img", "imgs", "pictures", "screenshots",
    "logger", "logs", "results", "final_results", "auprc", "auroc", "trajectory",
    "build", "dist", ".venv", "venv", "env", "node_modules", "cache", "__cache__",
    "docs", "documentation", "notebooks", "code files",
}


def has_whitelisted_ext(path: Path) -> bool:
    if not path.is_file():
        return False
    # Handle multi-suffix like .tsv.gz
    suffixes = [s.lower() for s in path.suffixes]
    if not suffixes:
        return False
    # If any suffix is blacklisted, skip
    if any(s in BLACKLIST_EXTS for s in suffixes):
        return False
    # If any suffix is whitelisted, keep
    return any(s in WHITELIST_EXTS for s in suffixes)


DATA_DIR_HINTS = {"data", "dataset", "datasets", "manual"}


def path_in_preferred_data_dirs(root_dir: Path, file_path: Path) -> bool:
    rel = file_path.relative_to(root_dir)
    parts = {p.lower() for p in rel.parts[:-1]}  # exclude filename
    return any(h in parts for h in DATA_DIR_HINTS)


def should_skip_dir(dir_name: str) -> bool:
    name = dir_name.lower()
    return name in BLACKLIST_DIR_NAMES


def copy_filtered_tree(src_root: Path, dst_root: Path) -> tuple[int, int, int]:
    """
    Copy files from src_root to dst_root filtering by whitelist/blacklist.
    Returns: (files_copied, files_skipped, dirs_skipped)
    """
    files_copied = 0
    files_skipped = 0
    dirs_skipped = 0

    for current_dir, dirnames, filenames in walk_with_pruning(src_root):
        rel_dir = current_dir.relative_to(src_root)
        target_dir = dst_root / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        for fname in filenames:
            src_file = current_dir / fname
            # Prefer files under data/dataset/manual folders; otherwise keep only if whitelisted
            if path_in_preferred_data_dirs(src_root, src_file):
                if has_whitelisted_ext(src_file):
                    dst_file = target_dir / fname
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    files_copied += 1
                else:
                    files_skipped += 1
                continue

            if has_whitelisted_ext(src_file):
                dst_file = target_dir / fname
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                files_copied += 1
            else:
                files_skipped += 1

    return files_copied, files_skipped, dirs_skipped


def walk_with_pruning(root: Path):
    """os.walk with directory pruning based on BLACKLIST_DIR_NAMES."""
    for dirpath, dirnames, filenames in os_walk_sorted(root):
        # Prune blacklisted dirs in-place
        keep = []
        for d in dirnames:
            if should_skip_dir(d):
                continue
            keep.append(d)
        dirnames[:] = keep
        yield Path(dirpath), dirnames, filenames


def os_walk_sorted(root: Path):
    # Local import to avoid overhead if not used elsewhere
    import os
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        yield dirpath, dirnames, filenames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a cleaned dataset upload bundle")
    p.add_argument("--source-root", default=".", help="Workspace root containing dataset folders")
    p.add_argument("--out", default="upload_bundle/adr_datasets", help="Output directory for cleaned bundle")
    p.add_argument("--auto-detect", action="store_true", help="Auto-include any top-level folder that contains whitelisted data files")
    p.add_argument("--zip-out", default=None, help="Write files directly to this zip archive instead of copying to --out")
    p.add_argument("--max-file-mb", type=int, default=0, help="Skip files larger than this size (MB). 0 disables size filter.")
    p.add_argument("--dry-run", action="store_true", help="List what would be copied without writing files")
    return p.parse_args()


def find_candidate_dirs(source_root: Path, auto_detect: bool) -> list[Path]:
    candidates: list[Path] = []
    if auto_detect:
        for p in source_root.iterdir():
            if not p.is_dir():
                continue
            if p.name in {"colab_pipeline", ".git", "upload_bundle"}:
                continue
            # Heuristic: include if it contains at least one whitelisted data file somewhere
            try:
                for dirpath, _, filenames in os_walk_sorted(p):
                    if any(has_whitelisted_ext(Path(dirpath) / f) for f in filenames):
                        candidates.append(p)
                        break
            except PermissionError:
                continue
    else:
        for name in DEFAULT_INCLUDE_DIRS:
            p = source_root / name
            if p.exists() and p.is_dir():
                candidates.append(p)
    return candidates


def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def main():
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out).resolve()

    if not args.zip_out:
        out_root.mkdir(parents=True, exist_ok=True)

    candidates = find_candidate_dirs(source_root, args.auto_detect)
    if not candidates:
        print("No candidate dataset directories found. Consider --auto-detect.")
        sys.exit(1)

    print("Preparing cleaned upload bundle…\n")
    print(f"Source root: {source_root}")
    print(f"Output dir : {out_root}")
    print("Included dataset folders:")
    for c in candidates:
        print(f"  • {c.relative_to(source_root)}")
    print()

    total_copied = 0
    total_skipped = 0

    if args.dry_run:
        print("Dry-run mode: not copying files. Estimating sizes…")

    def size_ok(p: Path) -> bool:
        if args.max_file_mb and p.is_file():
            try:
                return (p.stat().st_size / (1024 * 1024)) <= args.max_file_mb
            except OSError:
                return False
        return True

    if args.zip_out:
        zip_path = Path(args.zip_out).resolve()
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            for c in candidates:
                size = dir_size(c)
                print(f"Would ZIP {c.name} (current size: {human_size(size)})")
        else:
            with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for c in candidates:
                    print(f"Zipping {c.name}…")
                    for current_dir, dirnames, filenames in walk_with_pruning(c):
                        for fname in filenames:
                            src_file = current_dir / fname
                            if not has_whitelisted_ext(src_file):
                                continue
                            if not (path_in_preferred_data_dirs(c, src_file) or has_whitelisted_ext(src_file)):
                                continue
                            if not size_ok(src_file):
                                total_skipped += 1
                                continue
                            arcname = Path(c.name) / src_file.relative_to(c)
                            zf.write(src_file, arcname)
                            total_copied += 1
                    print(f"{c.name}: added to zip")
            print(f"\nZIP created: {zip_path}")
    else:
        for c in candidates:
            target = out_root / c.name
            if args.dry_run:
                size = dir_size(c)
                print(f"Would process {c.name} (current size: {human_size(size)})")
                continue
            copied, skipped, _ = copy_filtered_tree(c, target)
            print(f"{c.name}: copied {copied:,} files, skipped {skipped:,}")
            total_copied += copied
            total_skipped += skipped

    if not args.dry_run:
        if args.zip_out:
            print("\nDone.")
            print(f"Total files zipped: {total_copied:,}")
            print(f"Total files skipped: {total_skipped:,}")
        else:
            bundle_size = dir_size(out_root)
            print("\nDone.")
            print(f"Total files copied: {total_copied:,}")
            print(f"Total files skipped: {total_skipped:,}")
            print(f"Bundle size: {human_size(bundle_size)} at {out_root}")
            # Write a MANIFEST for reference
            manifest = out_root / "MANIFEST.txt"
            with manifest.open("w", encoding="utf-8") as mf:
                mf.write("Cleaned dataset bundle manifest\n")
                mf.write(f"Source root: {source_root}\n")
                mf.write("Included folders:\n")
                for c in candidates:
                    mf.write(f"  - {c.relative_to(source_root)}\n")
            print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
