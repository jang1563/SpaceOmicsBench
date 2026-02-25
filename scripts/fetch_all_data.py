#!/usr/bin/env python3
"""
Master script to fetch all public processed data for SpaceOmicsBench v2.

Usage:
    python fetch_all_data.py                # Full fetch (GeneLab + GEO)
    python fetch_all_data.py --dry-run      # Preview without downloading
    python fetch_all_data.py --genelab-only # Only GeneLab/OSDR data
    python fetch_all_data.py --supp-only    # Only supplementary data instructions
"""

import subprocess
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_script(script_name: str, args: list = None):
    """Run a sub-script."""
    script_path = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script_path)] + (args or [])
    print(f"\n{'#' * 70}")
    print(f"# Running: {script_name}")
    print(f"{'#' * 70}\n")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR.parent))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Fetch all data for SpaceOmicsBench v2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--genelab-only", action="store_true")
    parser.add_argument("--supp-only", action="store_true")
    args = parser.parse_args()

    extra_args = []
    if args.dry_run:
        extra_args.append("--dry-run")

    print("=" * 70)
    print("SpaceOmicsBench v2 (Public) -- Master Data Fetch")
    print("=" * 70)

    if not args.supp_only:
        # Step 1: Fetch GeneLab/OSDR processed data
        print("\n\nSTEP 1: Fetching GeneLab/OSDR processed data...")
        # Priority datasets first
        priority = "OSD-569,OSD-530,OSD-571"
        rc = run_script("fetch_genelab.py", ["--datasets", priority] + extra_args)
        if rc != 0:
            print(f"Warning: GeneLab fetch returned code {rc}")

    if not args.genelab_only:
        # Step 2: Generate supplementary download instructions + fetch GEO
        print("\n\nSTEP 2: Supplementary data + GEO...")
        rc = run_script("fetch_supplementary.py", extra_args)
        if rc != 0:
            print(f"Warning: Supplementary fetch returned code {rc}")

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Data fetch complete. Next steps:

1. Check data/ directory for GeneLab processed files
2. Follow docs/DOWNLOAD_INSTRUCTIONS.md for manual supplementary downloads
3. After all data is collected, run the processing pipeline:
   python scripts/process_to_benchmark.py (coming soon)

Key directories:
  data/clinical/           -- CBC/CMP biomarkers
  data/transcriptomics/    -- RNA-seq, cfRNA, single-cell
  data/proteomics/         -- Plasma + EVP proteins
  data/metabolomics/       -- Plasma metabolites
  sources/P01-P10/         -- Raw supplementary files by paper
""")


if __name__ == "__main__":
    main()
