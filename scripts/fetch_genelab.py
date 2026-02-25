#!/usr/bin/env python3
"""
Fetch processed data from NASA GeneLab / OSDR for SpaceOmicsBench v2.

Verified methods (tested 2026-02-24):
  1. Biodata REST API file listing  --> most reliable for getting file names
  2. GEODE download endpoint        --> follows S3 redirect, works for all files
  3. OSDR file listing API          --> works for some datasets (fallback)
  4. AWS S3 bulk download           --> fastest for large datasets (requires aws cli)

NOTE: The Biodata Query API (file.data_type=normalized_counts) does NOT work
for our datasets because they use custom pipelines (CLC Genomics, ONT pipeline),
not GeneLab's standard bulk RNA-seq pipeline.

No authentication required for any method.

IMPORTANT: File names use GLDS IDs, not OSD IDs:
  OSD-569 -> GLDS-561 / LSDS-7
  OSD-570 -> GLDS-562
  OSD-571 -> GLDS-563
  OSD-530 -> GLDS-530
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path

# ---- Configuration ----

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BIODATA_API = "https://visualization.osdr.nasa.gov/biodata/api"
OSDR_API = "https://osdr.nasa.gov"
GEODE_DOWNLOAD = "https://osdr.nasa.gov/geode-py/ws/studies"

# Datasets to fetch with VERIFIED file patterns (based on actual API responses)
DATASETS = {
    "OSD-569": {
        "description": "I4 Whole Blood (CBC, long-read RNA-seq, WGS, m6A, clonal hematopoiesis)",
        "glds_id": "GLDS-561",
        "lsds_id": "LSDS-7",
        "target_dirs": {
            "clinical": ["CBC"],
            "transcriptomics/bulk_rnaseq": ["Gene_Expression_Processed", "m6A_Processed"],
        },
        # Actual verified file names from biodata API
        "processed_files": [
            "LSDS-7_Complete_Blood_Count_CBC_TRANSFORMED.csv",
            "LSDS-7_Complete_Blood_Count_CBC.upload_SUBMITTED.csv",
            "GLDS-561_long-readRNAseq_Direct_RNA_seq_Gene_Expression_Processed.xlsx",
            "GLDS-561_directm6Aseq_Direct_RNA_seq_m6A_Processed_Data.xlsx",
        ],
    },
    "OSD-570": {
        "description": "I4 PBMCs (snRNA-seq + snATAC-seq, TCR/BCR V(D)J)",
        "glds_id": "GLDS-562",
        "target_dirs": {
            "transcriptomics/single_cell": ["Processed_Data", "VDJ_Results"],
        },
        "processed_files": [
            "GLDS-562_snRNA-Seq_PBMC_Gene_Expression_snRNA-seq_Processed_Data.xlsx",
            "GLDS-562_snATAC-Seq_PBMC_Chromatin_Accessibility_snATAC-seq_Processed_Data.xlsx",
            "GLDS-562_scRNA-Seq_VDJ_Results.xlsx",
        ],
    },
    "OSD-571": {
        "description": "I4 Plasma (proteomics, EVP proteomics, metabolomics)",
        "glds_id": "GLDS-563",
        "target_dirs": {
            "proteomics": ["proteomics"],
            "metabolomics": ["metabolomics"],
        },
        "processed_files": [
            "GLDS-563_proteomics_Plasma_Proteomics_Processed_Data.xlsx",
            "GLDS-563_proteomics_plasma_proteomics_preprocessed_data.tsv",
            "GLDS-563_proteomics_plasma_metadata_all_samples_collapsed.csv",
            "GLDS-563_proteomics_EVP_Proteomics_Processed_Data.xlsx",
            "GLDS-563_proteomics_EVPs_proteomics_preprocessed_data.txt",
            "GLDS-563_proteomics_EVPs_sample_metadata.csv",
            "GLDS-563_metabolomics_Plasma_Metabolomics_Processed_Data.xlsx",
            "GLDS-563_metabolomics_metabolomics_RPPOS-NEG_preprocessed_data.xlsx",
            "GLDS-563_metabolomics_metabolomics_ANPPOS-NEG_preprocessed_data.xlsx",
        ],
    },
    "OSD-530": {
        "description": "JAXA CFE Human cfRNA (6 astronauts, 11 timepoints)",
        "glds_id": "GLDS-530",
        "target_dirs": {
            "transcriptomics/cfrna": ["totalcount", "normalized", "466genes"],
        },
        "processed_files": [
            "GLDS-530_rna-seq_TGB_050_64samples_3group_totalcount_all0removed_scalingnormalized.xlsx",
            "GLDS-530_rna-seq_TGB_050_64samples_3group_totalcount.xlsx",
            "GLDS-530_rna-seq_TGB_050_64samples_3group_totalcount_all0removed_scalingnormalized_ANOVA_FDRpval005_2x_50difference_466genes.xlsx",
            "GLDS-530_rna-seq_TGB_050_1_2_64samples_9group_totalcount_all0removed_scalingnormalized_pairwise_analysis_included.xlsx",
            "GLDS-530_rna-seq_TGB_050_1_2_64samples_9group_totalcount_all0removed_scalingnormalized_SEM.xlsx",
            "GLDS-530_rna-seq_TGB_050_1_2_64samples_11group_totalcount_all0removed_scalingnormalized_SEM.xlsx",
            "GLDS-530_rna-seq_TGB_063_Input_vs_IP_totalcount_all0removed_scalingnormalized.xlsx",
        ],
    },
    "OSD-574": {
        "description": "I4 Skin (spatial transcriptomics NanoString GeoMx, metagenomics)",
        "glds_id": None,  # Need to discover
        "target_dirs": {
            "spatial_transcriptomics": ["Processed", "DCC", "GeoMx"],
        },
        "processed_files": [],  # Will be discovered via API
    },
    "OSD-572": {
        "description": "I4 Skin/Oral/Nasal Swabs (metagenomics + metatranscriptomics)",
        "glds_id": None,
        "target_dirs": {
            "microbiome/human": ["abundance", "taxonomy", "report"],
        },
        "processed_files": [],  # 5,404 files -- will filter via API
    },
    "OSD-573": {
        "description": "I4 Environmental (Dragon capsule metagenomics)",
        "glds_id": None,
        "target_dirs": {
            "microbiome/environmental": ["abundance", "taxonomy", "report"],
        },
        "processed_files": [],
    },
    "OSD-575": {
        "description": "I4 Blood Serum (metabolic panel, cytokine arrays)",
        "glds_id": None,
        "target_dirs": {
            "clinical": ["Serum", "cytokine", "metabolic"],
        },
        "processed_files": [],
    },
    "OSD-630": {
        "description": "I4 Stool (gut microbiome metagenomics)",
        "glds_id": None,
        "target_dirs": {
            "microbiome/human": ["abundance", "taxonomy", "report"],
        },
        "processed_files": [],
    },
    "OSD-656": {
        "description": "I4 Urine (inflammation panel, NULISAseq)",
        "glds_id": None,
        "target_dirs": {
            "clinical": ["Urine", "NULISA", "inflammation"],
        },
        "processed_files": [],
    },
    "OSD-687": {
        "description": "I4 T-cells (histone modification: H3K4me1, H3K4me3, H3K27ac)",
        "glds_id": None,
        "target_dirs": {
            "epigenomics": ["histone", "peaks", "Processed"],
        },
        "processed_files": [],
    },
    "OSD-532": {
        "description": "JAXA MHU-1 Mouse cfRNA",
        "glds_id": "GLDS-532",
        "target_dirs": {
            "transcriptomics/cfrna": ["counts", "normalized"],
        },
        "processed_files": [],
    },
}


def list_files_biodata_api(osd_id: str) -> dict:
    """List files using the Biological Data REST API (MOST RELIABLE METHOD)."""
    url = f"{BIODATA_API}/v2/dataset/{osd_id}/files/"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Response structure: {osd_id: {files: {filename: info, ...}}}
        osd_data = data.get(osd_id, data)
        files = osd_data.get("files", {})
        if isinstance(files, dict):
            return files
        return {}
    except Exception as e:
        print(f"  Warning: Biodata API failed for {osd_id}: {e}")
        return {}


def list_files_osdr_api(osd_id: str) -> list:
    """List files using OSDR file listing API (FALLBACK -- inconsistent)."""
    numeric_id = osd_id.replace("OSD-", "")
    url = f"{OSDR_API}/osdr/data/osd/files/{numeric_id}"
    params = {"page": 0, "size": 25, "all_files": "true"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        study = data.get("studies", {}).get(osd_id, {})
        return study.get("study_files", [])
    except Exception as e:
        print(f"  Warning: OSDR API failed for {osd_id}: {e}")
        return []


def download_file_geode(osd_id: str, filename: str, outpath: Path) -> bool:
    """Download a file using the GEODE endpoint (follows S3 redirect)."""
    url = f"{GEODE_DOWNLOAD}/{osd_id}/download"
    params = {"source": "datamanager", "file": filename}

    try:
        # allow_redirects=True follows the 302 -> S3 presigned URL
        resp = requests.get(url, params=params, stream=True, timeout=300,
                            allow_redirects=True)
        resp.raise_for_status()

        outpath.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with open(outpath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)

        size_mb = total / (1024 * 1024)
        print(f"  Downloaded: {filename} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
        if outpath.exists() and outpath.stat().st_size == 0:
            outpath.unlink()
        return False


def determine_target_dir(filename: str, config: dict) -> str:
    """Determine which data subdirectory a file belongs to."""
    for target_dir, patterns in config.get("target_dirs", {}).items():
        if any(pat.lower() in filename.lower() for pat in patterns):
            return target_dir
    # Default: first target dir
    target_dirs = list(config.get("target_dirs", {}).keys())
    return target_dirs[0] if target_dirs else "misc"


def fetch_dataset(osd_id: str, config: dict, dry_run: bool = False):
    """Fetch processed files for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {osd_id} -- {config['description']}")
    print(f"{'='*60}")

    # Step 1: Use pre-verified file list if available
    files_to_download = config.get("processed_files", [])

    # Step 2: If no pre-verified list, discover via API
    if not files_to_download:
        print(f"  Discovering files via Biodata API...")
        file_dict = list_files_biodata_api(osd_id)

        if file_dict:
            all_filenames = list(file_dict.keys())
            print(f"  Found {len(all_filenames)} total files")

            # Filter: skip raw data, keep processed
            for fname in all_filenames:
                fl = fname.lower()
                # Skip raw FASTQ, BAM, md5, metadata zips
                if any(skip in fl for skip in [".fastq", ".bam", ".bai", ".sra",
                                                 "md5sum", "metadata_osd"]):
                    continue
                # Keep processed files (xlsx, csv, tsv, txt data, reports)
                if any(ext in fl for ext in [".xlsx", ".csv", ".tsv",
                                              "processed", "report"]):
                    files_to_download.append(fname)

            # For large datasets (microbiome), limit to key files
            if len(files_to_download) > 50:
                print(f"  Large dataset ({len(files_to_download)} processed files)")
                print(f"  Filtering to key summary files only...")
                key_files = [f for f in files_to_download
                             if any(kw in f.lower() for kw in
                                    ["processed", "summary", "report", "abundance",
                                     "taxonomy", "metadata"])]
                if key_files:
                    files_to_download = key_files[:30]
                else:
                    files_to_download = files_to_download[:30]

        if not files_to_download:
            # Fallback: OSDR API
            print(f"  Trying OSDR file listing API...")
            osdr_files = list_files_osdr_api(osd_id)
            for f in osdr_files:
                fname = f.get("file_name", "")
                if fname and not any(skip in fname.lower()
                                     for skip in [".fastq", ".bam", "md5sum"]):
                    files_to_download.append(fname)

    if not files_to_download:
        print(f"  No processable files found. Try manually:")
        print(f"  https://osdr.nasa.gov/bio/repo/data/studies/{osd_id}")
        return

    print(f"\n  Files to download: {len(files_to_download)}")

    # Step 3: Download each file
    downloaded = 0
    for fname in files_to_download:
        target_dir = determine_target_dir(fname, config)
        target_path = DATA_DIR / target_dir
        target_path.mkdir(parents=True, exist_ok=True)
        outpath = target_path / fname

        if outpath.exists() and outpath.stat().st_size > 0:
            print(f"  Skipped (exists): {fname}")
            downloaded += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] Would download: {fname} -> data/{target_dir}/")
        else:
            if download_file_geode(osd_id, fname, outpath):
                downloaded += 1
            time.sleep(0.5)  # Rate limiting

    print(f"\n  Result: {downloaded}/{len(files_to_download)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GeneLab processed data for SpaceOmicsBench v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_genelab.py --dry-run                    # Preview all downloads
  python fetch_genelab.py --datasets OSD-530           # Fetch JAXA CFE only
  python fetch_genelab.py --datasets OSD-569,OSD-571   # Fetch I4 clinical + proteomics
  python fetch_genelab.py --method s3                   # Show AWS S3 commands
        """,
    )
    parser.add_argument(
        "--datasets", type=str, default="all",
        help="Comma-separated OSD IDs or 'all'",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview without downloading",
    )
    parser.add_argument(
        "--method", choices=["api", "s3", "both"], default="api",
        help="Download method",
    )
    args = parser.parse_args()

    if args.datasets == "all":
        selected = DATASETS
    else:
        ids = [d.strip() for d in args.datasets.split(",")]
        selected = {k: v for k, v in DATASETS.items() if k in ids}

    if not selected:
        print("No valid datasets selected. Available:")
        for k, v in DATASETS.items():
            print(f"  {k}: {v['description']}")
        sys.exit(1)

    print("SpaceOmicsBench v2 -- GeneLab Data Fetcher")
    print(f"Fetching {len(selected)} datasets...")
    print(f"Output directory: {DATA_DIR}")
    if args.dry_run:
        print("*** DRY RUN MODE ***\n")

    for osd_id, config in selected.items():
        fetch_dataset(osd_id, config, dry_run=args.dry_run)

    if args.method in ("s3", "both"):
        print("\n\n--- AWS S3 commands (alternative bulk download) ---")
        print("Requires: pip install awscli\n")
        for osd_id, config in selected.items():
            print(f"# {osd_id}: {config['description']}")
            print(f'aws s3 ls s3://nasa-osdr/{osd_id}/ --no-sign-request')
            print(f'aws s3 sync s3://nasa-osdr/{osd_id}/ ./{osd_id}/ '
                  f'--no-sign-request --exclude "*" '
                  f'--include "*.xlsx" --include "*.csv" --include "*.tsv"')
            print()

    print("\nDone!")


if __name__ == "__main__":
    main()
