#!/usr/bin/env python3
"""
Fetch supplementary data from published papers for SpaceOmicsBench v2.

Downloads processed data tables from Nature, Nature Communications,
Communications Biology, and Science paper supplementary files.

Note: Some papers host supplementary data as direct download links,
others require manual download from the article page. This script
handles the automated cases and provides instructions for manual ones.
"""

import os
import sys
import time
import argparse
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES_DIR = BASE_DIR / "sources"

# Paper metadata and supplementary file info
PAPERS = {
    "P01": {
        "short_title": "SOMA Atlas",
        "doi": "10.1038/s41586-024-07639-y",
        "article_url": "https://www.nature.com/articles/s41586-024-07639-y",
        "dir": "P01_SOMA_atlas",
        "supp_count": 8,
        "supp_description": "Supplementary Tables 1-8 (sample details, OSDR annotations, pathway analysis, deconvolution, microbial CV)",
        "auto_download": False,
        "notes": "Download supplementary tables manually from the article page. Tables are in Excel format.",
    },
    "P02": {
        "short_title": "Secretome (Proteomics+Metabolomics)",
        "doi": "10.1038/s41467-024-48841-w",
        "article_url": "https://www.nature.com/articles/s41467-024-48841-w",
        "dir": "P02_secretome",
        "supp_count": 1,
        "supp_description": "Source Data ZIP (3 MB): proteomics + metabolomics differential results",
        "auto_download": False,
        "notes": "Download Source Data ZIP from article page. Contains all figure-level processed data.",
    },
    "P03": {
        "short_title": "Single-cell Multi-ome",
        "doi": "10.1038/s41467-024-49211-2",
        "article_url": "https://www.nature.com/articles/s41467-024-49211-2",
        "dir": "P03_singlecell_multiome",
        "supp_count": 15,
        "supp_description": "Supplementary Data 1-15: GO pathways, IPA, TF motifs, GSEA, drug candidates, cytokines, conserved signatures",
        "auto_download": False,
        "notes": "Download all 15 Supplementary Data files from article page.",
    },
    "P04": {
        "short_title": "Epigenomics",
        "doi": "10.1038/s41467-024-48806-z",
        "article_url": "https://www.nature.com/articles/s41467-024-48806-z",
        "dir": "P04_epigenomics",
        "supp_count": 1,
        "supp_description": "Source Data: epigenomics and clonal hematopoiesis results",
        "auto_download": False,
        "notes": "Download Source Data from article page.",
    },
    "P05": {
        "short_title": "Microbiome",
        "doi": "10.1038/s41564-024-01635-8",
        "article_url": "https://www.nature.com/articles/s41564-024-01635-8",
        "dir": "P05_microbiome",
        "supp_count": 0,
        "supp_description": "Supplementary data files: microbial abundance, diversity",
        "auto_download": False,
        "notes": "Download supplementary files from article page.",
    },
    "P06": {
        "short_title": "Spatial Skin",
        "doi": "10.1038/s41467-024-48625-2",
        "article_url": "https://www.nature.com/articles/s41467-024-48625-2",
        "dir": "P06_spatial_skin",
        "supp_count": 3,
        "supp_description": "Supp Data 1 (9.5 MB DEGs), Supp Data 2 (7.3 MB pathways), Supp Data 3 (1.7 MB microbiome taxonomy), Source Data (26.6 MB)",
        "auto_download": False,
        "notes": "Download Supplementary Data 1-3 + Source Data from article page. Total ~45 MB.",
    },
    "P07": {
        "short_title": "Hemoglobin",
        "doi": "10.1038/s41467-024-49289-8",
        "article_url": "https://www.nature.com/articles/s41467-024-49289-8",
        "dir": "P07_hemoglobin",
        "supp_count": 1,
        "supp_description": "Source Data (2.3 MB): cross-mission hemoglobin expression, DE results, z-scores",
        "auto_download": False,
        "notes": "Download Source Data xlsx from article page. Contains I4+JAXA+Twins hemoglobin data.",
    },
    "P08": {
        "short_title": "JAXA CFE cfRNA",
        "doi": "10.1038/s41467-023-41995-z",
        "article_url": "https://www.nature.com/articles/s41467-023-41995-z",
        "dir": "P08_JAXA_CFE_cfRNA",
        "supp_count": 6,
        "supp_description": "Supp Data 1-6: DRRs, correlations, CD36 genes, tissue specificity, mouse DRRs",
        "auto_download": False,
        "notes": "Download Supplementary Data 1-6 from article page.",
    },
    "P09": {
        "short_title": "TERRA",
        "doi": "10.1038/s42003-024-06014-x",
        "article_url": "https://www.nature.com/articles/s42003-024-06014-x",
        "dir": "P09_TERRA",
        "supp_count": 5,
        "supp_description": "Supp Data 1-5: TERRA motifs (I4, Twins, Everest), simulated microgravity, in vitro stats",
        "auto_download": False,
        "notes": "Download Supplementary Data 1-5 from article page.",
    },
    "P10": {
        "short_title": "NASA Twins Study",
        "doi": "10.1126/science.aau8650",
        "article_url": "https://www.science.org/doi/10.1126/science.aau8650",
        "dir": "P10_twins_study",
        "supp_count": 0,
        "supp_description": "Supplementary tables from Science paper (aggregate results only)",
        "auto_download": False,
        "notes": "Download supplementary materials from Science article page. Use only aggregate/summary data (no individual-level).",
    },
}

# GEO datasets to fetch
GEO_DATASETS = {
    "GSE213808": {
        "description": "JAXA MHU-1 Mouse cfRNA (quantile-normalized counts)",
        "target_dir": "P08_JAXA_CFE_cfRNA",
        "files": [
            "GSE213808_TGB_022_3group_N6_totalcount_quantilenormalized.xlsx",
            "GSE213808_Mus_musculus_mm10_GeneAnnotation.xlsx",
        ],
    },
}


def download_file(url: str, outpath: Path, timeout: int = 120) -> bool:
    """Download a file from a URL."""
    try:
        resp = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = outpath.stat().st_size / 1024
        print(f"  Downloaded: {outpath.name} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def fetch_geo_data(geo_id: str, config: dict, dry_run: bool = False):
    """Fetch processed data from GEO."""
    print(f"\n  Fetching GEO: {geo_id} -- {config['description']}")
    target = SOURCES_DIR / config["target_dir"]

    for filename in config.get("files", []):
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_id[:7]}nnn/{geo_id}/suppl/{filename}"
        outpath = target / filename

        if dry_run:
            print(f"  [DRY RUN] Would download: {filename}")
        else:
            if not download_file(url, outpath):
                # Try alternative URL format
                alt_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={geo_id}&format=file&file={filename}"
                download_file(alt_url, outpath)
            time.sleep(1)


def generate_download_instructions():
    """Generate a manual download instruction file."""
    instructions = []
    instructions.append("# SpaceOmicsBench v2 -- Manual Download Instructions")
    instructions.append("")
    instructions.append("Nature/Science supplementary files require browser-based download.")
    instructions.append("Follow these steps for each paper:")
    instructions.append("")

    for paper_id, info in PAPERS.items():
        instructions.append(f"## {paper_id}: {info['short_title']}")
        instructions.append(f"")
        instructions.append(f"1. Open: {info['article_url']}")
        instructions.append(f"2. Scroll to 'Supplementary Information' section")
        instructions.append(f"3. Download: {info['supp_description']}")
        instructions.append(f"4. Save files to: sources/{info['dir']}/")
        if info.get("notes"):
            instructions.append(f"5. Note: {info['notes']}")
        instructions.append(f"")

    instructions.append("## GEO Datasets (automated)")
    instructions.append("")
    for geo_id, config in GEO_DATASETS.items():
        instructions.append(f"- {geo_id}: {config['description']}")
        instructions.append(f"  URL: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_id}")
    instructions.append("")

    return "\n".join(instructions)


def main():
    parser = argparse.ArgumentParser(description="Fetch supplementary data from papers")
    parser.add_argument(
        "--papers",
        type=str,
        default="all",
        help="Comma-separated paper IDs (e.g., P01,P02,P08) or 'all'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--instructions-only",
        action="store_true",
        help="Only generate the manual download instructions file",
    )
    args = parser.parse_args()

    print("SpaceOmicsBench v2 -- Supplementary Data Fetcher")
    print(f"Output directory: {SOURCES_DIR}")

    # Always generate instructions
    instructions = generate_download_instructions()
    instructions_path = BASE_DIR / "docs" / "DOWNLOAD_INSTRUCTIONS.md"
    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    instructions_path.write_text(instructions)
    print(f"\nGenerated: {instructions_path}")

    if args.instructions_only:
        print("\nInstructions file generated. Open it for manual download steps.")
        return

    # Select papers
    if args.papers == "all":
        selected = PAPERS
    else:
        ids = [p.strip() for p in args.papers.split(",")]
        selected = {k: v for k, v in PAPERS.items() if k in ids}

    # Ensure source directories exist
    for paper_id, info in selected.items():
        target = SOURCES_DIR / info["dir"]
        target.mkdir(parents=True, exist_ok=True)

    # Report what needs manual download
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOADS REQUIRED")
    print("=" * 60)
    for paper_id, info in selected.items():
        print(f"\n{paper_id}: {info['short_title']}")
        print(f"  URL: {info['article_url']}")
        print(f"  Data: {info['supp_description']}")
        print(f"  Save to: sources/{info['dir']}/")

    # Fetch GEO data (automated)
    print("\n" + "=" * 60)
    print("AUTOMATED GEO DOWNLOADS")
    print("=" * 60)
    for geo_id, config in GEO_DATASETS.items():
        fetch_geo_data(geo_id, config, dry_run=args.dry_run)

    print("\n\nDone! Check docs/DOWNLOAD_INSTRUCTIONS.md for manual download steps.")


if __name__ == "__main__":
    main()
