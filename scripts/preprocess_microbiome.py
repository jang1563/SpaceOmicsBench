#!/usr/bin/env python3
"""
Preprocess microbiome data for SpaceOmicsBench v2.

Input files:
  Human (GLDS-564, OSD-572): 327 samples, 4 crew × 10 body sites × multiple timepoints
  Human gut (GLDS-599, OSD-630): 8 samples, 2 crew (C002/C004) × 4 timepoints, assembly-based
  Environmental (GLDS-565, OSD-573): 39 samples, capsule surface swabs

Output files:
  - data/processed/microbiome_human_taxonomy_cpm.csv: human multi-site taxonomy (CPM)
  - data/processed/microbiome_human_pathways_cpm.csv: human pathway abundances (CPM)
  - data/processed/microbiome_human_metaphlan.csv: MetaPhlAn taxonomy profiles
  - data/processed/microbiome_gut_taxonomy_cpm.csv: gut-specific taxonomy
  - data/processed/microbiome_gut_pathways_cpm.csv: gut pathway abundances
  - data/processed/microbiome_env_taxonomy_cpm.csv: environmental taxonomy
  - data/processed/microbiome_env_pathways_cpm.csv: environmental pathways
  - data/processed/microbiome_metadata.csv: unified metadata
"""

import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
HUMAN_DIR = BASE_DIR / "data" / "microbiome" / "human"
ENV_DIR = BASE_DIR / "data" / "microbiome" / "environmental"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Body site abbreviations
BODY_SITES = {
    "ARM": "forearm",
    "EAR": "ear",
    "GLU": "gluteal_fold",
    "NAC": "nasal_cavity",
    "NAP": "nape",
    "ORC": "oral_cheek",
    "PIT": "axilla",
    "TZO": "toe_web",
    "UMB": "navel",
    "WEB": "toe_web_alt",
}

# Timepoint mappings
I4_TIMEPOINTS = {
    "L-92": {"days": -92, "phase": "pre_flight"},
    "L-44": {"days": -44, "phase": "pre_flight"},
    "FD2": {"days": 2, "phase": "in_flight"},
    "FD3": {"days": 3, "phase": "in_flight"},
    "R+1": {"days": 1, "phase": "post_flight"},
    "R+45": {"days": 45, "phase": "recovery"},
    "R+82": {"days": 82, "phase": "recovery"},
}


def parse_human_sample(col: str) -> dict | None:
    """Parse GLDS-564 sample name: C001_FD2_ARM or Communal_AllFlight_H20."""
    m = re.match(r"(C\d{3})_([^_]+)_(.+)$", col)
    if not m:
        return None  # Control/communal samples
    crew, tp, site = m.groups()
    if tp not in I4_TIMEPOINTS:
        return None
    # Exclude water controls (H20 = H2O) and unknown OAC samples
    if site in ("H20", "OAC"):
        return None
    return {
        "sample_col": col,
        "sample_id": f"{crew}_{tp}_{site}",
        "crew": crew,
        "timepoint": tp,
        "timepoint_days": I4_TIMEPOINTS[tp]["days"],
        "phase": I4_TIMEPOINTS[tp]["phase"],
        "body_site": site,
        "body_site_name": BODY_SITES.get(site, site),
        "source": "human",
        "mission": "I4",
    }


def parse_gut_sample(col: str) -> dict | None:
    """Parse GLDS-599 sample name: C002_L-44."""
    m = re.match(r"(C\d{3})_(.+)$", col)
    if not m:
        return None
    crew, tp = m.groups()
    if tp not in I4_TIMEPOINTS:
        return None
    return {
        "sample_col": col,
        "sample_id": f"{crew}_{tp}_stool",
        "crew": crew,
        "timepoint": tp,
        "timepoint_days": I4_TIMEPOINTS[tp]["days"],
        "phase": I4_TIMEPOINTS[tp]["phase"],
        "body_site": "stool",
        "body_site_name": "stool",
        "source": "human_gut",
        "mission": "I4",
    }


def parse_env_sample(col: str) -> dict | None:
    """Parse GLDS-565 sample name: I4_Capsule_FD2_0."""
    m = re.match(r"I4_Capsule_([^_]+)_(\d+)$", col)
    if not m:
        return None
    tp_raw, loc = m.groups()
    # Map tp_raw: FD2, FD3, LM44, LM92
    tp_map = {"FD2": "FD2", "FD3": "FD3", "LM44": "L-44", "LM92": "L-92"}
    tp = tp_map.get(tp_raw)
    if tp is None or tp not in I4_TIMEPOINTS:
        return None
    return {
        "sample_col": col,
        "sample_id": f"capsule_{tp}_{loc}",
        "crew": "capsule",
        "timepoint": tp,
        "timepoint_days": I4_TIMEPOINTS[tp]["days"],
        "phase": I4_TIMEPOINTS[tp]["phase"],
        "body_site": f"capsule_loc{loc}",
        "body_site_name": "dragon_capsule",
        "source": "environmental",
        "mission": "I4",
    }


def read_taxonomy_cpm(filepath: Path, sample_parser) -> tuple[pd.DataFrame, list[dict]]:
    """Read taxonomy CPM file and parse sample columns."""
    df = pd.read_csv(filepath, sep="\t")
    tax_cols = ["taxid", "domain", "phylum", "class", "order", "family", "genus", "species"]
    other_cols = [c for c in df.columns if c not in tax_cols]

    sample_meta = []
    valid_cols = []
    skipped = []
    for col in other_cols:
        info = sample_parser(col)
        if info:
            sample_meta.append(info)
            valid_cols.append(col)
        else:
            skipped.append(col)

    if skipped:
        print(f"    Skipped columns (controls/non-crew): {len(skipped)}")

    # Keep taxonomy cols + valid sample cols
    out = df[tax_cols + valid_cols]
    return out, sample_meta


def read_pathways_cpm(filepath: Path, sample_parser) -> tuple[pd.DataFrame, list[dict]]:
    """Read pathway CPM file."""
    df = pd.read_csv(filepath, sep="\t")
    pathway_col = df.columns[0]  # "# Pathway"
    other_cols = list(df.columns[1:])

    # Strip _Abundance-CPM suffix from column names (present in pathway files but not taxonomy)
    col_rename = {}
    for col in other_cols:
        clean = re.sub(r"_Abundance-CPM$", "", col)
        if clean != col:
            col_rename[col] = clean
    if col_rename:
        df = df.rename(columns=col_rename)
        other_cols = [col_rename.get(c, c) for c in other_cols]

    sample_meta = []
    valid_cols = []
    for col in other_cols:
        info = sample_parser(col)
        if info:
            sample_meta.append(info)
            valid_cols.append(col)

    out = df[[pathway_col] + valid_cols]
    out = out.rename(columns={pathway_col: "pathway"})
    return out, sample_meta


def read_metaphlan(filepath: Path, sample_parser) -> tuple[pd.DataFrame, list[dict]]:
    """Read MetaPhlAn taxonomy file (has comment header line)."""
    df = pd.read_csv(filepath, sep="\t", comment="#")
    # After comment skip, first col should be clade_name
    clade_col = df.columns[0]
    other_cols = list(df.columns[1:])

    sample_meta = []
    valid_cols = []
    for col in other_cols:
        info = sample_parser(col)
        if info:
            sample_meta.append(info)
            valid_cols.append(col)

    out = df[[clade_col] + valid_cols]
    out = out.rename(columns={clade_col: "clade"})
    return out, sample_meta


def main():
    print("=" * 60)
    print("SpaceOmicsBench v2 -- Microbiome Preprocessing")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metadata = []

    # ---- 1. Human multi-site microbiome (GLDS-564) ----
    print("\n--- Human multi-site microbiome (GLDS-564) ---")

    human_tax_path = HUMAN_DIR / "GLDS-564_GMetagenomics_Combined-contig-level-taxonomy-coverages-CPM_GLmetagenomics.tsv"
    human_path_path = HUMAN_DIR / "GLDS-564_GMetagenomics_Pathway-abundances-cpm_GLmetagenomics.tsv"
    human_mph_path = HUMAN_DIR / "GLDS-564_GMetagenomics_Metaphlan-taxonomy_GLmetagenomics.tsv"

    print("  Reading taxonomy CPM...")
    human_tax, human_tax_meta = read_taxonomy_cpm(human_tax_path, parse_human_sample)
    print(f"  Taxonomy: {human_tax.shape[0]} taxa × {len(human_tax_meta)} crew samples")

    print("  Reading pathway CPM...")
    human_path, _ = read_pathways_cpm(human_path_path, parse_human_sample)
    print(f"  Pathways: {human_path.shape[0]} pathways")

    print("  Reading MetaPhlAn...")
    human_mph, _ = read_metaphlan(human_mph_path, parse_human_sample)
    print(f"  MetaPhlAn: {human_mph.shape[0]} clades")

    all_metadata.extend(human_tax_meta)

    # Verify body sites
    body_sites_found = set(m["body_site"] for m in human_tax_meta)
    print(f"  Body sites: {sorted(body_sites_found)}")
    crews_found = set(m["crew"] for m in human_tax_meta)
    print(f"  Crews: {sorted(crews_found)}")

    # ---- 2. Human gut metagenomics (GLDS-599) ----
    print("\n--- Human gut metagenomics (GLDS-599) ---")

    gut_tax_path = HUMAN_DIR / "GLDS-599_GMetagenomics_Combined-contig-level-taxonomy-coverages-CPM_GLmetagenomics.tsv"
    gut_path_path = HUMAN_DIR / "GLDS-599_GMetagenomics_Pathway-abundances-cpm_GLmetagenomics.tsv"

    print("  Reading taxonomy CPM...")
    gut_tax, gut_tax_meta = read_taxonomy_cpm(gut_tax_path, parse_gut_sample)
    print(f"  Taxonomy: {gut_tax.shape[0]} taxa × {len(gut_tax_meta)} samples")

    print("  Reading pathway CPM...")
    gut_path, _ = read_pathways_cpm(gut_path_path, parse_gut_sample)
    print(f"  Pathways: {gut_path.shape[0]} pathways")

    all_metadata.extend(gut_tax_meta)

    # ---- 3. Environmental microbiome (GLDS-565) ----
    print("\n--- Environmental microbiome (GLDS-565) ---")

    env_tax_path = ENV_DIR / "GLDS-565_GMetagenomics_Combined-contig-level-taxonomy-coverages-CPM_GLmetagenomics.tsv"
    env_path_path = ENV_DIR / "GLDS-565_GMetagenomics_Pathway-abundances-cpm_GLmetagenomics.tsv"

    print("  Reading taxonomy CPM...")
    env_tax, env_tax_meta = read_taxonomy_cpm(env_tax_path, parse_env_sample)
    print(f"  Taxonomy: {env_tax.shape[0]} taxa × {len(env_tax_meta)} samples")

    print("  Reading pathway CPM...")
    env_path, _ = read_pathways_cpm(env_path_path, parse_env_sample)
    print(f"  Pathways: {env_path.shape[0]} pathways")

    all_metadata.extend(env_tax_meta)

    # ---- 4. Build unified metadata ----
    print("\n--- Building metadata ---")
    meta_df = pd.DataFrame(all_metadata)
    meta_df = meta_df.drop(columns=["sample_col"], errors="ignore")
    meta_df = meta_df.drop_duplicates(subset=["sample_id"])
    meta_df = meta_df.sort_values(["source", "crew", "body_site", "timepoint_days"]).reset_index(drop=True)
    print(f"  Total metadata records: {len(meta_df)}")
    for src in meta_df["source"].unique():
        n = len(meta_df[meta_df["source"] == src])
        print(f"    {src}: {n} samples")

    # ---- 5. Sanity checks ----
    print("\n--- Sanity checks ---")

    # All CPM values should be non-negative
    for name, df in [("human_tax", human_tax), ("gut_tax", gut_tax), ("env_tax", env_tax)]:
        tax_meta_cols = ["taxid", "domain", "phylum", "class", "order", "family", "genus", "species"]
        num_cols = [c for c in df.columns if c not in tax_meta_cols]
        vals = df[num_cols].values.flatten()
        vals = vals[~pd.isna(vals)]
        if len(vals) > 0:
            assert vals.min() >= 0, f"Negative CPM in {name}"
            print(f"  {name} CPM range: [{vals.min():.1f}, {vals.max():.1f}]")

    # Human should have 4 crews
    assert len(crews_found) == 4, f"Expected 4 crews, got {crews_found}"

    # ---- 6. Save ----
    print("\n--- Saving outputs ---")
    outputs = {
        "microbiome_human_taxonomy_cpm.csv": human_tax,
        "microbiome_human_pathways_cpm.csv": human_path,
        "microbiome_human_metaphlan.csv": human_mph,
        "microbiome_gut_taxonomy_cpm.csv": gut_tax,
        "microbiome_gut_pathways_cpm.csv": gut_path,
        "microbiome_env_taxonomy_cpm.csv": env_tax,
        "microbiome_env_pathways_cpm.csv": env_path,
        "microbiome_metadata.csv": meta_df,
    }
    for fname, df in outputs.items():
        path = OUTPUT_DIR / fname
        df.to_csv(path, index=False)
        size_kb = path.stat().st_size / 1024
        print(f"  Saved: {fname} ({size_kb:.1f} KB)")

    # ---- 7. Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Human multi-site (GLDS-564):")
    print(f"    {len(human_tax_meta)} crew samples, {len(body_sites_found)} body sites")
    print(f"    {human_tax.shape[0]} taxa, {human_path.shape[0]} pathways, {human_mph.shape[0]} MetaPhlAn clades")
    print("  Human gut (GLDS-599):")
    print(f"    {len(gut_tax_meta)} samples (C002, C004 × 4 timepoints)")
    print(f"    {gut_tax.shape[0]} taxa, {gut_path.shape[0]} pathways")
    print("  Environmental (GLDS-565):")
    print(f"    {len(env_tax_meta)} samples (Dragon capsule surface)")
    print(f"    {env_tax.shape[0]} taxa, {env_path.shape[0]} pathways")
    print()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
