#!/usr/bin/env python3
"""
Preprocess proteomics data for SpaceOmicsBench v2.

Input files (from OSDR OSD-571 / GLDS-563):
  - Plasma_Proteomics_Processed_Data.xlsx: limma DE (1,765 proteins)
  - EVP_Proteomics_Processed_Data.xlsx: limma DE (527 proteins)
  - plasma_proteomics_preprocessed_data.tsv: per-sample intensities (long, 42 samples with tech replicates)
  - EVPs_proteomics_preprocessed_data.txt: per-sample intensities (wide, 24+5 control samples)
  - plasma_metadata_all_samples_collapsed.csv: 21 samples
  - EVPs_sample_metadata.csv: 29 samples

Output files:
  - data/processed/proteomics_plasma_de.csv: limma DE for plasma (1,765 proteins)
  - data/processed/proteomics_evp_de.csv: limma DE for EVP (527 proteins)
  - data/processed/proteomics_plasma_matrix.csv: sample × protein intensity (21 samples, tech reps averaged)
  - data/processed/proteomics_evp_matrix.csv: sample × protein intensity (24 crew samples, no controls)
  - data/processed/proteomics_metadata.csv: unified sample metadata
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROT_DIR = BASE_DIR / "data" / "proteomics"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Map proteomics month-based timepoints to I4 study days
# I4 launched Sept 15, 2021; returned Sept 18, 2021
TIMEPOINT_MAP = {
    "June_Pre": {"timepoint": "L-92", "days": -92, "phase": "pre_flight"},
    "Aug_Pre": {"timepoint": "L-44", "days": -44, "phase": "pre_flight"},
    "Sept_Pre": {"timepoint": "L-3", "days": -3, "phase": "pre_flight"},
    "Sept_Post": {"timepoint": "R+1", "days": 1, "phase": "post_flight"},
    "Nov_Post": {"timepoint": "R+45", "days": 45, "phase": "recovery"},
    "Dec_Post": {"timepoint": "R+82", "days": 82, "phase": "recovery"},
}

# EVP sample number to timepoint
EVP_NUM_TO_TIMEPOINT = {
    1: "June_Pre",
    2: "Aug_Pre",
    3: "Sept_Pre",
    4: "Sept_Post",
    5: "Nov_Post",
    6: "Dec_Post",
}


def read_processed_de(xlsx_path: Path) -> pd.DataFrame:
    """Read limma DE results from processed xlsx (has 6 metadata header rows)."""
    df = pd.read_excel(xlsx_path, skiprows=6)
    # First column should be Gene/ID
    assert df.columns[0] in ("Gene", "ID"), f"Unexpected first column: {df.columns[0]}"
    assert "logFC" in df.columns, "Missing logFC column"
    assert "adj.P.Val" in df.columns, "Missing adj.P.Val column"
    # Rename for consistency
    df = df.rename(columns={
        df.columns[0]: "gene",
        "P.Value": "pval",
        "adj.P.Val": "adj_pval",
    })
    return df


def main():
    print("=" * 60)
    print("SpaceOmicsBench v2 -- Proteomics Preprocessing")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Read DE results ----
    print("\n--- Reading limma DE results ---")
    plasma_de = read_processed_de(PROT_DIR / "GLDS-563_proteomics_Plasma_Proteomics_Processed_Data.xlsx")
    evp_de = read_processed_de(PROT_DIR / "GLDS-563_proteomics_EVP_Proteomics_Processed_Data.xlsx")
    print(f"  Plasma DE: {plasma_de.shape[0]} proteins × {plasma_de.shape[1]} cols")
    print(f"  EVP DE: {evp_de.shape[0]} proteins × {evp_de.shape[1]} cols")

    assert plasma_de.shape[0] == 1765, f"Expected 1765 plasma proteins, got {plasma_de.shape[0]}"
    assert evp_de.shape[0] == 527, f"Expected 527 EVP proteins, got {evp_de.shape[0]}"

    # ---- 2. Read metadata ----
    print("\n--- Reading metadata ---")
    plasma_meta = pd.read_csv(PROT_DIR / "GLDS-563_proteomics_plasma_metadata_all_samples_collapsed.csv")
    evp_meta = pd.read_csv(PROT_DIR / "GLDS-563_proteomics_EVPs_sample_metadata.csv")
    print(f"  Plasma metadata: {len(plasma_meta)} samples")
    print(f"  EVP metadata: {len(evp_meta)} samples (including controls)")

    assert len(plasma_meta) == 21
    assert len(evp_meta) == 29  # 24 crew + 5 controls

    # ---- 3. Process plasma preprocessed data (long format → wide) ----
    print("\n--- Processing plasma preprocessed data ---")
    plasma_raw = pd.read_csv(PROT_DIR / "GLDS-563_proteomics_plasma_proteomics_preprocessed_data.tsv", sep="\t")
    print(f"  Raw long format: {len(plasma_raw)} rows")

    # Parse sample name: C001_Aug_Pre-1 → crew=C001, timepoint=Aug_Pre, plate=1
    plasma_raw["crew"] = plasma_raw["Sample Name"].str.extract(r"(C\d+)_")[0]
    plasma_raw["month_tp"] = plasma_raw["Sample Name"].str.extract(r"C\d+_(.+)-\d+$")[0]
    plasma_raw["plate"] = plasma_raw["Sample Name"].str.extract(r"-(\d+)$")[0]

    # Verify parsing
    assert plasma_raw["crew"].notna().all(), "Failed to parse crew from some sample names"
    assert plasma_raw["month_tp"].notna().all(), "Failed to parse timepoint from some sample names"

    n_samples = plasma_raw[["crew", "month_tp"]].drop_duplicates().shape[0]
    print(f"  Unique biological samples: {n_samples}")
    assert n_samples == 21

    # Average technical replicates (plate 1 and 2)
    plasma_avg = (
        plasma_raw
        .groupby(["crew", "month_tp", "Gene Names"])["Intensity (Log10)"]
        .mean()
        .reset_index()
    )

    # Pivot to wide: rows=samples, columns=proteins
    plasma_wide = plasma_avg.pivot_table(
        index=["crew", "month_tp"],
        columns="Gene Names",
        values="Intensity (Log10)",
    ).reset_index()
    plasma_wide.columns.name = None

    # Add metadata
    plasma_wide["sample_id"] = plasma_wide["crew"] + "_" + plasma_wide["month_tp"]
    for _, row in plasma_wide.iterrows():
        tp_info = TIMEPOINT_MAP.get(row["month_tp"])
        if tp_info is None:
            print(f"  WARNING: Unknown timepoint {row['month_tp']} for {row['crew']}")

    # Build metadata columns
    plasma_wide["timepoint"] = plasma_wide["month_tp"].map(lambda x: TIMEPOINT_MAP[x]["timepoint"])
    plasma_wide["timepoint_days"] = plasma_wide["month_tp"].map(lambda x: TIMEPOINT_MAP[x]["days"])
    plasma_wide["phase"] = plasma_wide["month_tp"].map(lambda x: TIMEPOINT_MAP[x]["phase"])
    plasma_wide["mission"] = "I4"
    plasma_wide["tissue"] = "plasma"

    # Reorder columns: metadata first, then proteins
    meta_cols = ["sample_id", "crew", "month_tp", "timepoint", "timepoint_days", "phase", "mission", "tissue"]
    protein_cols = sorted([c for c in plasma_wide.columns if c not in meta_cols])
    plasma_matrix = plasma_wide[meta_cols + protein_cols]

    n_proteins = len(protein_cols)
    print(f"  Plasma matrix: {len(plasma_matrix)} samples × {n_proteins} proteins")

    # ---- 4. Process EVP preprocessed data (wide format) ----
    print("\n--- Processing EVP preprocessed data ---")
    evp_raw = pd.read_csv(PROT_DIR / "GLDS-563_proteomics_EVPs_proteomics_preprocessed_data.txt", sep="\t")
    print(f"  Raw wide format: {evp_raw.shape[0]} proteins × {evp_raw.shape[1]} cols")

    # Strip leading spaces from column names
    evp_raw.columns = [c.strip() for c in evp_raw.columns]

    # Identify crew sample columns (C001_1 through C004_6)
    crew_cols = [c for c in evp_raw.columns if c.startswith("C00")]
    ctrl_cols = [c for c in evp_raw.columns if c.startswith("CTRL") or c.startswith("000")]
    print(f"  Crew sample columns: {len(crew_cols)}")
    print(f"  Control columns: {len(ctrl_cols)}")
    assert len(crew_cols) == 24, f"Expected 24 crew cols, got {len(crew_cols)}"

    # Extract gene names from Description (format: "ProteinName OS=Homo sapiens ... [GENE_HUMAN]")
    # Use Accession as primary ID since Description may not have clean gene names
    evp_clean = evp_raw[["Accession", "Description"] + crew_cols].copy()

    # Transpose to get samples × proteins
    evp_t = evp_clean.set_index("Accession")[crew_cols].T
    evp_t.index.name = "sample_col"
    evp_t = evp_t.reset_index()

    # Parse sample column names: C001_1 → crew=C001, num=1
    evp_t["crew"] = evp_t["sample_col"].str.extract(r"(C\d+)_")[0]
    evp_t["num"] = evp_t["sample_col"].str.extract(r"C\d+_(\d+)$")[0].astype(int)
    evp_t["month_tp"] = evp_t["num"].map(EVP_NUM_TO_TIMEPOINT)
    evp_t["sample_id"] = evp_t["crew"] + "_" + evp_t["month_tp"]
    evp_t["timepoint"] = evp_t["month_tp"].map(lambda x: TIMEPOINT_MAP[x]["timepoint"])
    evp_t["timepoint_days"] = evp_t["month_tp"].map(lambda x: TIMEPOINT_MAP[x]["days"])
    evp_t["phase"] = evp_t["month_tp"].map(lambda x: TIMEPOINT_MAP[x]["phase"])
    evp_t["mission"] = "I4"
    evp_t["tissue"] = "EVP"

    evp_meta_cols = ["sample_id", "crew", "month_tp", "timepoint", "timepoint_days", "phase", "mission", "tissue"]
    evp_protein_cols = sorted([c for c in evp_t.columns if c not in evp_meta_cols + ["sample_col", "num"]])
    evp_matrix = evp_t[evp_meta_cols + evp_protein_cols]

    print(f"  EVP matrix: {len(evp_matrix)} samples × {len(evp_protein_cols)} proteins (accessions)")

    # ---- 5. Build unified metadata ----
    print("\n--- Building metadata ---")
    meta_records = []
    for _, row in plasma_matrix[["sample_id", "crew", "month_tp", "timepoint", "timepoint_days", "phase"]].iterrows():
        meta_records.append({**row.to_dict(), "tissue": "plasma", "mission": "I4"})
    for _, row in evp_matrix[["sample_id", "crew", "month_tp", "timepoint", "timepoint_days", "phase"]].iterrows():
        meta_records.append({**row.to_dict(), "tissue": "EVP", "mission": "I4"})
    meta_df = pd.DataFrame(meta_records).drop_duplicates(subset=["crew", "month_tp", "tissue"])
    meta_df = meta_df.sort_values(["tissue", "crew", "timepoint_days"]).reset_index(drop=True)
    print(f"  Metadata: {len(meta_df)} records")

    # ---- 6. Sanity checks ----
    print("\n--- Sanity checks ---")

    # Plasma DE: check logFC range
    print(f"  Plasma DE logFC range: [{plasma_de['logFC'].min():.2f}, {plasma_de['logFC'].max():.2f}]")
    print(f"  Plasma DE sig (adj_pval<0.05): {(plasma_de['adj_pval'] < 0.05).sum()} proteins")

    # EVP DE: check logFC range
    print(f"  EVP DE logFC range: [{evp_de['logFC'].min():.2f}, {evp_de['logFC'].max():.2f}]")
    print(f"  EVP DE sig (adj_pval<0.05): {(evp_de['adj_pval'] < 0.05).sum()} proteins")

    # Plasma matrix: intensities should be log10, typically 0-25 range
    intensity_cols = [c for c in plasma_matrix.columns if c not in meta_cols]
    all_vals = plasma_matrix[intensity_cols].values.flatten()
    all_vals = all_vals[~pd.isna(all_vals)]
    print(f"  Plasma intensity range: [{all_vals.min():.2f}, {all_vals.max():.2f}] (log10)")
    assert all_vals.min() >= 0, "Negative log10 intensity"
    assert all_vals.max() < 30, "Implausibly high log10 intensity"

    # No duplicate sample_ids
    assert plasma_matrix["sample_id"].duplicated().sum() == 0, "Duplicate plasma sample IDs"
    assert evp_matrix["sample_id"].duplicated().sum() == 0, "Duplicate EVP sample IDs"
    print("  No duplicate sample IDs")

    # C002 missing Sept_Pre in plasma
    c002_tps = plasma_matrix[plasma_matrix["crew"] == "C002"]["month_tp"].tolist()
    assert "Sept_Pre" not in c002_tps, "C002 should not have Sept_Pre"
    print(f"  C002 plasma timepoints: {c002_tps} (Sept_Pre missing as expected)")

    # ---- 7. Save ----
    print("\n--- Saving outputs ---")
    outputs = {
        "proteomics_plasma_de.csv": plasma_de,
        "proteomics_evp_de.csv": evp_de,
        "proteomics_plasma_matrix.csv": plasma_matrix,
        "proteomics_evp_matrix.csv": evp_matrix,
        "proteomics_metadata.csv": meta_df,
    }
    for fname, df in outputs.items():
        path = OUTPUT_DIR / fname
        df.to_csv(path, index=False)
        size_kb = path.stat().st_size / 1024
        print(f"  Saved: {fname} ({size_kb:.1f} KB)")

    # ---- 8. Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Plasma proteomics:")
    print(f"    DE results: {len(plasma_de)} proteins (limma, R+1 vs pre-flight)")
    print(f"    Expression matrix: {len(plasma_matrix)} samples × {n_proteins} proteins (log10 intensity)")
    print(f"    Samples: 21 (4 crew × 5-6 timepoints, C002 missing Sept_Pre)")
    print("  EVP proteomics:")
    print(f"    DE results: {len(evp_de)} proteins (limma, R+1 vs pre-flight)")
    print(f"    Expression matrix: {len(evp_matrix)} samples × {len(evp_protein_cols)} proteins")
    print(f"    Samples: 24 (4 crew × 6 timepoints)")
    print()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
