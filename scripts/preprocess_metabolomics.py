#!/usr/bin/env python3
"""
Preprocess metabolomics data for SpaceOmicsBench v2.

Input files (from OSDR OSD-571 / GLDS-563):
  - Plasma_Metabolomics_Processed_Data.xlsx: limma DE (656 metabolites, 4 contrasts)
  - metabolomics_RPPOS-NEG_preprocessed_data.xlsx: 226 metabolites × 24 samples
  - metabolomics_ANPPOS-NEG_preprocessed_data.xlsx: 454 metabolites × 24 samples (with pathway annotations)

Output files:
  - data/processed/metabolomics_de.csv: limma DE for all 4 contrasts
  - data/processed/metabolomics_rppos_matrix.csv: RPPOS-NEG intensity matrix
  - data/processed/metabolomics_anppos_matrix.csv: ANPPOS-NEG intensity matrix with annotations
  - data/processed/metabolomics_metadata.csv: sample metadata
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MET_DIR = BASE_DIR / "data" / "metabolomics"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Metabolomics sample naming: MTBE_p_{crew}_{timepoint} or p_{crew}_{timepoint}
# crew: 1=C001, 2=C002, 3=C003, 4=C004
# timepoint: 1=June_Pre(L-92), 2=Aug_Pre(L-44), 3=Sept_Pre(L-3),
#            4=Sept_Post(R+1), 5=Nov_Post(R+45), 6=Dec_Post(R+82)
CREW_MAP = {1: "C001", 2: "C002", 3: "C003", 4: "C004"}
TP_MAP = {
    1: {"month": "June_Pre", "timepoint": "L-92", "days": -92, "phase": "pre_flight"},
    2: {"month": "Aug_Pre", "timepoint": "L-44", "days": -44, "phase": "pre_flight"},
    3: {"month": "Sept_Pre", "timepoint": "L-3", "days": -3, "phase": "pre_flight"},
    4: {"month": "Sept_Post", "timepoint": "R+1", "days": 1, "phase": "post_flight"},
    5: {"month": "Nov_Post", "timepoint": "R+45", "days": 45, "phase": "recovery"},
    6: {"month": "Dec_Post", "timepoint": "R+82", "days": 82, "phase": "recovery"},
}

# DE contrast names
CONTRASTS = {
    "I4-FP1": "(R+1) vs (L-92, L-44, L-3)",
    "I4-FP2": "(R+1, R+45, R+82) vs (L-92, L-44, L-3)",
    "I4-RP3": "(R+45, R+82) vs (R+1)",
    "I4-LP3": "(R+45, R+82) vs (L-92, L-44, L-3)",
}


def parse_sample_col(col_name: str) -> dict | None:
    """Parse sample column name like 'MTBE_p_1_1' or 'p_1_1' to crew+timepoint."""
    import re
    # Match patterns: MTBE_p_N_M or p_N_M
    m = re.search(r"p_(\d+)_(\d+)$", col_name)
    if not m:
        return None
    crew_num = int(m.group(1))
    tp_num = int(m.group(2))
    if crew_num not in CREW_MAP or tp_num not in TP_MAP:
        return None
    crew = CREW_MAP[crew_num]
    tp_info = TP_MAP[tp_num]
    return {
        "col": col_name,
        "crew": crew,
        "month_tp": tp_info["month"],
        "timepoint": tp_info["timepoint"],
        "timepoint_days": tp_info["days"],
        "phase": tp_info["phase"],
        "sample_id": f"{crew}_{tp_info['month']}",
    }


def main():
    print("=" * 60)
    print("SpaceOmicsBench v2 -- Metabolomics Preprocessing")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Read DE results (4 sheets) ----
    print("\n--- Reading limma DE results ---")
    de_path = MET_DIR / "GLDS-563_metabolomics_Plasma_Metabolomics_Processed_Data.xlsx"
    xl = pd.ExcelFile(de_path)
    print(f"  Sheets: {xl.sheet_names}")

    all_de = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(de_path, sheet_name=sheet, skiprows=6)
        assert "ID" in df.columns or "Gene" in df.columns, f"Sheet {sheet}: no ID/Gene column"
        first_col = df.columns[0]
        df = df.rename(columns={
            first_col: "metabolite",
            "P.Value": "pval",
            "adj.P.Val": "adj_pval",
        })
        df["contrast"] = sheet
        df["comparison"] = CONTRASTS.get(sheet, sheet)
        all_de.append(df)
        n_sig = (df["adj_pval"] < 0.05).sum()
        print(f"  {sheet}: {len(df)} metabolites, {n_sig} significant (adj_pval<0.05)")

    de_all = pd.concat(all_de, ignore_index=True)
    print(f"  Total DE rows: {len(de_all)} ({len(de_all) // len(xl.sheet_names)} metabolites × {len(xl.sheet_names)} contrasts)")

    # Verify all sheets have same number of metabolites
    counts = [len(d) for d in all_de]
    assert len(set(counts)) == 1, f"Sheets have different metabolite counts: {counts}"
    n_metabolites_de = counts[0]

    # ---- 2. Read RPPOS-NEG preprocessed data ----
    print("\n--- Reading RPPOS-NEG preprocessed data ---")
    rppos_path = MET_DIR / "GLDS-563_metabolomics_metabolomics_RPPOS-NEG_preprocessed_data.xlsx"
    rppos_raw = pd.read_excel(rppos_path)

    # Drop unnamed columns (Excel artifacts)
    rppos_raw = rppos_raw.loc[:, ~rppos_raw.columns.str.startswith("Unnamed")]
    print(f"  Shape after dropping unnamed cols: {rppos_raw.shape}")

    # Identify sample columns
    annotation_cols = ["Tentative  Name", "Assigned ID", "Confidence of metabolite annotation",
                       "Formula", "Mass", "RT", "CAS ID"]
    # Filter to only columns that exist
    annotation_cols = [c for c in annotation_cols if c in rppos_raw.columns]
    sample_cols = [c for c in rppos_raw.columns if c not in annotation_cols]

    # Parse sample columns
    sample_info = []
    valid_sample_cols = []
    for col in sample_cols:
        info = parse_sample_col(col)
        if info:
            sample_info.append(info)
            valid_sample_cols.append(col)
        else:
            print(f"  WARNING: Could not parse sample column '{col}'")

    print(f"  Annotation columns: {len(annotation_cols)}")
    print(f"  Sample columns: {len(valid_sample_cols)}")
    assert len(valid_sample_cols) == 24, f"Expected 24 sample columns, got {len(valid_sample_cols)}"

    # Build output: metabolite annotations + sample intensities with clean column names
    rppos_out = rppos_raw[annotation_cols].copy()
    rppos_out = rppos_out.rename(columns={
        "Tentative  Name": "metabolite_name",
        "Assigned ID": "metabolite_id",
        "Confidence of metabolite annotation": "annotation_confidence",
    })
    for info in sample_info:
        rppos_out[info["sample_id"]] = rppos_raw[info["col"]].values

    print(f"  RPPOS-NEG matrix: {len(rppos_out)} metabolites × {len(valid_sample_cols)} samples")

    # ---- 3. Read ANPPOS-NEG preprocessed data ----
    print("\n--- Reading ANPPOS-NEG preprocessed data ---")
    anppos_path = MET_DIR / "GLDS-563_metabolomics_metabolomics_ANPPOS-NEG_preprocessed_data.xlsx"
    anppos_raw = pd.read_excel(anppos_path)
    print(f"  Raw shape: {anppos_raw.shape}")

    # Annotation columns
    anp_annot_cols = ["SuperPathway", "SubPathway", "Compound Name",
                      "Confidence of metabolite annotation", "Formula", "Mass", "RT",
                      "CAS ID", "Mode", "KEGG", "HMB"]
    anp_annot_cols = [c for c in anp_annot_cols if c in anppos_raw.columns]

    anp_sample_cols = [c for c in anppos_raw.columns if c not in anp_annot_cols]
    anp_sample_info = []
    anp_valid_cols = []
    for col in anp_sample_cols:
        info = parse_sample_col(col)
        if info:
            anp_sample_info.append(info)
            anp_valid_cols.append(col)

    print(f"  Annotation columns: {len(anp_annot_cols)}")
    print(f"  Sample columns: {len(anp_valid_cols)}")
    assert len(anp_valid_cols) == 24, f"Expected 24 sample columns, got {len(anp_valid_cols)}"

    # Build output
    anppos_out = anppos_raw[anp_annot_cols].copy()
    anppos_out = anppos_out.rename(columns={
        "Compound Name": "metabolite_name",
        "Confidence of metabolite annotation": "annotation_confidence",
        "HMB": "HMDB",
    })
    for info in anp_sample_info:
        anppos_out[info["sample_id"]] = anppos_raw[info["col"]].values

    print(f"  ANPPOS-NEG matrix: {len(anppos_out)} metabolites × {len(anp_valid_cols)} samples")

    # Check pathway annotations
    n_with_pathway = anppos_out["SuperPathway"].notna().sum()
    unique_pathways = anppos_out["SuperPathway"].dropna().unique()
    print(f"  Metabolites with SuperPathway: {n_with_pathway}/{len(anppos_out)}")
    print(f"  Unique SuperPathways: {list(unique_pathways)}")

    # ---- 4. Build metadata ----
    print("\n--- Building metadata ---")
    meta_records = []
    for info in sample_info:
        meta_records.append({
            "sample_id": info["sample_id"],
            "crew": info["crew"],
            "month_tp": info["month_tp"],
            "timepoint": info["timepoint"],
            "timepoint_days": info["timepoint_days"],
            "phase": info["phase"],
            "mission": "I4",
            "tissue": "plasma",
        })
    meta_df = pd.DataFrame(meta_records).drop_duplicates()
    meta_df = meta_df.sort_values(["crew", "timepoint_days"]).reset_index(drop=True)
    print(f"  Metadata: {len(meta_df)} samples")

    # ---- 5. Sanity checks ----
    print("\n--- Sanity checks ---")

    # DE: check logFC ranges
    for contrast in CONTRASTS:
        subset = de_all[de_all["contrast"] == contrast]
        print(f"  {contrast} logFC: [{subset['logFC'].min():.2f}, {subset['logFC'].max():.2f}]")

    # RPPOS-NEG: intensities should be positive
    rppos_sample_ids = [info["sample_id"] for info in sample_info]
    rppos_vals = rppos_out[rppos_sample_ids].values.flatten()
    rppos_vals = rppos_vals[~pd.isna(rppos_vals)]
    print(f"  RPPOS-NEG intensity range: [{rppos_vals.min():.0f}, {rppos_vals.max():.0f}]")
    assert rppos_vals.min() >= 0, "Negative RPPOS intensity"

    # ANPPOS-NEG: intensities should be positive
    anp_sample_ids = [info["sample_id"] for info in anp_sample_info]
    anp_vals = anppos_out[anp_sample_ids].values.flatten()
    anp_vals = anp_vals[~pd.isna(anp_vals)]
    print(f"  ANPPOS-NEG intensity range: [{anp_vals.min():.0f}, {anp_vals.max():.0f}]")

    # Verify no duplicate sample IDs in metadata
    assert meta_df["sample_id"].duplicated().sum() == 0

    # ---- 6. Save ----
    print("\n--- Saving outputs ---")
    outputs = {
        "metabolomics_de.csv": de_all,
        "metabolomics_rppos_matrix.csv": rppos_out,
        "metabolomics_anppos_matrix.csv": anppos_out,
        "metabolomics_metadata.csv": meta_df,
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
    print(f"  DE results: {n_metabolites_de} metabolites × {len(CONTRASTS)} contrasts")
    print(f"  RPPOS-NEG: {len(rppos_out)} metabolites × 24 samples")
    print(f"  ANPPOS-NEG: {len(anppos_out)} metabolites × 24 samples (with pathway annotations)")
    print(f"  Samples: 24 (4 crew × 6 timepoints)")
    print(f"  SuperPathways: {list(unique_pathways)}")
    print()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
