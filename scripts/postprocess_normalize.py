#!/usr/bin/env python3
"""
Post-process: normalize, clean, and create derived versions of preprocessed data.

1. Log-transform EVP proteomics and metabolomics matrices (raw intensity -> log10)
2. Create EVP NaN version (zeros -> NaN for missing-not-at-random)
3. Clean proteomics DE (remove contaminants, calibrants, isoform duplicates)

Output files:
  - data/processed/proteomics_evp_matrix_log10.csv: log10(x+1) transformed
  - data/processed/metabolomics_rppos_matrix_log10.csv: log10 transformed
  - data/processed/metabolomics_anppos_matrix_log10.csv: log10 transformed
  - data/processed/proteomics_evp_matrix_nan.csv: zeros replaced with NaN
  - data/processed/proteomics_plasma_de_clean.csv: contaminants/artifacts removed
  - data/processed/proteomics_evp_de_clean.csv: contaminants/artifacts removed
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

# Metadata columns to skip during transformation
META_COLS = {
    "sample_id", "crew", "month_tp", "timepoint", "timepoint_days",
    "phase", "mission", "tissue", "num", "sample_col",
    # Metabolomics annotation columns
    "SuperPathway", "SubPathway", "metabolite_name", "annotation_confidence",
    "Formula", "Mass", "RT", "CAS ID", "Mode", "KEGG", "HMDB",
}


def log10_transform(df: pd.DataFrame, pseudocount: float = 0) -> pd.DataFrame:
    """Log10-transform numeric columns, preserving metadata columns."""
    out = df.copy()
    for col in out.columns:
        if col not in META_COLS:
            vals = pd.to_numeric(out[col], errors="coerce")
            if pseudocount > 0:
                out[col] = np.log10(vals + pseudocount)
            else:
                # log10(x), NaN where x <= 0 or was NaN
                out[col] = np.log10(vals.where(vals > 0))
    return out


def zeros_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace exact zeros with NaN in numeric columns."""
    out = df.copy()
    for col in out.columns:
        if col not in META_COLS:
            vals = pd.to_numeric(out[col], errors="coerce")
            out[col] = vals.where(vals != 0)
    return out


def clean_de(df: pd.DataFrame) -> pd.DataFrame:
    """Remove contaminants, calibrants, and isoform duplicates from DE results."""
    gene_col = "gene"
    n_orig = len(df)

    # 1. Remove CON_ contaminants (MaxQuant format)
    con_mask = df[gene_col].str.startswith("CON_", na=False)
    n_con = con_mask.sum()

    # 2. Remove PEPCAL calibration peptides
    pepcal_mask = df[gene_col].str.startswith("PEPCAL", na=False)
    n_pepcal = pepcal_mask.sum()

    # 3. Remove isoform suffix entries (.1, .2, .3, etc.)
    isoform_mask = df[gene_col].str.match(r".*\.\d+$", na=False)
    n_isoform = isoform_mask.sum()

    # Apply all filters
    keep = ~(con_mask | pepcal_mask | isoform_mask)
    out = df[keep].reset_index(drop=True)

    n_removed = n_orig - len(out)
    print(f"    Removed: {n_con} CON_, {n_pepcal} PEPCAL, {n_isoform} isoforms = {n_removed} total")
    print(f"    Kept: {len(out)}/{n_orig}")

    return out


def main():
    print("=" * 60)
    print("SpaceOmicsBench v2 -- Post-processing & Normalization")
    print("=" * 60)

    # ---- 1. Log-transform EVP proteomics ----
    print("\n--- 1. EVP proteomics log10(x+1) ---")
    evp = pd.read_csv(PROC_DIR / "proteomics_evp_matrix.csv")
    print(f"  Input: {evp.shape[0]} samples × {evp.shape[1]} cols")

    evp_log = log10_transform(evp, pseudocount=1)
    # Verify range
    num_cols = [c for c in evp_log.columns if c not in META_COLS]
    vals = evp_log[num_cols].values.flatten()
    vals = vals[~np.isnan(vals)]
    print(f"  Log10(x+1) range: [{vals.min():.2f}, {vals.max():.2f}]")

    evp_log.to_csv(PROC_DIR / "proteomics_evp_matrix_log10.csv", index=False)
    print(f"  Saved: proteomics_evp_matrix_log10.csv")

    # ---- 2. Log-transform metabolomics RPPOS ----
    print("\n--- 2. Metabolomics RPPOS log10 ---")
    rppos = pd.read_csv(PROC_DIR / "metabolomics_rppos_matrix.csv")
    print(f"  Input: {rppos.shape[0]} samples × {rppos.shape[1]} cols")

    rppos_log = log10_transform(rppos, pseudocount=0)
    num_cols_rp = [c for c in rppos_log.columns if c not in META_COLS]
    vals_rp = rppos_log[num_cols_rp].values.flatten()
    vals_rp = vals_rp[~np.isnan(vals_rp)]
    print(f"  Log10 range: [{vals_rp.min():.2f}, {vals_rp.max():.2f}]")

    rppos_log.to_csv(PROC_DIR / "metabolomics_rppos_matrix_log10.csv", index=False)
    print(f"  Saved: metabolomics_rppos_matrix_log10.csv")

    # ---- 3. Log-transform metabolomics ANPPOS ----
    print("\n--- 3. Metabolomics ANPPOS log10 ---")
    anppos = pd.read_csv(PROC_DIR / "metabolomics_anppos_matrix.csv")
    print(f"  Input: {anppos.shape[0]} samples × {anppos.shape[1]} cols")

    anppos_log = log10_transform(anppos, pseudocount=0)
    num_cols_anp = [c for c in anppos_log.columns if c not in META_COLS]
    vals_anp = anppos_log[num_cols_anp].values.flatten()
    vals_anp = vals_anp[~np.isnan(vals_anp)]
    print(f"  Log10 range: [{vals_anp.min():.2f}, {vals_anp.max():.2f}]")

    anppos_log.to_csv(PROC_DIR / "metabolomics_anppos_matrix_log10.csv", index=False)
    print(f"  Saved: metabolomics_anppos_matrix_log10.csv")

    # ---- 4. EVP zeros -> NaN ----
    print("\n--- 4. EVP proteomics zeros -> NaN ---")
    evp_nan = zeros_to_nan(evp)
    num_cols_evp = [c for c in evp_nan.columns if c not in META_COLS]
    total_cells = evp_nan[num_cols_evp].size
    nan_cells = evp_nan[num_cols_evp].isna().sum().sum()
    pct_nan = nan_cells / total_cells * 100
    print(f"  Missing (NaN): {nan_cells}/{total_cells} ({pct_nan:.1f}%)")

    evp_nan.to_csv(PROC_DIR / "proteomics_evp_matrix_nan.csv", index=False)
    print(f"  Saved: proteomics_evp_matrix_nan.csv")

    # ---- 5. Clean plasma DE ----
    print("\n--- 5. Plasma proteomics DE cleanup ---")
    plasma_de = pd.read_csv(PROC_DIR / "proteomics_plasma_de.csv")
    print(f"  Input: {len(plasma_de)} proteins")
    plasma_de_clean = clean_de(plasma_de)

    # Verify significant hits preserved
    n_sig_orig = (plasma_de["adj_pval"].dropna() < 0.05).sum()
    n_sig_clean = (plasma_de_clean["adj_pval"].dropna() < 0.05).sum()
    print(f"  Significant (adj_pval<0.05): {n_sig_orig} -> {n_sig_clean}")

    plasma_de_clean.to_csv(PROC_DIR / "proteomics_plasma_de_clean.csv", index=False)
    print(f"  Saved: proteomics_plasma_de_clean.csv")

    # ---- 6. Clean EVP DE ----
    print("\n--- 6. EVP proteomics DE cleanup ---")
    evp_de = pd.read_csv(PROC_DIR / "proteomics_evp_de.csv")
    print(f"  Input: {len(evp_de)} proteins")
    evp_de_clean = clean_de(evp_de)

    n_sig_evp_orig = (evp_de["adj_pval"].dropna() < 0.05).sum()
    n_sig_evp_clean = (evp_de_clean["adj_pval"].dropna() < 0.05).sum()
    print(f"  Significant (adj_pval<0.05): {n_sig_evp_orig} -> {n_sig_evp_clean}")

    evp_de_clean.to_csv(PROC_DIR / "proteomics_evp_de_clean.csv", index=False)
    print(f"  Saved: proteomics_evp_de_clean.csv")

    # ---- 7. Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Log10-transformed matrices:")
    print(f"    EVP proteomics:    log10(x+1), range [{vals.min():.2f}, {vals.max():.2f}]")
    print(f"    RPPOS metabolomics: log10(x),  range [{vals_rp.min():.2f}, {vals_rp.max():.2f}]")
    print(f"    ANPPOS metabolomics: log10(x), range [{vals_anp.min():.2f}, {vals_anp.max():.2f}]")
    print(f"    (Plasma proteomics already log10 in source)")
    print()
    print("  EVP NaN version:")
    print(f"    {pct_nan:.1f}% missing (zeros -> NaN)")
    print()
    print("  Cleaned DE results:")
    print(f"    Plasma: {len(plasma_de)} -> {len(plasma_de_clean)} (removed {len(plasma_de)-len(plasma_de_clean)} artifacts)")
    print(f"    EVP:    {len(evp_de)} -> {len(evp_de_clean)} (removed {len(evp_de)-len(evp_de_clean)} artifacts)")
    print()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
