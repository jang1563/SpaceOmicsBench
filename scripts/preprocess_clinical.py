#!/usr/bin/env python3
"""
Preprocess clinical data for SpaceOmicsBench v2.

Reads TRANSFORMED CSVs from data/clinical/, extracts value/concentration columns,
normalizes sample IDs, adds metadata (crew, timepoint, phase), and outputs
clean matrices to data/processed/.

Input files (all from NASA GeneLab OSDR):
  - LSDS-7_Complete_Blood_Count_CBC_TRANSFORMED.csv (28 samples, 20 analytes)
  - LSDS-8_Comprehensive_Metabolic_Panel_CMP_TRANSFORMED.csv (28 samples, 19 analytes)
  - LSDS-8_Multiplex_serum_immune_EvePanel_TRANSFORMED.csv (28 samples, 71 markers)
  - LSDS-8_Multiplex_serum_cardiovascular_EvePanel_TRANSFORMED.csv (28 samples, 9 markers)
  - LSDS-8_Multiplex_serum.immune.AlamarPanel_TRANSFORMED.csv (27 samples, ~200 markers)
  - LSDS-64_Multiplex_urine.immune.AlamarPanel_TRANSFORMED.csv (22 samples, ~200 markers)

Output files:
  - data/processed/clinical_cbc.csv
  - data/processed/clinical_cmp.csv
  - data/processed/clinical_cytokines_eve.csv
  - data/processed/clinical_cardiovascular_eve.csv
  - data/processed/clinical_cytokines_alamar_serum.csv
  - data/processed/clinical_cytokines_alamar_urine.csv
  - data/processed/clinical_merged_serum.csv (CBC+CMP+Eve panels merged on crew+timepoint)
  - data/processed/clinical_metadata.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CLINICAL_DIR = BASE_DIR / "data" / "clinical"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# I4 mission: 3-day LEO flight
# Timepoints relative to launch (L-) and return (R+), in days
TIMEPOINT_DAYS = {
    "L-92": -92,
    "L-44": -44,
    "L-3": -3,
    "R+1": 1,
    "R+45": 45,
    "R+82": 82,
    "R+194": 194,
}

# Phase assignments following I4 literature convention
PHASE_MAP = {
    "L-92": "pre_flight",
    "L-44": "pre_flight",
    "L-3": "pre_flight",
    "R+1": "post_flight",
    "R+45": "recovery",
    "R+82": "recovery",
    "R+194": "recovery",
}


def parse_sample_name(name: str) -> dict:
    """Parse I4 sample name into components.

    Formats:
      CBC: C00X_whole-blood_L-Y_cbc  or  C00X_whole-blood_R+Y_cbc
      CMP/serum: C00X_serum_L-Y  or  C00X_serum_R+Y
      Urine: C00X_urine_L-Y  or  C00X_urine_R+Y
    """
    # Remove trailing _cbc suffix if present
    cleaned = re.sub(r"_cbc$", "", name)

    parts = cleaned.split("_")
    crew = parts[0]  # C001, C002, C003, C004
    tissue = parts[1]  # whole-blood, serum, urine

    # Timepoint is the rest joined (handles cases like L-3, R+194)
    timepoint = "_".join(parts[2:])

    assert crew in ("C001", "C002", "C003", "C004"), f"Unexpected crew: {crew} in {name}"
    assert tissue in ("whole-blood", "serum", "urine"), f"Unexpected tissue: {tissue} in {name}"
    assert timepoint in TIMEPOINT_DAYS, f"Unexpected timepoint: {timepoint} in {name}"

    return {
        "sample_id": f"{crew}_{timepoint}",
        "crew": crew,
        "tissue": tissue,
        "timepoint": timepoint,
        "timepoint_days": TIMEPOINT_DAYS[timepoint],
        "phase": PHASE_MAP[timepoint],
        "mission": "I4",
    }


def extract_cbc_values(df: pd.DataFrame) -> pd.DataFrame:
    """Extract value columns from CBC TRANSFORMED data."""
    value_cols = [c for c in df.columns if "_value_" in c]
    assert len(value_cols) == 20, f"Expected 20 CBC value columns, got {len(value_cols)}: {value_cols}"

    result = df[["Sample Name"] + value_cols].copy()

    # Clean column names: absolute_basophils_value_cells_per_microliter -> absolute_basophils
    rename = {}
    for col in value_cols:
        clean = re.sub(r"_value_.*$", "", col)
        rename[col] = f"cbc_{clean}"
    result = result.rename(columns=rename)

    return result


def extract_cmp_values(df: pd.DataFrame) -> pd.DataFrame:
    """Extract value columns from CMP TRANSFORMED data."""
    value_cols = [c for c in df.columns if "_value_" in c or "_value" in c]
    # Filter to only actual value columns (not range_min or max)
    value_cols = [c for c in value_cols if "range" not in c]
    assert len(value_cols) == 19, f"Expected 19 CMP value columns, got {len(value_cols)}: {value_cols}"

    result = df[["Sample Name"] + value_cols].copy()

    # Clean column names: albumin_value_gram_per_deciliter -> albumin
    rename = {}
    for col in value_cols:
        clean = re.sub(r"_value_.*$", "", col)
        clean = re.sub(r"_value$", "", clean)
        rename[col] = f"cmp_{clean}"
    result = result.rename(columns=rename)

    return result


def extract_concentration_cols(df: pd.DataFrame, id_col: str, prefix: str) -> pd.DataFrame:
    """Extract concentration columns from cytokine/multiplex panels."""
    conc_cols = [c for c in df.columns if "_concentration_" in c]
    assert len(conc_cols) > 0, f"No concentration columns found in {prefix}"

    result = df[[id_col] + conc_cols].copy()

    # Clean column names: 6ckine_concentration_picogram_per_milliliter -> 6ckine
    rename = {}
    for col in conc_cols:
        clean = re.sub(r"_concentration_.*$", "", col)
        rename[col] = f"{prefix}_{clean}"
    result = result.rename(columns=rename)

    return result


def build_metadata(samples: list[dict]) -> pd.DataFrame:
    """Build metadata DataFrame from parsed sample info."""
    meta = pd.DataFrame(samples)
    meta = meta.sort_values(["crew", "timepoint_days"]).reset_index(drop=True)
    return meta


def verify_output(df: pd.DataFrame, name: str, expected_rows: int, min_cols: int):
    """Verify output DataFrame meets expectations."""
    actual_rows = len(df)
    actual_cols = len(df.columns)
    assert actual_rows == expected_rows, (
        f"[{name}] Expected {expected_rows} rows, got {actual_rows}"
    )
    assert actual_cols >= min_cols, (
        f"[{name}] Expected >= {min_cols} columns, got {actual_cols}"
    )

    # Check for no all-NaN value columns (excluding metadata cols)
    meta_cols = {"sample_id", "crew", "tissue", "timepoint", "timepoint_days", "phase", "mission"}
    value_cols = [c for c in df.columns if c not in meta_cols]
    for col in value_cols:
        non_null = df[col].notna().sum()
        if non_null == 0:
            print(f"  WARNING [{name}]: Column '{col}' is entirely NaN", file=sys.stderr)

    print(f"  VERIFIED [{name}]: {actual_rows} rows × {actual_cols} cols")


def main():
    print("=" * 60)
    print("SpaceOmicsBench v2 -- Clinical Data Preprocessing")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Read all input files ----
    print("\n--- Reading input files ---")

    cbc_path = CLINICAL_DIR / "LSDS-7_Complete_Blood_Count_CBC_TRANSFORMED.csv"
    cmp_path = CLINICAL_DIR / "LSDS-8_Comprehensive_Metabolic_Panel_CMP_TRANSFORMED.csv"
    eve_immune_path = CLINICAL_DIR / "LSDS-8_Multiplex_serum_immune_EvePanel_TRANSFORMED.csv"
    eve_cardio_path = CLINICAL_DIR / "LSDS-8_Multiplex_serum_cardiovascular_EvePanel_TRANSFORMED.csv"
    alamar_serum_path = CLINICAL_DIR / "LSDS-8_Multiplex_serum.immune.AlamarPanel_TRANSFORMED.csv"
    alamar_urine_path = CLINICAL_DIR / "LSDS-64_Multiplex_urine.immune.AlamarPanel_TRANSFORMED.csv"

    for p in [cbc_path, cmp_path, eve_immune_path, eve_cardio_path, alamar_serum_path, alamar_urine_path]:
        assert p.exists(), f"Missing input file: {p}"
        print(f"  Found: {p.name}")

    cbc_raw = pd.read_csv(cbc_path)
    cmp_raw = pd.read_csv(cmp_path)
    eve_immune_raw = pd.read_csv(eve_immune_path)
    eve_cardio_raw = pd.read_csv(eve_cardio_path)
    alamar_serum_raw = pd.read_csv(alamar_serum_path)
    alamar_urine_raw = pd.read_csv(alamar_urine_path)

    print(f"\n  CBC:            {cbc_raw.shape[0]} samples × {cbc_raw.shape[1]} cols")
    print(f"  CMP:            {cmp_raw.shape[0]} samples × {cmp_raw.shape[1]} cols")
    print(f"  Eve immune:     {eve_immune_raw.shape[0]} samples × {eve_immune_raw.shape[1]} cols")
    print(f"  Eve cardio:     {eve_cardio_raw.shape[0]} samples × {eve_cardio_raw.shape[1]} cols")
    print(f"  Alamar serum:   {alamar_serum_raw.shape[0]} samples × {alamar_serum_raw.shape[1]} cols")
    print(f"  Alamar urine:   {alamar_urine_raw.shape[0]} samples × {alamar_urine_raw.shape[1]} cols")

    # ---- 2. Verify expected sample counts ----
    print("\n--- Verifying sample counts ---")
    assert cbc_raw.shape[0] == 28, f"CBC: expected 28 rows, got {cbc_raw.shape[0]}"
    assert cmp_raw.shape[0] == 28, f"CMP: expected 28 rows, got {cmp_raw.shape[0]}"
    assert eve_immune_raw.shape[0] == 28, f"Eve immune: expected 28, got {eve_immune_raw.shape[0]}"
    assert eve_cardio_raw.shape[0] == 28, f"Eve cardio: expected 28, got {eve_cardio_raw.shape[0]}"
    assert alamar_serum_raw.shape[0] == 27, f"Alamar serum: expected 27, got {alamar_serum_raw.shape[0]}"
    assert alamar_urine_raw.shape[0] == 22, f"Alamar urine: expected 22, got {alamar_urine_raw.shape[0]}"
    print("  All sample counts match expected values")

    # ---- 3. Parse sample names and build metadata ----
    print("\n--- Parsing sample names ---")

    # CBC uses "Sample Name" with _cbc suffix
    cbc_meta = [parse_sample_name(n) for n in cbc_raw["Sample Name"]]
    # CMP uses "Sample Name" without suffix
    cmp_meta = [parse_sample_name(n) for n in cmp_raw["Sample Name"]]
    # Eve panels use "Sample Name"
    eve_immune_meta = [parse_sample_name(n) for n in eve_immune_raw["Sample Name"]]
    eve_cardio_meta = [parse_sample_name(n) for n in eve_cardio_raw["Sample Name"]]
    # Alamar serum uses "Sample ID" (different column name!)
    alamar_serum_meta = [parse_sample_name(n) for n in alamar_serum_raw["Sample ID"]]
    # Alamar urine uses "Sample Name"
    alamar_urine_meta = [parse_sample_name(n) for n in alamar_urine_raw["Sample Name"]]

    print("  All sample names parsed successfully")

    # Verify CBC and CMP have identical sample sets (just different order possibly)
    cbc_ids = sorted([m["sample_id"] for m in cbc_meta])
    cmp_ids = sorted([m["sample_id"] for m in cmp_meta])
    eve_immune_ids = sorted([m["sample_id"] for m in eve_immune_meta])
    eve_cardio_ids = sorted([m["sample_id"] for m in eve_cardio_meta])
    assert cbc_ids == cmp_ids, "CBC and CMP sample IDs don't match"
    assert cbc_ids == eve_immune_ids, "CBC and Eve immune sample IDs don't match"
    assert cbc_ids == eve_cardio_ids, "CBC and Eve cardio sample IDs don't match"
    print("  CBC, CMP, Eve immune, Eve cardio all share identical 28 samples")

    # Check Alamar serum missing sample
    alamar_serum_ids = sorted([m["sample_id"] for m in alamar_serum_meta])
    missing_in_alamar = set(cbc_ids) - set(alamar_serum_ids)
    print(f"  Alamar serum missing: {missing_in_alamar}")

    # Check urine samples
    alamar_urine_ids = sorted([m["sample_id"] for m in alamar_urine_meta])
    urine_timepoints = sorted(set(m["timepoint"] for m in alamar_urine_meta))
    print(f"  Urine timepoints: {urine_timepoints}")
    print(f"  Urine samples: {len(alamar_urine_ids)}")

    # ---- 4. Extract value columns ----
    print("\n--- Extracting value columns ---")

    cbc_vals = extract_cbc_values(cbc_raw)
    print(f"  CBC: {len([c for c in cbc_vals.columns if c.startswith('cbc_')])} analytes")

    cmp_vals = extract_cmp_values(cmp_raw)
    print(f"  CMP: {len([c for c in cmp_vals.columns if c.startswith('cmp_')])} analytes")

    eve_immune_vals = extract_concentration_cols(eve_immune_raw, "Sample Name", "eve_immune")
    n_eve_immune = len([c for c in eve_immune_vals.columns if c.startswith("eve_immune_")])
    print(f"  Eve immune: {n_eve_immune} markers")

    eve_cardio_vals = extract_concentration_cols(eve_cardio_raw, "Sample Name", "eve_cardio")
    n_eve_cardio = len([c for c in eve_cardio_vals.columns if c.startswith("eve_cardio_")])
    print(f"  Eve cardio: {n_eve_cardio} markers")

    # Alamar panels: handle potential empty columns from CSV formatting issues
    # Drop unnamed/empty columns first
    alamar_serum_clean = alamar_serum_raw.loc[:, alamar_serum_raw.columns.notna()]
    alamar_serum_clean = alamar_serum_clean.loc[
        :, ~alamar_serum_clean.columns.str.startswith("Unnamed")
    ]
    alamar_serum_vals = extract_concentration_cols(alamar_serum_clean, "Sample ID", "alamar_serum")
    n_alamar_serum = len([c for c in alamar_serum_vals.columns if c.startswith("alamar_serum_")])
    print(f"  Alamar serum: {n_alamar_serum} markers")

    alamar_urine_clean = alamar_urine_raw.loc[:, alamar_urine_raw.columns.notna()]
    alamar_urine_clean = alamar_urine_clean.loc[
        :, ~alamar_urine_clean.columns.str.startswith("Unnamed")
    ]
    alamar_urine_vals = extract_concentration_cols(alamar_urine_clean, "Sample Name", "alamar_urine")
    n_alamar_urine = len([c for c in alamar_urine_vals.columns if c.startswith("alamar_urine_")])
    print(f"  Alamar urine: {n_alamar_urine} markers")

    # ---- 5. Add metadata and create output DataFrames ----
    print("\n--- Building output matrices ---")

    def add_metadata(vals_df: pd.DataFrame, meta_list: list[dict], id_col: str) -> pd.DataFrame:
        """Add metadata columns and use sample_id as index-friendly column."""
        meta_df = pd.DataFrame(meta_list)
        # Align by row position (same order as input)
        vals_df = vals_df.drop(columns=[id_col]).reset_index(drop=True)
        out = pd.concat([meta_df, vals_df], axis=1)
        return out

    cbc_out = add_metadata(cbc_vals, cbc_meta, "Sample Name")
    cmp_out = add_metadata(cmp_vals, cmp_meta, "Sample Name")
    eve_immune_out = add_metadata(eve_immune_vals, eve_immune_meta, "Sample Name")
    eve_cardio_out = add_metadata(eve_cardio_vals, eve_cardio_meta, "Sample Name")
    alamar_serum_out = add_metadata(alamar_serum_vals, alamar_serum_meta, "Sample ID")
    alamar_urine_out = add_metadata(alamar_urine_vals, alamar_urine_meta, "Sample Name")

    # Convert numeric columns - handle "NA", "N/A", empty strings
    for out_df in [cbc_out, cmp_out, eve_immune_out, eve_cardio_out, alamar_serum_out, alamar_urine_out]:
        meta_cols = {"sample_id", "crew", "tissue", "timepoint", "timepoint_days", "phase", "mission"}
        for col in out_df.columns:
            if col not in meta_cols:
                out_df[col] = pd.to_numeric(out_df[col], errors="coerce")

    # ---- Correct C003_L-92 CBC: NASA GeneLab source has column-shift error ----
    # Root cause: MCH row was dropped during GeneLab's merge, shifting all subsequent
    # values down by one row. Verified against original Quest Diagnostics lab reports at:
    #   CBC_Quest/2022_05_10_Eliah/Processed_Tables/C003_2021-06-16_*.csv
    print("\n--- Correcting C003_L-92 CBC (GeneLab column-shift error) ---")
    c003_mask = cbc_out["sample_id"] == "C003_L-92"
    assert c003_mask.sum() == 1, f"Expected 1 C003_L-92 row, got {c003_mask.sum()}"

    corrections = {
        "cbc_mch": 29.6,
        "cbc_mchc": 32.4,
        "cbc_rdw": 12.9,
        "cbc_platelet_count": 330.0,
        "cbc_mpv": 10.1,
        "cbc_absolute_neutrophils": 6826.0,
        "cbc_absolute_lymphocytes": 1997.0,
        "cbc_absolute_monocytes": 614.0,
        "cbc_absolute_eosinophils": 125.0,
        "cbc_absolute_basophils": 38.0,
    }
    for col, val in corrections.items():
        old_val = cbc_out.loc[c003_mask, col].values[0]
        cbc_out.loc[c003_mask, col] = val
        print(f"  {col}: {old_val} -> {val}")
    print("  C003_L-92 corrected from Quest Diagnostics original")

    # ---- 6. Build merged serum matrix ----
    print("\n--- Building merged serum matrix ---")

    # Merge CBC + CMP + Eve immune + Eve cardio on sample_id (28 samples)
    meta_cols_list = ["sample_id", "crew", "tissue", "timepoint", "timepoint_days", "phase", "mission"]

    # Get value columns only (not metadata) from each
    cbc_value_cols = [c for c in cbc_out.columns if c.startswith("cbc_")]
    cmp_value_cols = [c for c in cmp_out.columns if c.startswith("cmp_")]
    eve_immune_value_cols = [c for c in eve_immune_out.columns if c.startswith("eve_immune_")]
    eve_cardio_value_cols = [c for c in eve_cardio_out.columns if c.startswith("eve_cardio_")]

    # Start with CBC (tissue=whole-blood), set tissue to "blood" for merged
    merged = cbc_out[["sample_id"] + cbc_value_cols].copy()
    merged = merged.merge(
        cmp_out[["sample_id"] + cmp_value_cols],
        on="sample_id", how="inner"
    )
    merged = merged.merge(
        eve_immune_out[["sample_id"] + eve_immune_value_cols],
        on="sample_id", how="inner"
    )
    merged = merged.merge(
        eve_cardio_out[["sample_id"] + eve_cardio_value_cols],
        on="sample_id", how="inner"
    )

    # Add metadata from CMP (serum, but sample_id is crew+timepoint)
    meta_for_merged = pd.DataFrame(cbc_meta)[meta_cols_list].copy()
    meta_for_merged["tissue"] = "blood+serum"  # merged from different tissues
    merged = meta_for_merged.merge(merged, on="sample_id", how="inner")

    total_features = len(cbc_value_cols) + len(cmp_value_cols) + len(eve_immune_value_cols) + len(eve_cardio_value_cols)
    print(f"  Merged: {len(merged)} samples × {total_features} features")
    print(f"    CBC: {len(cbc_value_cols)}, CMP: {len(cmp_value_cols)}, "
          f"Eve immune: {len(eve_immune_value_cols)}, Eve cardio: {len(eve_cardio_value_cols)}")

    # ---- 7. Build metadata CSV ----
    print("\n--- Building metadata ---")
    all_meta = []
    for meta_list, tissue in [
        (cbc_meta, "whole-blood"),
        (cmp_meta, "serum"),
        (alamar_urine_meta, "urine"),
    ]:
        for m in meta_list:
            all_meta.append(m)

    # Deduplicate (CBC and CMP have same samples with different tissue labels)
    meta_df = pd.DataFrame(all_meta).drop_duplicates(subset=["crew", "timepoint", "tissue"])
    meta_df = meta_df.sort_values(["crew", "tissue", "timepoint_days"]).reset_index(drop=True)
    print(f"  Metadata: {len(meta_df)} unique crew×tissue×timepoint combinations")

    # ---- 8. Verify and save ----
    print("\n--- Verification ---")

    verify_output(cbc_out, "clinical_cbc", 28, 27)  # 7 meta + 20 values
    verify_output(cmp_out, "clinical_cmp", 28, 26)  # 7 meta + 19 values
    verify_output(eve_immune_out, "clinical_cytokines_eve", 28, 75)
    verify_output(eve_cardio_out, "clinical_cardiovascular_eve", 28, 16)
    verify_output(alamar_serum_out, "clinical_cytokines_alamar_serum", 27, 200)
    verify_output(alamar_urine_out, "clinical_cytokines_alamar_urine", 22, 200)
    verify_output(merged, "clinical_merged_serum", 28, 120)

    # Double-check: verify actual data values make sense
    print("\n--- Sanity checks on actual values ---")

    # CBC: WBC should be ~3.8-10.8 thousand/uL
    wbc_col = "cbc_white_blood_cell_count"
    wbc_vals = cbc_out[wbc_col].dropna()
    print(f"  WBC range: {wbc_vals.min():.1f} - {wbc_vals.max():.1f} (expected ~3.8-10.8 K/uL)")
    assert wbc_vals.min() > 0, "WBC has zero/negative values"
    assert wbc_vals.max() < 50, "WBC has implausibly high values"

    # CBC: Hemoglobin should be ~12-18 g/dL
    hgb_col = "cbc_hemoglobin"
    hgb_vals = cbc_out[hgb_col].dropna()
    print(f"  Hemoglobin range: {hgb_vals.min():.1f} - {hgb_vals.max():.1f} (expected ~12-18 g/dL)")
    assert hgb_vals.min() > 5, "Hemoglobin implausibly low"
    assert hgb_vals.max() < 25, "Hemoglobin implausibly high"

    # CMP: Glucose should be ~65-200 mg/dL
    gluc_col = "cmp_glucose"
    gluc_vals = cmp_out[gluc_col].dropna()
    print(f"  Glucose range: {gluc_vals.min():.0f} - {gluc_vals.max():.0f} (expected ~65-200 mg/dL)")
    assert gluc_vals.min() > 30, "Glucose implausibly low"
    assert gluc_vals.max() < 500, "Glucose implausibly high"

    # CMP: Sodium should be ~135-146 mmol/L
    na_col = "cmp_sodium"
    na_vals = cmp_out[na_col].dropna()
    print(f"  Sodium range: {na_vals.min():.0f} - {na_vals.max():.0f} (expected ~135-146 mmol/L)")

    # Check no duplicate sample_ids within each output
    for name, out_df in [
        ("cbc", cbc_out), ("cmp", cmp_out),
        ("eve_immune", eve_immune_out), ("eve_cardio", eve_cardio_out),
        ("alamar_serum", alamar_serum_out), ("alamar_urine", alamar_urine_out),
        ("merged", merged),
    ]:
        dups = out_df["sample_id"].duplicated().sum()
        assert dups == 0, f"[{name}] Found {dups} duplicate sample_ids"
    print("  No duplicate sample_ids in any output")

    # ---- 9. Save outputs ----
    print("\n--- Saving outputs ---")

    outputs = {
        "clinical_cbc.csv": cbc_out,
        "clinical_cmp.csv": cmp_out,
        "clinical_cytokines_eve.csv": eve_immune_out,
        "clinical_cardiovascular_eve.csv": eve_cardio_out,
        "clinical_cytokines_alamar_serum.csv": alamar_serum_out,
        "clinical_cytokines_alamar_urine.csv": alamar_urine_out,
        "clinical_merged_serum.csv": merged,
        "clinical_metadata.csv": meta_df,
    }

    for filename, df in outputs.items():
        outpath = OUTPUT_DIR / filename
        df.to_csv(outpath, index=False)
        size_kb = outpath.stat().st_size / 1024
        print(f"  Saved: {filename} ({size_kb:.1f} KB)")

    # ---- 10. Print summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Mission: Inspiration4 (I4), 3-day LEO flight")
    print(f"  Crew: C001, C002, C003, C004")
    print(f"  Timepoints: {sorted(TIMEPOINT_DAYS.keys(), key=lambda x: TIMEPOINT_DAYS[x])}")
    print(f"  Phases: pre_flight (L-92, L-44, L-3), post_flight (R+1), recovery (R+45, R+82, R+194)")
    print()
    print(f"  CBC:            28 samples × {len(cbc_value_cols)} analytes")
    print(f"  CMP:            28 samples × {len(cmp_value_cols)} analytes")
    print(f"  Eve immune:     28 samples × {n_eve_immune} markers")
    print(f"  Eve cardio:     28 samples × {n_eve_cardio} markers")
    print(f"  Alamar serum:   27 samples × {n_alamar_serum} markers (missing C003_L-44)")
    print(f"  Alamar urine:   22 samples × {n_alamar_urine} markers (no R+194; C001 missing L-3,R+1)")
    print(f"  Merged serum:   28 samples × {total_features} features")
    print()
    print(f"  Known data issues:")
    print(f"    - Alamar serum: C003_serum_L-44 missing (27/28 samples)")
    print(f"    - Alamar urine: no R+194 timepoint; C001 missing at L-3 and R+1 (22/28)")
    print(f"    - CMP: egfr range_max values are 'N/A' (not numeric)")
    print(f"    - CMP: bun_to_creatinine_ratio has NA values for some samples")
    print()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
