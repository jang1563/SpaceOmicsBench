#!/usr/bin/env python3
"""
signature_query.py — SpaceOmicsBench v2 Biological Signature Query Tool

Compare a new differential expression (DE) result against SpaceOmicsBench
reference signatures to identify overlap with known spaceflight responses.

Usage:
    python evaluation/signature_query.py --input my_de.csv [options]

Input CSV (minimum required columns):
    gene          : HGNC gene symbol
    log2FC        : log2 fold change (also accepted: logFC, log2FoldChange, FC)
    padj          : adjusted p-value (also accepted: adj_pval, FDR, adjusted_p_value)

Full DE results (all tested genes, not just DE) improve background estimates.

Reference signatures:
    I4_cfRNA_DRR          466 cfRNA spaceflight-responsive genes (JAXA IHSP DRR)
    I4_Plasma_Proteomics   57 plasma DE proteins (Inspiration4, adj_pval<0.05)
    I4_PBMC_CD4T          736 CD4+ T cell DE genes (Inspiration4)
    I4_PBMC_CD8T          661 CD8+ T cell DE genes (Inspiration4)
    I4_PBMC_CD14Mono      709 CD14+ Monocyte DE genes (Inspiration4)
    GeneLab_Mouse         134 rodent spaceflight DE genes (conserved with I4)
    JAXA_cfRNA             36 JAXA cfRNA DE genes
    CrossMission_Conserved 814 cross-mission conserved spaceflight genes

Metrics per signature:
    n_overlap           : # user DE genes found in signature
    overlap_coef        : n_overlap / min(|user_DE|, |signature|)
    jaccard             : n_overlap / |user_DE ∪ signature|
    hypergeom_p         : hypergeometric enrichment p-value
    hypergeom_q         : FDR-corrected p-value (Benjamini-Hochberg)
    direction_concordance: fraction of overlapping genes with same log2FC sign
    spearman_r          : Spearman correlation of log2FC (overlapping genes)
    spearman_p          : p-value for Spearman correlation

Caveats:
    - I4 crew n=4; signatures may not generalise to all spaceflight contexts.
    - cfRNA/JAXA: group-level DE, not individual-level.
    - Gene matching requires HGNC symbols; no alias resolution is performed.
    - Direction concordance assumes comparable biological comparison direction
      (user's spaceflight vs. pre-flight; reference signatures: spaceflight vs. pre-flight).
"""

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Reference signature definitions
# ---------------------------------------------------------------------------

SIGNATURES = [
    {
        "id": "I4_cfRNA_DRR",
        "description": "Inspiration4 cfRNA spaceflight-responsive genes (JAXA IHSP DRR set)",
        "mission": "Inspiration4 (JAXA IHSP)",
        "modality": "cfRNA / transcriptomics",
        "data_type": "gene",
        "load_fn": "_load_cfrna_drr",
        "n_expected": 466,
        "notes": "log2FC = edgeR pre-flight vs. in-flight (edge_pre_vs_flight_fc). "
                 "Negative = downregulated during spaceflight.",
    },
    {
        "id": "I4_Plasma_Proteomics",
        "description": "Inspiration4 plasma DE proteins (adj_pval < 0.05)",
        "mission": "Inspiration4",
        "modality": "plasma proteomics",
        "data_type": "protein",
        "load_fn": "_load_plasma_proteomics",
        "n_expected": 57,
        "notes": "limma logFC. Proteins only; gene symbol matching applies.",
    },
    {
        "id": "I4_PBMC_CD4T",
        "description": "Inspiration4 CD4+ T cell DE genes (adj_pval < 0.05)",
        "mission": "Inspiration4",
        "modality": "PBMC scRNA-seq",
        "data_type": "gene",
        "load_fn": "_load_pbmc",
        "load_kwargs": {"fc_col": "CD4_T", "padj_col": "CD4_T.padj"},
        "n_expected": 736,
        "notes": "Subset of 806 candidate conserved DEGs screened across cell types.",
    },
    {
        "id": "I4_PBMC_CD8T",
        "description": "Inspiration4 CD8+ T cell DE genes (adj_pval < 0.05)",
        "mission": "Inspiration4",
        "modality": "PBMC scRNA-seq",
        "data_type": "gene",
        "load_fn": "_load_pbmc",
        "load_kwargs": {"fc_col": "CD8_T", "padj_col": "CD8_T.padj"},
        "n_expected": 661,
        "notes": None,
    },
    {
        "id": "I4_PBMC_CD14Mono",
        "description": "Inspiration4 CD14+ Monocyte DE genes (adj_pval < 0.05)",
        "mission": "Inspiration4",
        "modality": "PBMC scRNA-seq",
        "data_type": "gene",
        "load_fn": "_load_pbmc",
        "load_kwargs": {"fc_col": "CD14_Mono", "padj_col": "CD14_Mono.padj"},
        "n_expected": 709,
        "notes": None,
    },
    {
        "id": "GeneLab_Mouse",
        "description": "GeneLab rodent spaceflight DE genes conserved with Inspiration4",
        "mission": "GeneLab (multi-study)",
        "modality": "bulk RNA-seq (mouse)",
        "data_type": "gene",
        "load_fn": "_load_pbmc",
        "load_kwargs": {"fc_col": "GeneLab", "padj_col": "GeneLab.padj"},
        "n_expected": 134,
        "notes": "Mouse-to-human ortholog mapping applied in preprocessing. "
                 "Cross-species comparisons should be interpreted cautiously.",
    },
    {
        "id": "JAXA_cfRNA",
        "description": "JAXA cfRNA DE genes (IHSP, padj < 0.05)",
        "mission": "JAXA IHSP",
        "modality": "cfRNA",
        "data_type": "gene",
        "load_fn": "_load_jaxa_cfrna",
        "n_expected": 36,
        "notes": "Group-level DE (no per-sample data). No directional FC available; "
                 "overlap only.",
    },
    {
        "id": "CrossMission_Conserved",
        "description": "Cross-mission conserved spaceflight genes (I4 + Twins + GeneLab)",
        "mission": "Multi-mission",
        "modality": "multi-omic",
        "data_type": "gene",
        "load_fn": "_load_cross_mission",
        "n_expected": 814,
        "notes": "Absolute FC only (mean_abs_log2fc); direction concordance not available.",
    },
]


# ---------------------------------------------------------------------------
# Signature loaders
# ---------------------------------------------------------------------------

def _load_cfrna_drr() -> pd.DataFrame:
    """Load I4 cfRNA DRR genes with directional log2FC from edgeR."""
    drr = pd.read_csv(_DATA_DIR / "cfrna_466drr.csv", usecols=["gene"])
    full = pd.read_csv(
        _DATA_DIR / "cfrna_3group_de_noleak.csv",
        usecols=["gene", "edge_pre_vs_flight_fc"],
    )
    merged = drr.merge(full, on="gene", how="left")
    merged = merged.rename(columns={"edge_pre_vs_flight_fc": "ref_log2fc"})
    merged = merged.dropna(subset=["gene"])
    merged["ref_log2fc"] = pd.to_numeric(merged["ref_log2fc"], errors="coerce")
    return merged[["gene", "ref_log2fc"]]


def _load_plasma_proteomics() -> pd.DataFrame:
    """Load I4 plasma DE proteins (adj_pval < 0.05)."""
    df = pd.read_csv(_DATA_DIR / "proteomics_plasma_de_clean.csv")
    de = df[df["adj_pval"] < 0.05][["gene", "logFC"]].copy()
    de = de.rename(columns={"logFC": "ref_log2fc"})
    de["ref_log2fc"] = pd.to_numeric(de["ref_log2fc"], errors="coerce")
    return de[["gene", "ref_log2fc"]]


def _load_pbmc(fc_col: str, padj_col: str) -> pd.DataFrame:
    """Load PBMC or GeneLab DE genes from gt_conserved_degs."""
    df = pd.read_csv(_DATA_DIR / "gt_conserved_degs.csv")
    de = df[df[padj_col] < 0.05][["human_gene", fc_col]].copy()
    de = de.rename(columns={"human_gene": "gene", fc_col: "ref_log2fc"})
    de["ref_log2fc"] = pd.to_numeric(de["ref_log2fc"], errors="coerce")
    return de[["gene", "ref_log2fc"]]


def _load_jaxa_cfrna() -> pd.DataFrame:
    """Load JAXA cfRNA DE genes (p-value based, no directional FC)."""
    df = pd.read_csv(_DATA_DIR / "gt_conserved_degs.csv")
    de = df[df["JAXA.cfRNA.pval"] < 0.05][["human_gene"]].copy()
    de = de.rename(columns={"human_gene": "gene"})
    de["ref_log2fc"] = np.nan  # no directional FC available
    return de[["gene", "ref_log2fc"]]


def _load_cross_mission() -> pd.DataFrame:
    """Load cross-mission conserved genes (label=1). FC is absolute, no direction."""
    df = pd.read_csv(_DATA_DIR / "cross_mission_gene_de.csv")
    conserved = df[df["label"] == 1][["gene", "mean_abs_log2fc"]].copy()
    conserved = conserved.rename(columns={"mean_abs_log2fc": "ref_log2fc"})
    # Mark as absolute (no directional info)
    conserved["ref_log2fc"] = np.nan  # set to NaN to skip direction concordance
    return conserved[["gene", "ref_log2fc"]]


def _dispatch_loader(sig_def: dict) -> pd.DataFrame:
    """Call the loader function specified in the signature definition."""
    fn_map = {
        "_load_cfrna_drr": _load_cfrna_drr,
        "_load_plasma_proteomics": _load_plasma_proteomics,
        "_load_pbmc": _load_pbmc,
        "_load_jaxa_cfrna": _load_jaxa_cfrna,
        "_load_cross_mission": _load_cross_mission,
    }
    fn = fn_map[sig_def["load_fn"]]
    kwargs = sig_def.get("load_kwargs", {})
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

_FC_CANDIDATES = ["log2fc", "log2foldchange", "logfc", "lfc", "fc", "log2_fc",
                  "log2fold_change", "fold_change", "log2foldchange"]
_PADJ_CANDIDATES = ["padj", "adj_pval", "adj_p_val", "fdr", "adjusted_p_value",
                    "adjusted_pvalue", "p_adj", "p.adj", "q_value", "qvalue",
                    "adj.p.val", "p.adjust"]
_GENE_CANDIDATES = ["gene", "gene_id", "gene_name", "gene_symbol", "hgnc_symbol",
                    "symbol", "geneid", "name"]


def _detect_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Find the first matching column (case-insensitive) in df."""
    col_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in col_lower:
            return col_lower[cand]
    raise ValueError(
        f"Cannot detect {label} column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def _hypergeom_pvalue(N: int, K: int, n: int, k: int) -> float:
    """
    Hypergeometric enrichment test.

    N: universe size (all tested genes)
    K: signature size (reference genes in universe)
    n: user DE gene count
    k: overlap count

    Returns p-value for observing >= k successes.
    """
    if k == 0:
        return 1.0
    rv = stats.hypergeom(N, K, n)
    return float(rv.sf(k - 1))  # P(X >= k) = 1 - P(X <= k-1)


def _bh_correction(pvalues: list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    arr = np.array(pvalues)
    order = np.argsort(arr)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, n + 1)
    qvals = arr * n / rank
    # Make monotone: q[i] = min(q[i:]) from right
    for i in range(n - 2, -1, -1):
        qvals[order[i]] = min(qvals[order[i]], qvals[order[i + 1]])
    return np.clip(qvals, 0, 1)


def compare_query_to_signature(
    query_genes: set[str],
    query_fc: dict[str, float],
    sig_df: pd.DataFrame,
    universe_size: int,
) -> dict:
    """
    Compute overlap statistics between user DE genes and one reference signature.

    Parameters
    ----------
    query_genes : set of user DE gene symbols
    query_fc    : {gene: log2fc} for user DE genes
    sig_df      : DataFrame with columns [gene, ref_log2fc]
    universe_size : total genes in user's input (background for hypergeom)

    Returns
    -------
    dict with keys: n_overlap, overlap_coef, jaccard, hypergeom_p,
                    direction_concordance, spearman_r, spearman_p,
                    has_direction, overlapping_genes
    """
    sig_genes = set(sig_df["gene"].dropna())
    sig_fc = dict(zip(sig_df["gene"], sig_df["ref_log2fc"]))

    N = universe_size
    K = len(sig_genes & set(query_fc.keys() | query_genes))
    # K = signature genes that were actually measured in user's background
    # If universe_size == len(query_genes), assume all sig genes in background
    # (conservative assumption when only DE genes are submitted)
    if N == len(query_genes):
        K = len(sig_genes)  # can't restrict; use full signature size

    n = len(query_genes)
    overlap = query_genes & sig_genes
    k = len(overlap)

    # Overlap metrics
    min_size = min(n, len(sig_genes))
    overlap_coef = k / min_size if min_size > 0 else 0.0
    union_size = len(query_genes | sig_genes)
    jaccard = k / union_size if union_size > 0 else 0.0

    # Hypergeometric test
    hypergeom_p = _hypergeom_pvalue(N, K, n, k)

    # Directional metrics (only when ref_log2fc is available)
    ref_fc_available = not sig_df["ref_log2fc"].isna().all()
    has_direction = ref_fc_available and len(query_fc) > 0

    direction_concordance = None
    spearman_r = None
    spearman_p = None

    if has_direction and k > 0:
        common = [g for g in overlap if g in query_fc and not np.isnan(sig_fc.get(g, np.nan))]
        if len(common) >= 2:
            user_signs = np.array([np.sign(query_fc[g]) for g in common])
            ref_signs = np.array([np.sign(sig_fc[g]) for g in common])
            direction_concordance = float(np.mean(user_signs == ref_signs))

            user_vals = np.array([query_fc[g] for g in common])
            ref_vals = np.array([sig_fc[g] for g in common])
            r, p = stats.spearmanr(user_vals, ref_vals)
            spearman_r = float(r) if not np.isnan(r) else None
            spearman_p = float(p) if not np.isnan(p) else None
        elif len(common) == 1:
            g = common[0]
            user_sign = np.sign(query_fc[g])
            ref_sign = np.sign(sig_fc[g])
            direction_concordance = 1.0 if user_sign == ref_sign else 0.0

    return {
        "n_sig_genes": len(sig_genes),
        "n_overlap": k,
        "overlap_coef": round(overlap_coef, 4),
        "jaccard": round(jaccard, 4),
        "hypergeom_p": hypergeom_p,
        "has_direction": has_direction,
        "direction_concordance": round(direction_concordance, 4) if direction_concordance is not None else None,
        "spearman_r": round(spearman_r, 4) if spearman_r is not None else None,
        "spearman_p": round(spearman_p, 4) if spearman_p is not None else None,
        "overlapping_genes": sorted(overlap),
    }


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_user_de(
    input_file: str,
    padj_threshold: float,
    fc_threshold: float,
    gene_col: Optional[str],
    fc_col: Optional[str],
    padj_col: Optional[str],
) -> tuple[pd.DataFrame, set, dict, int]:
    """
    Load and parse user DE results.

    Returns
    -------
    df_all        : full input DataFrame (all genes)
    query_genes   : set of DE gene symbols
    query_fc      : {gene: log2fc} for DE genes
    universe_size : total genes tested (len of df_all)
    """
    df = pd.read_csv(input_file)
    if df.empty:
        raise ValueError("Input file is empty.")

    # Detect columns
    gene_col = gene_col or _detect_column(df, _GENE_CANDIDATES, "gene")
    fc_col = fc_col or _detect_column(df, _FC_CANDIDATES, "log2FC")
    padj_col = padj_col or _detect_column(df, _PADJ_CANDIDATES, "padj")

    df = df.rename(columns={gene_col: "gene", fc_col: "log2fc", padj_col: "padj"})
    df["gene"] = df["gene"].astype(str).str.strip()
    df["log2fc"] = pd.to_numeric(df["log2fc"], errors="coerce")
    df["padj"] = pd.to_numeric(df["padj"], errors="coerce")
    df = df.dropna(subset=["gene"])

    # Remove duplicates: keep row with lowest padj
    df = df.sort_values("padj").drop_duplicates(subset=["gene"], keep="first")

    universe_size = len(df)

    # Apply DE filter
    de_mask = df["padj"] < padj_threshold
    if fc_threshold > 0:
        de_mask = de_mask & (df["log2fc"].abs() >= fc_threshold)

    de = df[de_mask].copy()
    query_genes = set(de["gene"])
    query_fc = dict(zip(de["gene"], de["log2fc"]))

    return df, query_genes, query_fc, universe_size


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _sig_bar(value: float, width: int = 20) -> str:
    """ASCII bar for overlap coefficient or concordance."""
    filled = int(round(value * width))
    return "[" + "=" * filled + " " * (width - filled) + "]"


def generate_markdown_report(
    input_file: str,
    n_query_all: int,
    n_query_de: int,
    padj_threshold: float,
    fc_threshold: float,
    results: list[dict],
    timestamp: str,
) -> str:
    """Generate a human-readable Markdown report."""
    lines = []
    lines.append("# SpaceOmicsBench Signature Query Report\n")
    lines.append(f"**Generated:** {timestamp}  ")
    lines.append(f"**Input:** `{Path(input_file).name}`  ")
    lines.append(f"**Total genes in input:** {n_query_all}  ")
    lines.append(f"**DE genes (padj < {padj_threshold}"
                 + (f", |log2FC| >= {fc_threshold}" if fc_threshold > 0 else "")
                 + f"):** {n_query_de}\n")

    lines.append("---\n")
    lines.append("## Overlap Summary\n")
    lines.append("| Signature | Mission | N sig | Overlap | Overlap Coef | "
                 "Jaccard | Hypergeom q | Direction Conc. | Spearman r |")
    lines.append("|-----------|---------|-------|---------|-------------|"
                 "--------|------------|-----------------|------------|")

    for r in results:
        def _fmt(v, fmt=".3f"):
            return format(v, fmt) if v is not None else "—"

        q_str = _fmt(r["hypergeom_q"])
        sig_q = " *" if (r["hypergeom_q"] is not None and r["hypergeom_q"] < 0.05) else ""
        lines.append(
            f"| **{r['id']}** | {r['mission']} | {r['n_sig_genes']} | "
            f"{r['n_overlap']} | {_fmt(r['overlap_coef'])} | "
            f"{_fmt(r['jaccard'])} | {q_str}{sig_q} | "
            f"{_fmt(r['direction_concordance'])} | "
            f"{_fmt(r['spearman_r'])} |"
        )

    lines.append("\n`*` FDR < 0.05 (significant enrichment)\n")

    # Detail sections for significant or notable overlaps
    notable = [r for r in results if r["n_overlap"] > 0]
    if notable:
        lines.append("---\n")
        lines.append("## Overlapping Genes by Signature\n")
        for r in notable:
            lines.append(f"### {r['id']}\n")
            lines.append(f"*{r['description']}*  ")
            lines.append(f"Overlap: **{r['n_overlap']}** / {r['n_sig_genes']} genes  ")
            if r["direction_concordance"] is not None:
                conc = r["direction_concordance"]
                lines.append(
                    f"Direction concordance: **{conc:.1%}** "
                    f"{_sig_bar(conc)}  "
                )
            if r["spearman_r"] is not None:
                p_str = f"{r['spearman_p']:.3f}" if r["spearman_p"] is not None else "n/a (n<3)"
                lines.append(
                    f"Spearman r = **{r['spearman_r']:.3f}** "
                    f"(p = {p_str})  "
                )
            if r["notes"]:
                lines.append(f"> Note: {r['notes']}  ")
            if r["overlapping_genes"]:
                gene_str = ", ".join(r["overlapping_genes"][:30])
                if len(r["overlapping_genes"]) > 30:
                    gene_str += f", … (+{len(r['overlapping_genes'])-30} more)"
                lines.append(f"\nGenes: {gene_str}\n")
            lines.append("")

    lines.append("---\n")
    lines.append("## Caveats\n")
    lines.append(
        textwrap.dedent("""\
        - **Small sample size**: Inspiration4 had n=4 crew members. Reference signatures
          reflect spaceflight responses for this specific mission and cohort.
        - **Group-level DE**: cfRNA and JAXA signatures are computed from group-level
          data (no per-sample counts). Statistical power is limited.
        - **Direction concordance**: Assumes your comparison direction matches the
          reference (spaceflight/post-flight vs. pre-flight). Reverse if needed.
        - **Gene symbol matching**: Only exact HGNC symbol matches are counted.
          Aliases, Ensembl IDs, or non-human symbols require prior conversion.
        - **CrossMission_Conserved**: Directional FC not available; Jaccard only.
        - This tool is for **exploratory overlap analysis**, not statistical validation.
        """)
    )
    lines.append(
        "\nFor citation information and source data, see: "
        "https://github.com/jang1563/SpaceOmicsBench\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare DE results against SpaceOmicsBench reference signatures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to user DE result CSV.",
    )
    parser.add_argument(
        "--padj-threshold", type=float, default=0.05,
        help="Adjusted p-value threshold for DE (default: 0.05).",
    )
    parser.add_argument(
        "--fc-threshold", type=float, default=0.0,
        help="Minimum |log2FC| threshold for DE (default: 0, disabled).",
    )
    parser.add_argument(
        "--gene-col", default=None,
        help="Name of gene symbol column in input (auto-detected if omitted).",
    )
    parser.add_argument(
        "--fc-col", default=None,
        help="Name of log2FC column in input (auto-detected if omitted).",
    )
    parser.add_argument(
        "--padj-col", default=None,
        help="Name of padj column in input (auto-detected if omitted).",
    )
    parser.add_argument(
        "--output-dir", "-o", default="results",
        help="Output directory for JSON and Markdown report (default: results/).",
    )
    parser.add_argument(
        "--signatures", nargs="+", default=None,
        help="Subset of signature IDs to query (default: all). "
             f"Options: {[s['id'] for s in SIGNATURES]}",
    )
    args = parser.parse_args()

    # ── Load user DE ──────────────────────────────────────────────────────
    print(f"Loading input: {args.input}")
    try:
        df_all, query_genes, query_fc, universe_size = load_user_de(
            input_file=args.input,
            padj_threshold=args.padj_threshold,
            fc_threshold=args.fc_threshold,
            gene_col=args.gene_col,
            fc_col=args.fc_col,
            padj_col=args.padj_col,
        )
    except Exception as e:
        print(f"ERROR loading input: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Total genes (background): {universe_size}")
    print(f"  DE genes (padj<{args.padj_threshold}"
          + (f", |FC|>={args.fc_threshold}" if args.fc_threshold else "")
          + f"): {len(query_genes)}")

    if len(query_genes) == 0:
        print("WARNING: No DE genes found with current thresholds. "
              "Check column names and thresholds.", file=sys.stderr)

    # ── Select signatures ─────────────────────────────────────────────────
    sig_defs = SIGNATURES
    if args.signatures:
        sig_ids = set(args.signatures)
        sig_defs = [s for s in SIGNATURES if s["id"] in sig_ids]
        missing = sig_ids - {s["id"] for s in sig_defs}
        if missing:
            print(f"WARNING: Unknown signature IDs ignored: {missing}", file=sys.stderr)

    # ── Compare ───────────────────────────────────────────────────────────
    raw_results = []
    for sig_def in sig_defs:
        print(f"  Querying signature: {sig_def['id']} ...", end=" ", flush=True)
        try:
            sig_df = _dispatch_loader(sig_def)
        except Exception as e:
            print(f"SKIP (load error: {e})")
            continue

        result = compare_query_to_signature(
            query_genes=query_genes,
            query_fc=query_fc,
            sig_df=sig_df,
            universe_size=universe_size,
        )
        result.update({
            "id": sig_def["id"],
            "description": sig_def["description"],
            "mission": sig_def["mission"],
            "modality": sig_def["modality"],
            "notes": sig_def.get("notes"),
        })
        raw_results.append(result)
        print(f"overlap={result['n_overlap']}/{result['n_sig_genes']}")

    # ── FDR correction across signatures ─────────────────────────────────
    pvalues = [r["hypergeom_p"] for r in raw_results]
    qvalues = _bh_correction(pvalues)
    for r, q in zip(raw_results, qvalues):
        r["hypergeom_q"] = round(float(q), 6)

    # Sort by overlap count descending
    raw_results.sort(key=lambda r: (-r["n_overlap"], r["id"]))

    # ── Output ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(args.input).stem

    # JSON
    json_data = {
        "query_info": {
            "input_file": args.input,
            "n_query_all": universe_size,
            "n_query_de": len(query_genes),
            "padj_threshold": args.padj_threshold,
            "fc_threshold": args.fc_threshold,
            "timestamp": timestamp,
        },
        "signature_comparisons": raw_results,
    }
    json_path = Path(args.output_dir) / f"signature_query_{stem}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")

    # Markdown
    md = generate_markdown_report(
        input_file=args.input,
        n_query_all=universe_size,
        n_query_de=len(query_genes),
        padj_threshold=args.padj_threshold,
        fc_threshold=args.fc_threshold,
        results=raw_results,
        timestamp=timestamp,
    )
    md_path = Path(args.output_dir) / f"signature_query_{stem}_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Report saved: {md_path}")

    # Console summary
    print("\n── Summary ──────────────────────────────────────")
    sig_hits = [(r["id"], r["n_overlap"], r["hypergeom_q"])
                for r in raw_results if r["n_overlap"] > 0]
    if sig_hits:
        for sid, n_ov, q in sig_hits:
            sig_mark = " **" if q < 0.05 else ""
            print(f"  {sid:<30} overlap={n_ov:3d}  FDR={q:.4f}{sig_mark}")
    else:
        print("  No overlapping genes found with any reference signature.")
    print()


if __name__ == "__main__":
    main()
