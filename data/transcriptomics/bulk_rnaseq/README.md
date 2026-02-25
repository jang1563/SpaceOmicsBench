# Bulk RNA-seq (Whole Blood)

Inspiration4 whole blood transcriptomics from multiple sequencing platforms.

## Sources
- **OSD-569**: Illumina, Oxford Nanopore (PromethION), Element Biosciences, Ultima Genomics
- GeneLab standard RNA-seq pipeline processing (DESeq2)

## Expected Files
- `OSD-569_normalized_counts.csv` -- DESeq2 normalized gene counts
- `OSD-569_VST_counts.csv` -- Variance-stabilized transform counts
- `OSD-569_differential_expression.csv` -- DE results (log2FC, padj)
- `OSD-569_contrasts.csv` -- Contrast definitions

## Processing
GeneLab bulk RNA-seq pipeline: STAR alignment -> RSEM quantification -> DESeq2 normalization + DE
