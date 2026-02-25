# Cell-Free RNA (cfRNA)

Plasma cell-free RNA from JAXA CFE astronauts and Inspiration4 crew.

## Sources
- **OSD-530**: JAXA CFE human cfRNA (6 astronauts, 11 timepoints, normalized counts) -- OPEN
- **OSD-532**: JAXA MHU-1 mouse cfRNA (raw + processed) -- OPEN
- **GEO GSE213808**: Mouse cfRNA quantile-normalized counts
- **P08 Supp Data 1-6**: DRRs, CD36 genes, coregulated clusters, tissue specificity

## Key Features
- 466 differentially represented cfRNAs (DRRs) across spaceflight
- 406 CD36-enriched genes (tissue-specific markers)
- Cross-species comparison (human astronauts vs. mouse MHU-1)
- 11 timepoints spanning pre-flight, in-flight, and post-flight

## Processing
CLC Genomics Workbench v10.1.1, hg19 (human) / mm10 (mouse)
Mean scaling normalization with median reference and 5% trimming
