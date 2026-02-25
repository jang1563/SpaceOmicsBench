# P10: NASA Twins Study

- **Paper**: "The NASA Twins Study: A multidimensional analysis of a year-long human spaceflight"
- **Journal**: Science, April 2019
- **DOI**: 10.1126/science.aau8650
- **Mission**: NASA One-Year Mission (340 days ISS, 2015-2016)

## Data Types Covered

- Telomere length dynamics
- Epigenomics (DNA methylation, genome-wide)
- Transcriptomics (bulk RNA-seq)
- Proteomics
- Metabolomics
- Immune profiling (cytokines, immune cell populations)
- Microbiome (16S, metagenomics)
- Cognitive performance testing

## Access Restrictions -- IMPORTANT

The NASA Twins Study involves only 2 subjects (identical twins, one astronaut),
making individual-level data inherently identifiable. Most data is restricted.

### What we CAN use (public, published):
- **Supplementary tables** from the Science paper (aggregate statistics, fold-changes)
- **Cross-referenced data** from P07 (hemoglobin) and P09 (TERRA) papers
- Summary-level pathway enrichment and gene set results

### What we CANNOT use (restricted):
- Individual-level omics data (requires dbGaP access)
- Raw sequencing data (requires LSDA formal request at https://nlsp.nasa.gov/explore/lsdahome/datarequest)
- Detailed clinical records (retained by NASA)

## Strategy for v2 Benchmark

Use only published aggregate/summary data from supplementary tables.
This provides sufficient cross-mission comparison context without privacy concerns.
The Twins data primarily serves as a reference point for long-duration effects,
contrasted with the short-duration I4 and medium-duration JAXA CFE data.

## Download

Supplementary materials from:
```
https://www.science.org/doi/10.1126/science.aau8650
```
