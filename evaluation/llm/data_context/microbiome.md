# Microbiome Data

## Dataset
- **Source**: Inspiration4 mission
- **Human samples**: 275 from 10 body sites across 4 crew, 7 timepoints
- **Environmental samples**: 39 from ISS surfaces
- **Human gut samples**: 8
- **Feature types**: Taxonomy-level CPM and Pathway-level CPM (MetaPhlAn/HUMAnN)

## Body Sites (10 sites, ~27-28 samples each)
EAR (ear canal), NAC (nasal cavity), ORC (oral cavity), PIT (axilla), TZO (toe zone), WEB (toe web), ARM (forearm), GLU (gluteal), NAP (nape), UMB (umbilicus)

## Benchmark Tasks
- **F1**: Body site classification from taxonomy (10-class) — Standard
  - N=275, LOCO, macro_f1, Best: RF=0.199
- **F2**: Flight phase detection from taxonomy (4-class) — Frontier
  - N=275, LOCO, macro_f1, Best: RF=0.238
- **F3**: Human vs environmental classification (binary) — Calibration
  - N=314, LOTO (7-fold), AUROC, Best: RF=0.841
- **F4**: Body site classification from pathways (10-class) — Standard
  - N=275, LOCO, macro_f1, Best: LogReg=0.163
- **F5**: Flight phase detection from pathways (4-class) — Frontier
  - N=275, LOCO, macro_f1, Best: RF=0.254

## Key Observations
- F3 is the only calibration-tier task (RF=0.841) — human vs ISS environment microbiomes are highly distinguishable
- Flight phase detection (F2, F5) is frontier difficulty — microbiome changes during 3-day spaceflight are subtle
- Body site classification achieves modest performance — 10-class problem with limited taxonomic resolution
- Taxonomy and pathway features give similar results for body site but slightly different for phase detection
