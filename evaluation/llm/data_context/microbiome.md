# Microbiome Data

## Dataset
- **Source**: Inspiration4 mission
- **Human samples**: 275 from 10 body sites across 4 crew, 7 timepoints (total human + environmental = 314)
- **Environmental samples**: 39 from ISS surfaces
- **Human gut samples**: 8 (separate from the 275 body-site samples; not used in F1-F5 tasks)
- **Feature types**: Taxonomy-level CPM and Pathway-level CPM (MetaPhlAn/HUMAnN)

## Body Sites (10 sites, ~27-28 samples each)
EAR (ear canal), NAC (nasal cavity), ORC (oral cavity), PIT (axilla), TZO (toe zone), WEB (toe web), ARM (forearm), GLU (gluteal), NAP (nape), UMB (umbilicus)

## Benchmark Tasks
- **F1**: Body site classification from taxonomy (10-class) — Standard
  - N=275, LOCO, macro_f1, Random: 0.112
  - Best: LightGBM=0.200 | RF=0.199 | XGBoost=0.193 | LogReg=0.147 | MLP=0.108
- **F2**: Flight phase detection from taxonomy (4-class: pre/in/post/recovery) — Frontier
  - N=275, LOCO, macro_f1, Random: 0.205
  - Best: LightGBM=0.280 | XGBoost=0.263 | RF=0.238 | LogReg=0.236 | MLP=0.204
- **F3**: Human vs environmental classification (binary) — Calibration
  - N=314, LOTO (7-fold), AUROC, Random: 0.402
  - Best: RF=0.841 | XGBoost=0.838 | LightGBM=0.838 | LogReg=0.574 | MLP=0.320
- **F4**: Body site classification from pathways (10-class) — Standard
  - N=275, LOCO, macro_f1, Random: 0.112
  - Best: LogReg=0.163 | LightGBM=0.160 | RF=0.151 | XGBoost=0.134 | MLP=0.096
- **F5**: Flight phase detection from pathways (4-class: pre/in/post/recovery) — Frontier
  - N=275, LOCO, macro_f1, Random: 0.205
  - Best: LightGBM=0.304 | XGBoost=0.300 | RF=0.254 | LogReg=0.240 | MLP=0.229

## Key Observations
- F3 is the only calibration-tier task (RF=0.841) — human vs ISS environment microbiomes are highly distinguishable
- Flight phase detection (F2, F5) is frontier difficulty — microbiome changes during 3-day spaceflight are subtle
- Body site classification achieves modest performance — 10-class problem with limited taxonomic resolution
- Taxonomy and pathway features give similar results for body site but slightly different for phase detection
