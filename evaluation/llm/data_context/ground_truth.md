# Ground Truth Key Facts

## Mission Facts
- I4: 4 civilian crew, 3 days LEO (~585 km), September 2021, SpaceX Dragon
- NASA Twins: 340 days ISS, Scott Kelly vs Mark Kelly (twin control), 2015-2016
- JAXA CFE: 6 astronauts, >120 days ISS, cell-free RNA epigenome

## Benchmark Statistics
- 21 ML tasks: 19 main + 2 supplementary (E2, E3)
- 9 categories: Clinical, cfRNA, Proteomics, Metabolomics, Spatial, Microbiome, Multi-modal, Cross-tissue, Cross-mission
- Tiers: Calibration=1, Standard=5, Advanced=9, Frontier=6

## Baseline Results (Best per Task)
| Task | Tier | Best Model | Score |
|------|------|-----------|-------|
| A1 | Standard | LogReg | 0.546 |
| A2 | Standard | LogReg | 0.493 |
| B1 | Advanced | RF | 0.885 |
| B2 | Advanced | LogReg | 0.154 |
| C1 | Standard | MLP | 0.517 |
| C2 | Frontier | RF | 0.555 |
| D1 | Advanced | RF | 0.676 |
| E1 | Advanced | LogReg | 0.017 |
| E4 | Advanced | LogReg | 0.023 |
| F1 | Standard | RF | 0.199 |
| F2 | Frontier | RF | 0.238 |
| F3 | Calibration | RF | 0.841 |
| F4 | Standard | LogReg | 0.163 |
| F5 | Frontier | RF | 0.254 |
| G1 | Advanced | LogReg | 0.517 |
| H1 | Advanced | RF | 0.266 |
| I1 | Frontier | RF | 0.005 |
| I2 | Advanced | RF | 0.706 |
| I3 | Advanced | LogReg | 0.090 |

## Composite Scores (Normalized)
- RF: 0.258 (best overall)
- LogReg: 0.201
- MLP: 0.133

## Cross-Mission Facts
- 146/452 pathways conserved between I4 and Twins (32.3%)
- 814/15,540 genes show conserved DE (5.2%)
- 57 hemoglobin/erythropoiesis genes in the dataset
- HBB shows ~40% post-flight expression increase

## Key Limitations
- N=4 crew (I4), N=1 treatment (Twins) — extremely small sample sizes
- 3-day mission may not capture chronic adaptation effects
- LOCO evaluation stringent but low-N means high variance
- Cross-mission comparisons confounded by different platforms, crews, and durations
