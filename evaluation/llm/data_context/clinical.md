# Clinical Biomarker Data

## Dataset
- **Source**: Inspiration4 mission, 4 civilian crew (C001-C004)
- **Samples**: 28 total (4 crew × 7 timepoints)
- **Phase distribution**: pre_flight=12, post_flight=4, recovery=12

## CBC (Complete Blood Count) — 20 features
Absolute counts: basophils, eosinophils, lymphocytes, monocytes, neutrophils
Percentages: basophils, eosinophils, lymphocytes, monocytes, neutrophils
Red cell indices: hematocrit, hemoglobin, MCH, MCHC, MCV, RBC count, RDW
Platelets: platelet count, MPV
White cells: WBC count

## CMP (Comprehensive Metabolic Panel)
Standard metabolic markers measured at same timepoints as CBC.

## Cytokines/Immune Markers — 71 features
EVE (extravehicular exposure) cytokine panel including inflammatory markers, growth factors, and chemokines. Measured in serum at all 7 timepoints.

## Benchmark Tasks
- **A1**: Flight phase classification from CBC + CMP features (3-class: pre/post/recovery)
  - Metric: macro_f1, LOCO evaluation (4-fold)
  - Best baseline: LogReg = 0.546
- **A2**: Flight phase classification from immune/cytokine markers (3-class)
  - Metric: macro_f1, LOCO evaluation (4-fold)
  - Best baseline: LogReg = 0.493

## Key Observations
- Post-flight class has only 4 samples (one per crew at R+1), creating severe imbalance
- CBC features show acute changes at R+1 including shifts in neutrophil/lymphocyte ratios
- LogReg outperforms RF and MLP, likely due to small N favoring simpler models
- LOCO evaluation is stringent: each fold tests on an entirely unseen crew member
