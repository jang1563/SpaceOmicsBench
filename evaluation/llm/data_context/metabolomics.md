# Metabolomics Data

## Dataset
- **Source**: Inspiration4 mission
- **Metabolites**: 433 measured across 4 crew, multiple timepoints
- **Analytical modes**: ANP-POS and RP-POS mass spectrometry
- **Spaceflight-responsive**: 91/433 metabolites (21%) show significant DE

## Features per Metabolite
- **Mass**: molecular mass from MS
- **RT**: retention time
- **annotation_confidence**: quality of metabolite identification
- **SuperPathway**: broad metabolic category
- **SubPathway**: specific metabolic pathway
- **Formula**: chemical formula (decomposed into C, H, N, O, S, P atom counts)

## Benchmark Task
- **D1**: Metabolite spaceflight response prediction (binary)
  - N=433, positives=91 (21%), metric=AUROC
  - Feature 80/20 split (5 reps)
  - Best: RF=0.676 | LightGBM=0.638 | XGBoost=0.617 | LogReg=0.561 | MLP=0.557, Random: 0.481

## Key Observations
- Moderate positive rate (21%) makes this amenable to AUROC evaluation
- Chemical structure features (atom counts) provide physicochemical context
- Pathway annotations enable biological interpretation of predictions
- Short-duration (3-day) metabolomic changes likely reflect acute stress response rather than chronic adaptation
