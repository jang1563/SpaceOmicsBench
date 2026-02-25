# Clinical Data (CBC / CMP)

Complete blood count and comprehensive metabolic panel biomarkers.

## Sources
- **OSD-569**: I4 whole blood CBC (4 crew, 6+ timepoints)
- **P01 Supp Table 1**: Sample collection metadata
- **P10**: Twins Study aggregate clinical data (from published supplementary)

## Expected Files (after fetch)
- `OSD-569_normalized_counts.csv` -- GeneLab-processed CBC values
- `OSD-569_SampleTable.csv` -- Sample metadata with timepoints
- I4 CBC/CMP feature matrix (from paper supplementary)

## Biomarkers (34 standard analytes)
WBC, RBC, Hemoglobin, Hematocrit, MCV, MCH, MCHC, RDW, Platelet Count,
MPV, Absolute Neutrophils, Absolute Lymphocytes, Absolute Monocytes,
Absolute Eosinophils, Absolute Basophils, NLR, plus CMP analytes.
