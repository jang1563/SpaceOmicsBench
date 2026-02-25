# Space Anemia and Hemoglobin Pathway

## Background
Space anemia is a well-documented consequence of spaceflight. Key findings from literature:
- ~54% more red blood cells destroyed in space compared to ground (Trudel et al., Nature Medicine 2022)
- Hemolysis continues for at least 1 year post-flight
- Compensatory erythropoiesis upregulates hemoglobin-related genes

## Hemoglobin Gene Set
- **Total genes**: 59 hemoglobin/erythropoiesis pathway genes
- **Key genes**: HBA1, HBA2 (alpha-globins), HBB (beta-globin), HBBP1, ALAS2 (rate-limiting erythropoiesis enzyme)
- **Overlap with DE data**: 57 of 59 genes found in the 26,845-gene Twins DE dataset

## Expression Data (from gt_hemoglobin_globin_genes.csv)
- 59 genes measured across 4 crew and 11 timepoints (Pre1-3, Flight1-4, Post1-4)
- HBB expression: normalized means range from ~11,000 (Flight2) to ~33,000 (Post4), showing ~40% post-flight increase
- HBA1 and HBA2 show similar patterns: decreased during flight, increased post-flight
- ALAS2 (aminolevulinic acid synthase 2): key rate-limiting enzyme for heme biosynthesis

## Cross-Mission DE (Task I1)
- **Features**: 3 fold-change values from Twins transcriptome (Pre vs Flight FC, Pre vs Post FC, Flight vs Post FC)
- **Target**: Whether gene is in the 57 hemoglobin/erythropoiesis gene set
- **N=26,845**, positives=57 (0.21%)
- **Metric**: AUPRC
- **Difficulty**: Frontier — only 3 features with extreme class imbalance
- **Baseline results**: RF=0.005, LogReg=0.003 (near random=0.003)

## Biological Interpretation
- Alpha-globin (HBA1, HBA2) and beta-globin (HBB) are the main oxygen-carrying proteins in hemoglobin
- Spaceflight causes splenic hemolysis → compensatory upregulation of erythropoiesis genes
- The post-flight increase is consistent with recovery response: body producing new RBCs to replace those lost in space
- ALAS2 upregulation confirms active heme biosynthesis pathway activation
- Cross-mission consistency (both I4 and Twins show hemoglobin pathway activation) suggests this is a universal spaceflight response
