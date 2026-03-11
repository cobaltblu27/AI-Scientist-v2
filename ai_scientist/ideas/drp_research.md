# Title
SOTA Drug-Blind IC50 Regression on GDSC with Substructure-Aware Modeling

# Keywords
drug response prediction, cancer cell lines, IC50, GDSC, drug-blind split, SMILES, molecular substructure, scaffolds, fragments, motifs, subgraph fingerprints, graph neural networks, transcriptomics, scFoundation, MOSA, multimodal learning, transfer learning, relation modeling, interpretability

# TL;DR
Primary objective: achieve state-of-the-art performance on the GDSC drug-blind split for log(IC50) regression, where test drugs are unseen during training and drug representation is derived from SMILES. Use the CSG2A paper's drug-blind setting and metrics as the benchmark target.

## Task Specification (Hard Constraints)
1. Primary benchmark: GDSC drug-blind split (same evaluation protocol as CSG2A Table 1).
2. Prediction target: log(IC50) regression.
3. Input requirement: include drug SMILES-based modeling as the core drug modality.
4. Core metrics:
   - RMSE (lower is better)
   - PCC (higher is better)
5. Reporting requirement: report mean +- std over 10-fold CV under the drug-blind split.
6. All model comparisons must be made under the same split/protocol.

## Baseline Targets (CSG2A paper, drug-blind Table 1)
Format: RMSE +- std / PCC +- std

| Method | Drug-blind score |
|---|---|
| CSG2A | 2.119 +- 0.397 / 0.611 +- 0.140 |
| SVM | 2.268 +- 0.437 / 0.520 +- 0.177 |
| DeepTTA | 2.322 +- 0.496 / 0.502 +- 0.198 |
| GraphDRP | 2.354 +- 0.394 / 0.466 +- 0.163 |
| DRPreter | 2.473 +- 0.360 / 0.443 +- 0.175 |
| RF | 2.671 +- 0.579 / 0.406 +- 0.256 |
| DeepCoVDR | 2.754 +- 0.245 / 0.387 +- 0.200 |
| Precily | 2.825 +- 0.400 / 0.362 +- 0.109 |
| PathDNN | 3.257 +- 0.666 / 0.336 +- 0.271 |

## Success Criteria
1. Minimum target: beat strong non-CSG2A baselines (e.g., GraphDRP and DeepTTA) under the same protocol.
2. SOTA target: meet or exceed CSG2A (RMSE <= 2.119 and PCC >= 0.611), with fold-wise robustness.
3. Include at least one statistically grounded comparison against CSG2A-level performance (e.g., fold-wise paired test or confidence interval overlap discussion).

## Abstract
We focus on a constrained, benchmark-driven research task: improving drug response prediction on the GDSC drug-blind split for log(IC50) regression, with explicit emphasis on drug SMILES-based representation learning and substructure-aware modeling. The work should prioritize methods that improve generalization to unseen drugs while preserving reproducible protocol alignment with CSG2A-style evaluation. Candidate directions include substructure-conditioned molecular encoders, relation-aware multimodal fusion with cell-state context (e.g., scFoundation-like embeddings), and robust optimization strategies that improve both RMSE and PCC. All proposed experiments should be designed around direct comparability with published drug-blind baselines and should clearly demonstrate whether improvements are genuine under identical split and metric definitions.
