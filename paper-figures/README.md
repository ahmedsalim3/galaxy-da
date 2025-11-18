# Results

Domain adaptation experiments for galaxy morphology classification (source: TNG50 simulation, target: SDSS).

## Training Information

| Method | Total Epochs | Best Epoch | Final Train Loss | Best Train Loss |
|---|---|---|---|---|
| Baseline | 54 | 44 | 0.0005 | 0.0003 |
| DANN | 83 | 13 | 0.6985 | 0.0003 |
| Euclidean (Fixed $\lambda$) | 136 | 14 | 0.0029 | 0.0003 |
| Euclidean ($\eta_1,\eta_2$) | 200 | 197 | **-2.5282** | **-2.5289** |
| Euclidean (scheduler $\sigma$ + $\eta_1 \eta_2$) | 200 | 197 | -2.5189 | -2.5217 |

## Source-domain performance

| Method | Source Acc | Source Macro F1 | Source ROC-AUC | Source AUPRC | Source ECE |
|---|---|---|---|---|---|
| Baseline | 85.9% | **0.680** | **0.895** | **0.727** | 0.0745 |
| DANN | 86.7% | 0.660 | 0.890 | 0.710 | 0.0725 |
| Euclidean (Fixed $\lambda$) | **87.1%** | 0.627 | 0.860 | 0.693 | **0.0679** |
| Euclidean ($\eta_1,\eta_2$) | 85.9% | 0.624 | 0.842 | 0.680 | 0.1145 |
| Euclidean (scheduler $\sigma$ + $\eta_1 \eta_2$) | 86.2% | 0.649 | 0.868 | 0.705 | 0.1059 |

## Target-domain performance

| Method | Target Acc | Target Macro F1 | Target ROC-AUC | Target AUPRC | Target ECE | Domain AUC (test) | A-distance (test) | Domain AUC (full) | A-distance (full) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 46.8% | 0.298 | 0.540 | 0.373 | **0.0465** | 1.0000 | **0.0000** | 1.0000 | **0.0023** |
| DANN | 86.5% | 0.606 | 0.852 | 0.637 | 0.1174 | 0.5827 | 0.8793 | 0.8955 | 0.8960 |
| Euclidean (Fixed $\lambda$) | 85.0% | 0.587 | 0.857 | 0.634 | 0.1352 | 0.5497 | 0.8793 | 0.7311 | 0.8960 |
| Euclidean ($\eta_1,\eta_2$) | **87.3%** | **0.626** | **0.858** | **0.660** | 0.1055 | 0.5269 | 0.8736 | 0.7102 | 0.8742 |
| Euclidean (scheduler $\sigma$ + $\eta_1 \eta_2$) | 85.4% | 0.601 | 0.851 | 0.649 | 0.1262 | **0.5140** | 0.8908 | **0.6399** | 0.8616 |

## Target per-class recall

| Method | Elliptical | Irregular | Spiral |
|---|---|---|---|
| Baseline | 4.3% | **55.8%** | 53.2% |
| DANN | 94.3% | 9.3% | **93.5%** |
| Euclidean (Fixed $\lambda$) | 95.7% | 4.7% | 91.8% |
| Euclidean ($\eta_1,\eta_2$) | **100.0%** | 9.3% | **93.5%** |
| Euclidean (scheduler $\sigma$ + $\eta_1 \eta_2$) | 97.1% | 7.0% | 91.8% |

## Target per-class precision

| Method | Elliptical | Irregular | Spiral |
|---|---|---|---|
| Baseline | 20.0% | 9.4% | 87.3% |
| DANN | 59.5% | **26.7%** | **96.6%** |
| Euclidean (Fixed $\lambda$) | 63.8% | 10.0% | 94.6% |
| Euclidean ($\eta_1,\eta_2$) | **66.0%** | 25.0% | 95.7% |
| Euclidean (scheduler $\sigma$ + $\eta_1 \eta_2$) | 64.8% | 13.0% | 95.3% |