# TrustMed

**Early prediction of intraoperative hypotension using physiological time-series data**

TrustMed explores a 1D ResNet + BiLSTM model for predicting dataset-provided hypotension labels from three physiological waveform channels: ART, ECG-II, and PLETH.

## Current project status

- E5 model architecture and trained checkpoint are included.
- Saved held-out test metrics are included.
- Exploratory RL and XAI findings are documented.
- This repository is a research prototype, not a clinical device.

## Model

- Architecture: `TrustMedE5ResNetBiLSTM`
- Input shape: `(batch, 3, 3000)`
- Parameters: 343,969
- Saved checkpoint epoch: 10
- Best validation AUPRC recorded during training: 0.248789

The saved checkpoint was strictly loaded into the archived architecture and passed a dummy-input inference smoke test. This verifies software compatibility, not clinical validity.

## Held-out test results

The saved test score file was audited against the final metrics, and all reported metrics were recomputed from the saved risk scores. The recomputed results matched the saved values to floating-point precision.

| Metric | Value |
|---|---:|
| Test windows | 150,760 |
| Test cases | 486 |
| Positive-label prevalence | 0.033145 |
| AUPRC | 0.256362 |
| AUROC | 0.880774 |
| Threshold used for reported test metrics | 0.60 |
| Precision | 0.255828 |
| Recall | 0.522714 |
| F1 | 0.343526 |
| Alert rate | 0.067724 |

These are measures of agreement with the labels available in the dataset. The original label-generation rules were not independently verified, so these results must not be described as validated clinical event-detection performance.

## Repository layout

- `src/`: model architecture and training-engine source.
- `checkpoints/`: saved E5 model checkpoint.
- `results/`: aggregate test metrics and selected aggregate XAI analyses.
- `reports/`: XAI analysis report and limitations.

Case-level test prediction tables, patient-linked sample records, raw physiological data, and training/recovery backups are intentionally excluded from this public release.

## Reproducibility status

Verified:
1. The final artifact archive passed ZIP integrity validation.
2. The checkpoint strictly loaded with no missing or unexpected state-dictionary keys.
3. The model accepted a dummy input of shape `(1, 3, 3000)` and returned a finite scalar logit.
4. The saved test-score CSV matched its working copy byte-for-byte.
5. Test metrics recomputed from the saved risk scores matched the saved metrics.

Not yet verified:
- Fresh inference reproducing the full held-out test scores from the original preprocessed test inputs.
- Complete preprocessing and label regeneration from raw data using a standalone pipeline.

The original preprocessed input shards and complete label-generation procedure are not included. Therefore, this release does not claim fully end-to-end reproduction from raw data.

## Exploratory RL and XAI

RL results are exploratory and did not establish a validated clinical policy. Integrated Gradients convergence was inconsistent, and channel-perturbation sensitivity is not evidence of causality or clinical importance. See `reports/trustmed_xai_analysis_report.md`.

## Data and clinical limitations

Use only data obtained and processed under the relevant dataset terms and institutional approvals. Do not commit patient-level data, identifiers, credentials, or access tokens.

This research prototype is not intended for diagnosis, treatment selection, intraoperative intervention, or autonomous clinical decision-making. Prospective and external validation, reliable label definitions, calibration, safety assessment, and clinical review would be required before clinical use.

## Environment

The code was developed in a Kaggle/PyTorch environment. See `requirements.txt` for a minimal dependency list. Exact environment and preprocessing reproducibility have not yet been established.
