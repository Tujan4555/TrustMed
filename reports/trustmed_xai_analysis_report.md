# TrustMed: Explainability and Sensitivity Analysis

**Report generated:** 2026-10-24

## 1. Purpose and scope

This report consolidates post-hoc explanation and sensitivity analyses for the frozen TrustMed E5 1D ResNet + BiLSTM model. It documents Integrated Gradients (IG), temporal attribution summaries, mean-replacement perturbation analysis, and case-level bootstrap comparisons.

These analyses describe model behavior on selected test windows. They do not establish causal physiological mechanisms, clinical utility, or the validity of the underlying dataset labels.

## 2. Model and analysis safeguards

- The E5 model was kept frozen and in evaluation mode.
- No retraining or checkpoint modification was performed.
- The final test decision threshold was not tuned during XAI analysis.
- The 60-window diagnostic sample contains 15 windows per prediction/label group.
- The 60-window sample is balanced for diagnostic comparison and is not representative of the natural test-set prevalence.
- Multiple windows may come from the same case; case-level aggregation and resampling were used for the reported bootstrap comparisons.

## 3. Representative-window Integrated Gradients

| xai_category           |   caseid |   label |   e5_risk_score | normalized_shard            |   normalized_row_index |
|:-----------------------|---------:|--------:|----------------:|:----------------------------|-----------------------:|
| positive_label_alerted |     4648 |       1 |          0.896  | normalized_shard_005598.npz |                     35 |
| positive_label_missed  |     5464 |       1 |          0.5999 | normalized_shard_006613.npz |                     97 |
| negative_label_alerted |     5971 |       0 |          0.9152 | normalized_shard_007264.npz |                      0 |

IG was computed with a zero baseline and the model output logit as the target. These choices define the attribution reference and target; they do not make the baseline physiologically neutral.

### IG attribution summary

| category               |   caseid |   label |   risk_score | channel   |   absolute_attribution_sum |   fraction_of_total_absolute_attribution |   mean_absolute_attribution |   ig_steps | baseline   |   completeness_delta_logit |
|:-----------------------|---------:|--------:|-------------:|:----------|---------------------------:|-----------------------------------------:|----------------------------:|-----------:|:-----------|---------------------------:|
| positive_label_alerted |     4648 |       1 |       0.896  | ART       |                     6.4852 |                                   0.7001 |                      0.0022 |        128 | zero       |                    -0.2372 |
| positive_label_alerted |     4648 |       1 |       0.896  | ECG_II    |                     1.9247 |                                   0.2078 |                      0.0006 |        128 | zero       |                    -0.2372 |
| positive_label_alerted |     4648 |       1 |       0.896  | PLETH     |                     0.8537 |                                   0.0922 |                      0.0003 |        128 | zero       |                    -0.2372 |
| positive_label_missed  |     5464 |       1 |       0.5999 | ART       |                     4.5846 |                                   0.4364 |                      0.0015 |        128 | zero       |                     0.1014 |
| positive_label_missed  |     5464 |       1 |       0.5999 | ECG_II    |                     5.596  |                                   0.5327 |                      0.0019 |        128 | zero       |                     0.1014 |
| positive_label_missed  |     5464 |       1 |       0.5999 | PLETH     |                     0.3245 |                                   0.0309 |                      0.0001 |        128 | zero       |                     0.1014 |
| negative_label_alerted |     5971 |       0 |       0.9152 | ART       |                     6.677  |                                   0.7938 |                      0.0022 |        128 | zero       |                    -0.2615 |
| negative_label_alerted |     5971 |       0 |       0.9152 | ECG_II    |                     0.9883 |                                   0.1175 |                      0.0003 |        128 | zero       |                    -0.2615 |
| negative_label_alerted |     5971 |       0 |       0.9152 | PLETH     |                     0.7465 |                                   0.0887 |                      0.0002 |        128 | zero       |                    -0.2615 |

### Temporal concentration

The following table reports the saved temporal-attribution summary. A high attribution share in a time region indicates concentration of attribution under the chosen method, not a validated clinical event boundary.

| category               |   total_absolute_attribution |   first_90_percent_share |   final_10_percent_share |   final_10_percent_to_first_90_percent_ratio |
|:-----------------------|-----------------------------:|-------------------------:|-------------------------:|---------------------------------------------:|
| positive_label_alerted |                       9.2635 |                   0.8155 |                   0.1845 |                                       0.2263 |
| positive_label_missed  |                      10.505  |                   0.8701 |                   0.1299 |                                       0.1492 |
| negative_label_alerted |                       8.4117 |                   0.7456 |                   0.2544 |                                       0.3412 |

### IG group comparison

| xai_group            |   windows |   unique_cases |   mean_final_10_share |   median_final_10_share |   std_final_10_share |   mean_abs_completeness_delta |   median_abs_completeness_delta |
|:---------------------|----------:|---------------:|----------------------:|------------------------:|---------------------:|------------------------------:|--------------------------------:|
| negative_alerted     |        15 |             13 |                0.2001 |                  0.1993 |               0.0339 |                        0.3549 |                          0.3681 |
| negative_not_alerted |        15 |             15 |                0.1242 |                  0.1213 |               0.0199 |                        0.048  |                          0.0034 |
| positive_alerted     |        15 |             12 |                0.2188 |                  0.2111 |               0.0475 |                        0.4421 |                          0.3395 |
| positive_missed      |        15 |             14 |                0.1546 |                  0.1347 |               0.0533 |                        0.1395 |                          0.0444 |

### IG numerical limitations

The earlier convergence check showed inconsistent completeness deltas across examples and integration-step counts. Increasing the number of steps did not reliably resolve the discrepancy. Consequently, IG maps should be treated as exploratory and should not be presented as quantitatively validated explanations.

| xai_group            |   caseid |   risk_score |   steps |   input_minus_baseline_logit |   attribution_sum |   completeness_delta |   absolute_completeness_delta |
|:---------------------|---------:|-------------:|--------:|-----------------------------:|------------------:|---------------------:|------------------------------:|
| positive_alerted     |     1595 |       0.6936 |      32 |                       4.4905 |            4.0973 |              -0.3933 |                        0.3933 |
| positive_alerted     |     1595 |       0.6936 |      64 |                       4.4905 |            3.9741 |              -0.5164 |                        0.5164 |
| positive_alerted     |     1595 |       0.6936 |     128 |                       4.4905 |            4.1511 |              -0.3395 |                        0.3395 |
| positive_alerted     |     1595 |       0.6936 |     256 |                       4.4905 |            4.3164 |              -0.1742 |                        0.1742 |
| positive_missed      |     3295 |       0.52   |      32 |                       3.7533 |            3.4926 |              -0.2606 |                        0.2606 |
| positive_missed      |     3295 |       0.52   |      64 |                       3.7533 |            3.4921 |              -0.2612 |                        0.2612 |
| positive_missed      |     3295 |       0.52   |     128 |                       3.7533 |            3.1728 |              -0.5805 |                        0.5805 |
| positive_missed      |     3295 |       0.52   |     256 |                       3.7533 |            3.3978 |              -0.3555 |                        0.3555 |
| negative_alerted     |     5950 |       0.7885 |      32 |                       4.9895 |            4.4061 |              -0.5833 |                        0.5833 |
| negative_alerted     |     5950 |       0.7885 |      64 |                       4.9895 |            4.5891 |              -0.4004 |                        0.4004 |
| negative_alerted     |     5950 |       0.7885 |     128 |                       4.9895 |            4.6214 |              -0.3681 |                        0.3681 |
| negative_alerted     |     5950 |       0.7885 |     256 |                       4.9895 |            4.5835 |              -0.4059 |                        0.4059 |
| negative_not_alerted |     5029 |       0.0287 |      32 |                       0.1503 |            0.1499 |              -0.0003 |                        0.0003 |
| negative_not_alerted |     5029 |       0.0287 |      64 |                       0.1503 |            0.1501 |              -0.0002 |                        0.0002 |
| negative_not_alerted |     5029 |       0.0287 |     128 |                       0.1503 |            0.1496 |              -0.0007 |                        0.0007 |
| negative_not_alerted |     5029 |       0.0287 |     256 |                       0.1503 |            0.1498 |              -0.0005 |                        0.0005 |
| positive_alerted     |     1416 |       0.8333 |      32 |                       5.2823 |            5.473  |               0.1907 |                        0.1907 |
| positive_alerted     |     1416 |       0.8333 |      64 |                       5.2823 |            5.9916 |               0.7093 |                        0.7093 |
| positive_alerted     |     1416 |       0.8333 |     128 |                       5.2823 |            5.5142 |               0.2319 |                        0.2319 |
| positive_alerted     |     1416 |       0.8333 |     256 |                       5.2823 |            5.782  |               0.4996 |                        0.4996 |

## 4. Mean-replacement perturbation analysis

For channel perturbation, one channel at a time was replaced with that channel's mean across the full 30-second window. For temporal perturbation, 100-sample segments were replaced across all channels with their respective window means. The signed logit change is defined as original logit minus perturbed logit; absolute logit change measures response magnitude.

Mean replacement may create waveforms unlike real physiological signals. These results measure sensitivity to the specified artificial perturbation, not intrinsic channel importance or clinical causality.

### Group-level channel sensitivity

| analysis_group       | channel   |   windows |   distinct_cases |   median_abs_logit_change |   mean_abs_logit_change |   std_abs_logit_change |   median_signed_logit_change |   mean_signed_logit_change |
|:---------------------|:----------|----------:|-----------------:|--------------------------:|------------------------:|-----------------------:|-----------------------------:|---------------------------:|
| negative_alerted     | ART       |        15 |               13 |                    0.8619 |                  0.8047 |                 0.3515 |                       0.8619 |                     0.8047 |
| negative_alerted     | ECG_II    |        15 |               13 |                    0.2365 |                  0.291  |                 0.249  |                       0.2365 |                     0.291  |
| negative_alerted     | PLETH     |        15 |               13 |                    0.0652 |                  0.1574 |                 0.2243 |                       0.0107 |                     0.1158 |
| negative_not_alerted | ART       |        15 |               15 |                    0.7173 |                  0.869  |                 0.7577 |                       0.7173 |                     0.8299 |
| negative_not_alerted | ECG_II    |        15 |               15 |                    0.622  |                  0.8472 |                 0.6624 |                       0.622  |                     0.8408 |
| negative_not_alerted | PLETH     |        15 |               15 |                    0.0857 |                  0.3691 |                 0.4718 |                       0.023  |                     0.3241 |
| positive_alerted     | ART       |        15 |               12 |                    0.9758 |                  0.8402 |                 0.3706 |                       0.9758 |                     0.806  |
| positive_alerted     | ECG_II    |        15 |               12 |                    0.2159 |                  0.2817 |                 0.3221 |                       0.2159 |                     0.2617 |
| positive_alerted     | PLETH     |        15 |               12 |                    0.0887 |                  0.1511 |                 0.1794 |                       0.0529 |                     0.1183 |
| positive_missed      | ART       |        15 |               14 |                    0.4656 |                  0.5006 |                 0.3168 |                       0.1971 |                     0.1573 |
| positive_missed      | ECG_II    |        15 |               14 |                    0.3601 |                  0.4587 |                 0.2959 |                       0.3601 |                     0.3885 |
| positive_missed      | PLETH     |        15 |               14 |                    0.0921 |                  0.1124 |                 0.1141 |                       0.0272 |                     0.0462 |

### Case-level bootstrap confidence intervals

Confidence intervals were generated by resampling cases, after aggregating multiple windows within each case. The number of cases is small, and the balanced diagnostic sample does not support prevalence-level conclusions.

| analysis_group       | channel   |   distinct_cases |   case_level_median_abs_logit_change |   ci_2.5_percent |   ci_97.5_percent |
|:---------------------|:----------|-----------------:|-------------------------------------:|-----------------:|------------------:|
| negative_alerted     | ART       |               13 |                               0.8619 |           0.7273 |            0.9389 |
| negative_alerted     | ECG_II    |               13 |                               0.2365 |           0.1212 |            0.3494 |
| negative_alerted     | PLETH     |               13 |                               0.0652 |           0.0131 |            0.1488 |
| negative_not_alerted | ART       |               15 |                               0.7173 |           0.5004 |            1.0953 |
| negative_not_alerted | ECG_II    |               15 |                               0.622  |           0.5554 |            1.1479 |
| negative_not_alerted | PLETH     |               15 |                               0.0857 |           0.0509 |            0.5771 |
| positive_alerted     | ART       |               12 |                               0.9647 |           0.6226 |            1.0757 |
| positive_alerted     | ECG_II    |               12 |                               0.1934 |           0.0476 |            0.3658 |
| positive_alerted     | PLETH     |               12 |                               0.0944 |           0.045  |            0.1876 |
| positive_missed      | ART       |               14 |                               0.4834 |           0.2639 |            0.6782 |
| positive_missed      | ECG_II    |               14 |                               0.4379 |           0.2574 |            0.6497 |
| positive_missed      | PLETH     |               14 |                               0.0814 |           0.0252 |            0.155  |

### Paired channel comparisons

Paired differences are defined as the first channel's absolute logit change minus the second channel's absolute logit change. An interval entirely above zero supports greater median sensitivity for the first channel within this sample; an interval entirely below zero supports the second.

| analysis_group       | comparison     |   distinct_cases |   median_paired_difference |   mean_paired_difference |   ci_2.5_percent |   ci_97.5_percent | direction     |
|:---------------------|:---------------|-----------------:|---------------------------:|-------------------------:|-----------------:|------------------:|:--------------|
| negative_alerted     | ART - ECG_II   |               13 |                     0.5894 |                   0.5567 |           0.4164 |            0.743  | ART higher    |
| negative_alerted     | ART - PLETH    |               13 |                     0.7042 |                   0.6898 |           0.5641 |            0.916  | ART higher    |
| negative_alerted     | ECG_II - PLETH |               13 |                     0.1111 |                   0.133  |           0.0508 |            0.1478 | ECG_II higher |
| negative_not_alerted | ART - ECG_II   |               15 |                    -0.2486 |                   0.0218 |          -0.448  |            0.4733 | ECG_II higher |
| negative_not_alerted | ART - PLETH    |               15 |                     0.4147 |                   0.4999 |           0.1761 |            0.707  | ART higher    |
| negative_not_alerted | ECG_II - PLETH |               15 |                     0.4932 |                   0.4781 |           0.0952 |            0.9293 | ECG_II higher |
| positive_alerted     | ART - ECG_II   |               12 |                     0.5962 |                   0.5454 |           0.2912 |            0.7348 | ART higher    |
| positive_alerted     | ART - PLETH    |               12 |                     0.7881 |                   0.687  |           0.3392 |            0.959  | ART higher    |
| positive_alerted     | ECG_II - PLETH |               12 |                     0.0796 |                   0.1417 |          -0.0281 |            0.2351 | ECG_II higher |
| positive_missed      | ART - ECG_II   |               14 |                     0.0917 |                   0.041  |          -0.2653 |            0.3296 | ART higher    |
| positive_missed      | ART - PLETH    |               14 |                     0.3646 |                   0.4045 |           0.1852 |            0.653  | ART higher    |
| positive_missed      | ECG_II - PLETH |               14 |                     0.2595 |                   0.3636 |           0.1267 |            0.5797 | ECG_II higher |

## 5. Integrated interpretation

The observed pattern in this diagnostic sample is that ART generally produced larger logit changes than ECG-II and PLETH when replaced with its window mean, particularly for positive-alerted and negative-alerted groups. For positive-missed windows, ART and ECG-II sensitivities were not clearly distinguishable in the paired case-level bootstrap. In negative-not-alerted windows, the ART-versus-ECG-II comparison was also inconclusive.

These are exploratory findings. They do not establish that ART is clinically superior, explain the cause of missed predictions, or show that the model learned clinically valid hypotension physiology. The results may depend on normalization, signal quality, the artificial replacement strategy, and the selected sample.

## 6. Limitations

1. **Small diagnostic sample:** 60 windows and 53 distinct cases; only 12–15 cases contribute to each group-level comparison.
2. **Balanced sampling:** Group-level summaries cannot estimate the frequency of these patterns in the full test population.
3. **Artificial perturbations:** Mean replacement can introduce atypical waveforms and model responses may be sensitive to this choice.
4. **IG completeness:** Convergence was inconsistent, limiting confidence in quantitative attribution magnitudes.
5. **Label uncertainty:** The original dataset label-generation rule, including its clinical threshold, duration, and temporal boundary handling, has not been independently verified.
6. **No causal interpretation:** Neither IG nor perturbation sensitivity establishes causal physiological importance.
7. **Multiple comparisons:** The paired analysis examined several channel comparisons without multiplicity adjustment; results are exploratory.

## 7. Recommended reporting language

> In an exploratory, balanced diagnostic sample, mean-replacement perturbation analysis showed greater model-output sensitivity to ART than to ECG-II and PLETH in the positive-alerted and negative-alerted groups. Channel sensitivity was less clearly separated in positive-missed and negative-not-alerted cases. Integrated Gradients exhibited inconsistent completeness convergence. These post-hoc findings characterize model sensitivity and should not be interpreted as causal or clinically validated explanations.

## 8. Reproducibility artifacts

- `xai_integrated_gradients_summary.csv` — Available
- `xai_temporal_concentration.csv` — Available
- `xai_group_comparison.csv` — Available
- `xai_ig_convergence.csv` — Available
- `xai_60_channel_sensitivity_summary.csv` — Available
- `xai_60_channel_sensitivity_bootstrap_ci.csv` — Available
- `xai_paired_channel_bootstrap.csv` — Available
- `xai_representative_test_windows.csv` — Available

- `xai_channel_occlusion.csv` — three representative windows, full-channel perturbations.
- `xai_temporal_occlusion.csv` — three representative windows, temporal perturbations.
- `xai_60_channel_sensitivity.csv` — 60-window channel-level perturbation results.
- `xai_60_channel_sensitivity_bootstrap_ci.csv` — case-level bootstrap intervals.
- `xai_paired_channel_bootstrap.csv` — paired channel comparisons.

## 9. Conclusion

The current XAI results provide preliminary evidence about which input channels influence E5 outputs under the tested perturbations. They are not sufficient to claim clinically validated explanations. The most defensible use is as a transparent exploratory analysis accompanying the model's predictive evaluation and its documented limitations.
