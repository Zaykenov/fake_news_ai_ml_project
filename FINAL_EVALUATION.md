# Final Evaluation

This evaluation summarizes results from:
- `artifacts/baseline/baseline_results.json`
- `artifacts/calibration/calibration_results.json`
- `artifacts/robustness/robustness_results.csv`
- `artifacts/tfidf_ablation/tfidf_ablation_results.csv`
- `artifacts/cross_dataset/cross_dataset_results.json`
- `artifacts/cross_dataset/analysis_notes.md`

## Overall performance (article-level Fake/True)

- Baseline (multi-view TF-IDF + tuned linear models) shows strong performance:
  - Logistic Regression: accuracy 0.907, F1 0.908, ROC-AUC 0.980
  - Linear SVM: accuracy 0.912, F1 0.912, ROC-AUC 0.981
- Calibration experiment reports near-saturated scores (SVM up to F1 0.9999, ROC-AUC ~1.0),
  which suggests extremely separable signals in this dataset and/or potential dataset artifacts.

## Robustness

- Perturbations (typos and truncation) barely affect performance.
- Best F1 remains ~0.9999 even with typos or truncation.
- This indicates the model is highly sensitive to strong lexical cues in the dataset, but
  it may also indicate dataset-specific artifacts rather than generalizable reasoning.

## Feature ablations

- Best Linear SVM results:
  - `body_char` and `all_views` tie at F1 ~0.9999
  - `body_word+char` is slightly lower but still ~0.9997
- Best Logistic Regression results:
  - `all_views` is strongest at F1 ~0.9984
- Character n-grams are extremely strong for this dataset.

## Cross-dataset generalization (articles ↔ LIAR claims)

- Training on Fake/True articles and testing on LIAR claims:
  - Accuracy 0.435, F1 0.038, ROC-AUC 0.570
  - Precision is high (0.609) but recall is near zero (0.019), indicating a severe mismatch.
- Training on LIAR claims and testing on Fake/True articles:
  - Accuracy 0.571, F1 0.667, ROC-AUC 0.673
  - Recall is high (0.901) but precision is modest (0.529), indicating over-prediction of REAL.

## Interpretation

- Article-level performance is strong, but cross-dataset results show poor transfer from
  articles to claims and only moderate transfer in the reverse direction.
- The extremely high scores in calibration/robustness/ablations likely reflect dataset artifacts
  or label-correlated phrasing rather than robust truthfulness detection.
- Cross-dataset degradation highlights that lexical models do not generalize across domains
  (long-form news vs. short claims) without adaptation.

## Ethical and methodological limitations

- Models rely on surface lexical patterns and do not verify claims against evidence.
- Domain shift leads to large performance drops, which can mislead users in real deployments.
- Lexical shortcuts can encode bias against speakers or topics, even when metadata is excluded.

## Bottom line

The pipeline is effective for Fake/True articles within the same dataset distribution, but
generalization to claim-level truthfulness (LIAR) is weak. This supports using LIAR strictly
as a benchmark and reporting cross-dataset gaps as a core limitation of lexical TF-IDF models.
