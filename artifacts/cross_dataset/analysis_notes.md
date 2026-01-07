Cross-dataset generalization
articles_to_liar: train=fake_true test=liar_test
- logreg: accuracy=0.4353 precision=0.6087 recall=0.0194 f1=0.0376 roc_auc=0.5696
liar_to_articles: train=liar_train+valid test=fake_true
- logreg: accuracy=0.5706 precision=0.5294 recall=0.9014 f1=0.6670 roc_auc=0.6733

Ethical limitations
- Lexical cues can correlate with style or topic rather than factuality.
- Dataset shift can drive errors when training on articles and testing on claims.
- Models ignore evidence and context, so they do not verify claims.
- Prediction errors can disproportionately affect speakers or groups.