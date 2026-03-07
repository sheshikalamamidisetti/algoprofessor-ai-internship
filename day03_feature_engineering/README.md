# Day 03 — Ensemble Models: XGBoost & LightGBM (Wine Quality Dataset)

**Intern:** Sheshikala Mamidisetti
**Internship:** AlgoProfessor AI R&D Internship — Batch 2026
**Phase:** Phase 1 | Week 2 | Days 6–10
**Date:** March 6, 2026
**Milestone:** M1 — Web Intelligence Synthesiser

## Objective
For Day 03 I implemented two powerful ensemble models — XGBoost
and LightGBM on the Wine Quality dataset. I wanted to compare
both models and find which one predicts wine quality better.
I converted the quality score into binary classification —
good wine (quality >= 6) and bad wine (quality < 6).

## Models Compared

| Model | Accuracy | ROC-AUC | Status |
|-------|----------|---------|--------|
| XGBoost | 0.8250 | 0.8812 | ✅ Winner |
| LightGBM | 0.7906 | 0.8695 | ✅ Done |

## What I Found
- XGBoost won on both Accuracy (0.825) and ROC-AUC (0.881)
- LightGBM was close but slightly lower on both metrics
- Most important feature for wine quality is alcohol content
- Sulphates and volatile acidity also strongly affect quality
- Both models performed well above 0.78 accuracy

## My Recommendation
XGBoost is the better model for this dataset. It gives
higher accuracy and better ROC-AUC with clean numeric features
like wine chemical properties.

## Dataset
Wine Quality Dataset — winequality-red.csv
Source: UCI Machine Learning Repository
Rows: 1599 | Features: 11 | Target: quality (binary)

## Tools Used
Python | Pandas | NumPy | Scikit-learn | XGBoost | LightGBM | Matplotlib | Seaborn

## Output
ensemble_models_results.png — saved in outputs/
- XGBoost Confusion Matrix
- LightGBM Confusion Matrix
- ROC Curves — Both Models Together
- Feature Importance Comparison (Normalized)
