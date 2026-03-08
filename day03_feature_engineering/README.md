# Day 03 — Feature Engineering & Advanced ML (Wine Quality Dataset)

**Intern:** Sheshikala Mamidisetti
**Internship:** AlgoProfessor AI R&D Internship — Batch 2026
**Phase:** Phase 1 | Week 2 | Days 6–10
**Dates:** March 6–7, 2026
**Milestone:** M1 — Web Intelligence Synthesiser

---

## Objective

For Day 03 I implemented advanced machine learning techniques
on the Wine Quality dataset. I covered ensemble models using
XGBoost and LightGBM, created new features through feature
engineering, reduced dimensions using PCA LDA and SVD, built
a production grade Scikit-learn pipeline, tuned hyperparameters
using GridSearchCV and RandomizedSearchCV, and finally built a
neural network completely from scratch using PyTorch.

---

## Dataset

| Property | Details |
|----------|---------|
| File | winequality-red.csv |
| Rows | 1599 |
| Features | 11 chemical properties |
| Target | quality — binary classification |
| Good Wine | quality >= 6 → label 1 |
| Bad Wine | quality < 6 → label 0 |
| Source | UCI Machine Learning Repository |

---

## 1. Ensemble Models (ensemble_models.py)

I implemented XGBoost and LightGBM on the Wine Quality dataset
and compared both models across accuracy and ROC-AUC.

| Model | Accuracy | ROC-AUC | Winner |
|-------|----------|---------|--------|
| XGBoost | 0.8250 | 0.8812 | Yes |
| LightGBM | 0.7906 | 0.8695 | No |

**Key Finding:** XGBoost won on both metrics. Alcohol content
is the most important feature for predicting wine quality.

---

## 2. Feature Engineering (feature_engineering.py)

I created 4 new features from existing ones and compared
model accuracy before and after feature engineering.

**New Features Created:**

| Feature | Formula |
|---------|---------|
| acidity_ratio | fixed_acidity / volatile_acidity |
| sulfur_ratio | free_sulfur / total_sulfur |
| alcohol_density | alcohol / density |
| total_acidity | fixed + volatile + citric acid |

**Accuracy Improvement:**

| | Accuracy |
|--|---------|
| Without new features | 0.8063 |
| With new features | 0.8219 |
| Improvement | +1.56% |

**Key Finding:** alcohol_density became the most important
feature overall — more than any original feature.

---

## 3. Dimensionality Reduction (dimensionality_reduction.py)

I applied PCA, LDA and SVD to reduce 11 features into
fewer components and compared their accuracy.

| Method | Components | Variance | Accuracy |
|--------|-----------|---------|----------|
| PCA | 2 | 45.68% | 0.7094 |
| LDA | 1 | 100% | 0.7125 |
| SVD | 2 | 45.68% | 0.7094 |

**Key Finding:** LDA gave best accuracy with just 1 component.
PCA needs 9 components to explain 95% of variance.
Accuracy drops when reducing to 2 components — expected
trade-off between compression and performance.

---

## 4. Production ML Pipeline (ml_pipeline.py)

I built 4 production-grade Scikit-learn pipelines combining
StandardScaler and model in one reusable step. Evaluated
using 5-fold cross validation to prevent data leakage.

| Pipeline | Accuracy | ROC-AUC | CV Mean |
|----------|----------|---------|---------|
| Logistic Regression | 0.740 | 0.824 | 0.732 |
| Random Forest | 0.803 | 0.902 | 0.733 |
| SVM | 0.763 | 0.837 | 0.737 |
| Gradient Boosting | 0.791 | 0.870 | 0.729 |

**Winner:** Random Forest — highest accuracy (0.803) and
best ROC-AUC (0.902).

**Key Finding:** Pipeline ensures no data leakage between
train and test sets — this is what makes it production grade.

---

## 5. Hyperparameter Tuning (hyperparameter_tuning.py)

I tuned Random Forest using GridSearchCV and Gradient Boosting
using RandomizedSearchCV and compared their best results.

| Model | Accuracy | ROC-AUC | CV Score | Method |
|-------|----------|---------|----------|--------|
| Random Forest | 0.8187 | 0.9065 | 0.8108 | GridSearchCV |
| Gradient Boosting | 0.8156 | 0.8895 | 0.7991 | RandomizedSearchCV |

**Best Params — Random Forest:**
n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1

**Best Params — Gradient Boosting:**
n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.7

**Winner:** Random Forest — higher accuracy and ROC-AUC after tuning.

**Key Finding:** Hyperparameter tuning improved Random Forest
accuracy from 0.803 to 0.8187 — a gain of +1.57%.

---

## 6. PyTorch Neural Network from Scratch (pytorch_nn_scratch.py)

I built a neural network completely from scratch using PyTorch
without using any high-level APIs — just layers, activations,
loss function and optimizer built manually.

| Property | Details |
|----------|---------|
| Architecture | 11 → 64 → 32 → 16 → 1 |
| Activation | ReLU (hidden) + Sigmoid (output) |
| Loss | Binary Cross Entropy |
| Optimizer | Adam (lr=0.001) |
| Dropout | 0.3 (regularization) |
| Epochs | 50 |
| Test Accuracy | 0.7594 |

**Key Finding:** Neural network learns through backpropagation
— adjusting weights after each epoch to reduce loss.
Loss decreased from 0.50 to 0.45 over 50 epochs showing
the network was learning correctly.

---

## 7. Auto Report (generate_report.py)

Auto generated complete summary of all Day 03 results
saved as outputs/day03_report.txt.

---

## 8. End-to-End Pipeline (run_pipeline.py)

Orchestrator script that runs all Day 03 modules in sequence:
Step 1 — ML Pipeline, Step 2 — Feature Engineering,
Step 3 — Dimensionality Reduction, Step 4 — Hyperparameter Tuning,
Step 5 — Ensemble Models.

---

## Overall Day 03 Summary

| Model | Accuracy | Type |
|-------|----------|------|
| XGBoost | 0.8250 | Ensemble — Best Overall |
| RF Tuned | 0.8187 | Hyperparameter Tuning |
| GB Tuned | 0.8156 | Hyperparameter Tuning |
| RF Pipeline | 0.8031 | Pipeline |
| LightGBM | 0.7906 | Ensemble |
| GB Pipeline | 0.7906 | Pipeline |
| PyTorch NN | 0.7594 | Deep Learning |
| LDA | 0.7125 | Dim Reduction |
| PCA | 0.7094 | Dim Reduction |
| SVD | 0.7094 | Dim Reduction |

**Best Model Overall:** XGBoost (Accuracy 0.8250)
**Best After Tuning:** Random Forest (Accuracy 0.8187)
**Best Feature:** alcohol_density (engineered feature)
**Best Dim Reduction:** LDA (0.7125 with 1 component)
**Feature Engineering:** +1.56% accuracy improvement

---

## Key Findings Across All Models

- Alcohol content is the most important wine quality predictor
- Feature engineering improved accuracy by +1.56%
- Hyperparameter tuning improved Random Forest by +1.57%
- Ensemble models outperform neural networks on small datasets
- LDA is most efficient dimensionality reduction technique
- Production pipelines prevent data leakage — industry standard
- Neural network loss decreased steadily showing correct learning

---

## Tools Used

Python | Pandas | NumPy | Scikit-learn | XGBoost | LightGBM
PyTorch | Matplotlib | Seaborn
