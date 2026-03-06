# AlgoProfessor AI R&D Internship — Phase 1

Intern: Sheshikala Mamidisetti  
Batch: 2026  

---

# Breast Cancer Dataset — Classification Models

## 1. Linear Regression (Baseline Model)

### Objective
Implemented Linear Regression as a baseline model to establish the preprocessing pipeline and evaluate how a regression-based approach performs on a classification-type dataset.

### Workflow
- Data loading and preprocessing  
- Handling categorical variables  
- Train-test split  
- Model training and evaluation  

### Results
- Accuracy: 81%  
- Served as an initial benchmark before applying classification algorithms  

### Dataset
Breast Cancer dataset (`breast-cancer.csv`)  
- 286 rows  
- 9 features  
- Target variable: recurrence / no recurrence  

### Key Insight
Linear Regression provided a baseline reference but is not ideal for classification problems.

---

## 2. Logistic Regression

### Objective
Applied Logistic Regression for binary classification including preprocessing, training, evaluation, and result visualization.

### Workflow
- Feature preprocessing  
- Train-test split  
- Model training  
- Evaluation using Accuracy and ROC-AUC  
- Visualization of results  

### Results
- Accuracy: 69%  
- ROC-AUC: 0.60  

### Output
`outputs/logistic_regression_results.png`

### Key Insight
Logistic Regression provided probabilistic outputs suitable for classification, though performance was moderate on this dataset.

---

# Heart Disease Dataset — Predictive Modeling

Dataset: `heart.csv`  
- 303 rows  
- 13 features  
- Target: presence or absence of heart disease  
- Source: UCI Machine Learning Repository  

---

## 1. Decision Tree Classifier

### Objective
Built and optimized a Decision Tree classifier to predict heart disease using supervised machine learning techniques.

### Workflow
- Exploratory Data Analysis (statistical summary, correlation heatmap, boxplots)  
- Train-test split  
- Model training  
- Hyperparameter tuning (max_depth)  
- Evaluation and visualization  

### Model Performance
- Initial Accuracy: ~0.80  
- Best Accuracy: 0.836  
- Optimal max_depth: 5  
- GridSearchCV and RandomizedSearchCV used for optimization  

### Observations
- Moderate depth improved accuracy  
- Very deep trees increased complexity without significant gain  
- Overfitting risk increases with high depth  
- Controlled complexity improves interpretability  

---

## 2. Random Forest Classifier

### Objective
Developed a Random Forest classifier to improve predictive performance using ensemble learning.

### Model Configuration
- n_estimators: 100  
- max_depth: 5  

### Evaluation Metrics

| Metric   | Score |
|----------|-------|
| Accuracy | 0.80  |
| ROC-AUC  | 0.91  |

### Output
`outputs/random_forest_results.png`

### Key Insight
Random Forest improved overall predictive stability and achieved a high ROC-AUC score, demonstrating the effectiveness of ensemble methods over single-tree models.

---

# Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Jupyter Notebook  

---
#Support Vector Machine (SVM) Classifier

#Objective
Applied Support Vector Machine (SVM) for binary classification to identify an optimal decision boundary between classes.

#Workflow
Data preprocessing and feature scaling
Train-test split
SVM model training
Evaluation using Accuracy and ROC-AUC

#Results
Accuracy: ~0.82
ROC-AUC: ~0.88

Key Insight
SVM showed strong performance on the dataset when proper feature scaling was applied, making it an effective classifier for structured healthcare data.
# Day 02 — K-Means Clustering (Heart Disease Dataset)

## Objective
Built a K-Means clustering model on the Heart Disease dataset
covering preprocessing, elbow method to find optimal K, cluster
analysis, PCA visualization and silhouette score evaluation.

## Model
K-Means Clustering — n_clusters: 2 | Scikit-learn

## Result
| Metric | Score |
|--------|-------|
| Silhouette Score | 0.167 |
| Optimal K | 2 |
| Cluster 0 | 195 samples |
| Cluster 1 | 108 samples |

## Key Finding
Silhouette Score was highest at K=2 (0.167) confirming 2 clusters
is optimal — aligning with binary nature of heart disease data.
PCA visualization shows clear separation between the two clusters.

## Dataset
Heart Disease Dataset — heart.csv — 303 rows — 13 features
Source: UCI Machine Learning Repository

## Tools Used
Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn

## Output
kmeans_results.png — saved in outputs/

# Day 02 — Model Comparison (Heart Disease Dataset)

## Objective
Compared all supervised classification models trained in Day 02
on the Heart Disease dataset across multiple metrics — Accuracy,
ROC-AUC, Precision, Recall and F1-Score to identify the best
model with reasoning and recommendation.

## Models Compared

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score | Train Time |
|-------|----------|---------|-----------|--------|----------|------------|
| Logistic Regression | 0.8033 | 0.8690 | 0.7692 | 0.9091 | 0.8333 | 0.024s |
| Decision Tree | 0.7869 | 0.8176 | 0.7500 | 0.9091 | 0.8219 | 0.082s |
| Random Forest | 0.8033 | 0.9113 | 0.7561 | 0.9394 | 0.8378 | 1.809s |
| SVM | 0.8197 | 0.8831 | 0.7750 | 0.9394 | 0.8493 | 0.090s |

## Best Model Recommendation

| Metric | Best Model | Score |
|--------|-----------|-------|
| Accuracy | SVM | 0.8197 |
| ROC-AUC | Random Forest | 0.9113 |
| F1-Score | SVM | 0.8493 |
| Fastest Training | Logistic Regression | 0.024s |

**Final Recommendation: SVM**
SVM gives highest accuracy (0.82) and best F1-Score (0.849)
with strong ROC-AUC (0.883) and fast training time (0.090s).
Best choice for heart disease classification task.

## Key Findings
- SVM is best overall — highest accuracy and F1-Score
- Random Forest has best ROC-AUC (0.911) but slowest training (1.8s)
- Logistic Regression is fastest to train (0.024s) with good accuracy
- Decision Tree is weakest overall across all metrics
- All models achieved above 0.78 accuracy on Heart Disease dataset

## Note
Linear Regression was used as a baseline regression model on the
Breast Cancer dataset — not included here as it solves a different
problem type (regression vs classification).

K-Means Clustering is an unsupervised model evaluated separately
using Silhouette Score — cannot be compared using accuracy metrics.

## Dataset
Heart Disease Dataset — heart.csv — 303 rows — 13 features
Source: UCI Machine Learning Repository

## Tools Used
Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn

## Output
model_comparison_results.png — saved in outputs/
- Accuracy Comparison Bar Chart
- ROC Curves — All Models Together
- All Metrics Heatmap
- F1-Score Comparison Bar Chart
