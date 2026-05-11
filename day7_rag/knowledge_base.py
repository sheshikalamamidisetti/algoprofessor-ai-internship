"""
knowledge_base.py
-----------------
Statistics and ML knowledge base for RAG.
Contains documents covering hypothesis testing, regression,
time series, feature engineering — directly aligned with
IIT Indore AI & DS curriculum.

Usage:
    from knowledge_base import DS_KNOWLEDGE_BASE, get_all_texts
"""

DS_KNOWLEDGE_BASE = [
    {
        "id": "stat_01",
        "category": "statistics",
        "title": "Hypothesis Testing",
        "content": (
            "A hypothesis test evaluates two mutually exclusive statements about a population. "
            "H0 (null hypothesis) states no effect exists. H1 (alternative) states an effect exists. "
            "The p-value is the probability of observing results as extreme as the data assuming H0 is true. "
            "If p-value < alpha (usually 0.05), we reject H0. "
            "Type I error: rejecting a true H0 (false positive). Alpha controls this. "
            "Type II error: failing to reject a false H0 (false negative). Beta controls this. "
            "Statistical power = 1 - beta. Common tests: t-test, chi-square, ANOVA, Mann-Whitney U. "
            "Effect size measures practical significance: Cohen's d for t-tests, eta-squared for ANOVA."
        ),
    },
    {
        "id": "stat_02",
        "category": "statistics",
        "title": "Confidence Intervals",
        "content": (
            "A 95% confidence interval means if we repeated the experiment 100 times, "
            "95 of the resulting intervals would contain the true population parameter. "
            "Formula: CI = point estimate plus or minus critical value times standard error. "
            "For proportions: CI = p_hat plus or minus z times sqrt(p_hat times (1-p_hat) divided by n). "
            "Wider CI means less precision. Larger sample size gives narrower CI. "
            "A CI that does not include zero indicates statistical significance at that alpha level. "
            "Bootstrap CI is used when distribution assumptions are violated."
        ),
    },
    {
        "id": "stat_03",
        "category": "statistics",
        "title": "ANOVA and Post-hoc Tests",
        "content": (
            "ANOVA (Analysis of Variance) tests whether means differ across three or more groups. "
            "F-statistic = variance between groups divided by variance within groups. "
            "If p < 0.05, at least one group mean differs significantly from the others. "
            "ANOVA does not tell which groups differ — use post-hoc tests for that. "
            "Tukey HSD: compares all pairs, controls family-wise error rate. "
            "Bonferroni: most conservative, divides alpha by number of comparisons. "
            "Assumptions: normality within groups, equal variances (Levene test), independence. "
            "Kruskal-Wallis is the non-parametric alternative when normality is violated."
        ),
    },
    {
        "id": "ml_01",
        "category": "machine_learning",
        "title": "Overfitting and Regularisation",
        "content": (
            "Overfitting occurs when a model learns noise in training data and fails to generalise. "
            "Signs: train accuracy much higher than test accuracy, gap greater than 15 percent is concerning. "
            "Causes: model too complex, insufficient training data, no regularisation applied. "
            "L1 regularisation (Lasso): adds sum of absolute weights to loss, drives some weights to zero, "
            "useful for feature selection. "
            "L2 regularisation (Ridge): adds sum of squared weights to loss, shrinks all weights, "
            "does not produce sparsity. "
            "Dropout: randomly deactivates neurons during training, prevents co-adaptation. "
            "Cross-validation: k-fold CV gives unbiased estimate of generalisation performance. "
            "Early stopping: halt training when validation loss stops improving."
        ),
    },
    {
        "id": "ml_02",
        "category": "machine_learning",
        "title": "Feature Importance and Selection",
        "content": (
            "Random Forest feature importance measures mean decrease in impurity across all trees. "
            "Limitation: biased toward high-cardinality and continuous features. "
            "Permutation importance: more reliable, measures score drop when feature is shuffled randomly. "
            "SHAP (SHapley Additive exPlanations): game-theoretic approach explaining individual predictions. "
            "Filter methods: correlation, mutual information — fast, model-independent, used as preprocessing. "
            "Wrapper methods: Recursive Feature Elimination uses model performance as selection criterion. "
            "For time series: always check that features do not use future information (data leakage). "
            "Variance Inflation Factor measures multicollinearity between features."
        ),
    },
    {
        "id": "ml_03",
        "category": "machine_learning",
        "title": "Random Forest and Ensemble Methods",
        "content": (
            "Random Forest builds multiple decision trees using bootstrap sampling and random feature subsets. "
            "Each tree votes and the majority class wins for classification. "
            "Key hyperparameters: n_estimators (more is better up to a point), "
            "max_depth (controls complexity), min_samples_split, max_features. "
            "Out-of-bag (OOB) error provides a free internal validation estimate. "
            "XGBoost uses gradient boosting: trees are built sequentially, each correcting previous errors. "
            "LightGBM: faster than XGBoost using leaf-wise growth and histogram-based splitting. "
            "Bagging reduces variance. Boosting reduces bias. Stacking combines different model families."
        ),
    },
    {
        "id": "ts_01",
        "category": "time_series",
        "title": "Time Series Stationarity and ARIMA",
        "content": (
            "A stationary time series has constant mean, variance, and autocorrelation structure over time. "
            "ARIMA models require stationarity. "
            "ADF (Augmented Dickey-Fuller) test: if p < 0.05, series is stationary. "
            "To make stationary: first differencing subtracts lag-1 value, "
            "log transform stabilises variance, seasonal differencing removes seasonal patterns. "
            "ACF (AutoCorrelation Function) shows correlation at each lag. "
            "PACF (Partial AutoCorrelation Function) shows correlation controlling for intermediate lags. "
            "AR(p) process: PACF cuts off at lag p. MA(q) process: ACF cuts off at lag q. "
            "ARIMA(p,d,q): p=AR order, d=differencing order, q=MA order."
        ),
    },
    {
        "id": "ts_02",
        "category": "time_series",
        "title": "Time Series Decomposition and Forecasting",
        "content": (
            "Time series = trend + seasonality + residual (additive) or product of these (multiplicative). "
            "STL decomposition separates these components robustly. "
            "Seasonal strength = 1 minus variance of remainder divided by variance of seasonal plus remainder. "
            "Evaluation metrics: MAE (mean absolute error, interpretable in original units), "
            "RMSE (penalises large errors more heavily), MAPE (percentage error for comparison across scales), "
            "sMAPE (symmetric version avoiding division by zero issues). "
            "Walk-forward validation: always use time-ordered train/test splits, never random shuffle. "
            "Naive baselines to beat: last-value forecast, seasonal naive, moving average."
        ),
    },
    {
        "id": "dl_01",
        "category": "deep_learning",
        "title": "Neural Network Fundamentals",
        "content": (
            "A neural network consists of input, hidden, and output layers of connected neurons. "
            "Forward pass: inputs multiply weights, add bias, pass through activation function. "
            "ReLU activation: max(0, x), avoids vanishing gradient, most common for hidden layers. "
            "Sigmoid: outputs 0 to 1, used for binary classification output. "
            "Softmax: converts logits to probabilities summing to 1, for multi-class output. "
            "Backpropagation computes gradients using chain rule. "
            "Adam optimiser: adaptive learning rates, combines momentum and RMSProp. "
            "Batch normalisation: normalises layer inputs, speeds training, reduces sensitivity to init. "
            "Learning rate is the most critical hyperparameter to tune."
        ),
    },
    {
        "id": "eda_01",
        "category": "eda",
        "title": "Exploratory Data Analysis",
        "content": (
            "EDA is the process of understanding a dataset before modelling. "
            "Start with: shape (rows and columns), dtypes, memory usage. "
            "Check missing values: df.isnull().sum(), visualise with heatmap. "
            "Univariate analysis: histograms for numeric, bar charts for categorical. "
            "Bivariate analysis: scatter plots, correlation matrix, box plots by category. "
            "Outlier detection: IQR method (values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR), Z-score. "
            "Class imbalance: check target distribution early, plan SMOTE or class weights if needed. "
            "Feature correlation: Pearson for numeric-numeric, Cramer V for categorical-categorical."
        ),
    },
]


def get_all_texts() -> list[str]:
    """Return all document texts for embedding."""
    return [f"{d['title']}: {d['content']}" for d in DS_KNOWLEDGE_BASE]


def get_by_category(category: str) -> list[dict]:
    """Return documents filtered by category."""
    return [d for d in DS_KNOWLEDGE_BASE if d["category"] == category]


def get_by_id(doc_id: str) -> dict | None:
    """Return a single document by ID."""
    for d in DS_KNOWLEDGE_BASE:
        if d["id"] == doc_id:
            return d
    return None


if __name__ == "__main__":
    print(f"Knowledge base: {len(DS_KNOWLEDGE_BASE)} documents")
    categories = {}
    for d in DS_KNOWLEDGE_BASE:
        categories[d["category"]] = categories.get(d["category"], 0) + 1
    for cat, count in categories.items():
        print(f"  {cat}: {count} docs")
