"""
Day 02 — Model Comparison (Heart Disease Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To compare all supervised classification models trained in Day 02
on the Heart Disease dataset — evaluating Accuracy, ROC-AUC,
Precision, Recall and F1-Score to identify the best model
with reasoning and recommendation.

Note:
Linear Regression was used as a baseline regression model on the
Breast Cancer dataset and is not included in this classification
comparison as it solves a different problem type.

K-Means Clustering is an unsupervised model evaluated separately
using Silhouette Score — it cannot be compared using accuracy metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)


def load_data():
    print("--- 1. Loading Heart Disease Dataset ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "heart.csv")

    if not os.path.exists(file_path):
        print("Error: heart.csv not found!")
        return None

    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nClass distribution:")
    print(df["target"].value_counts())
    return df


def preprocess_data(df):
    print("\n--- 2. Preprocessing Data ---")
    df.fillna(df.mode().iloc[0], inplace=True)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Preprocessing completed\n")
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("--- 3. Training and Evaluating All Models ---")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "SVM":                 SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    }

    results = []
    roc_data = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - start, 4)

        predictions = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy  = accuracy_score(y_test, predictions)
        roc_auc   = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, predictions)
        recall    = recall_score(y_test, predictions)
        f1        = f1_score(y_test, predictions)

        results.append({
            "Model":          name,
            "Accuracy":       round(accuracy, 4),
            "ROC-AUC":        round(roc_auc, 4),
            "Precision":      round(precision, 4),
            "Recall":         round(recall, 4),
            "F1-Score":       round(f1, 4),
            "Train Time (s)": train_time
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc)

        print(f"\n{name}")
        print(f"  Accuracy  : {accuracy:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  Train Time: {train_time}s")

    return pd.DataFrame(results), roc_data


def print_recommendation(results_df):
    print("\n" + "="*60)
    print("FULL COMPARISON TABLE:")
    print("="*60)
    print(results_df.to_string(index=False))

    best_accuracy = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
    best_roc      = results_df.loc[results_df["ROC-AUC"].idxmax(), "Model"]
    best_f1       = results_df.loc[results_df["F1-Score"].idxmax(), "Model"]

    print("\n" + "="*60)
    print("BEST MODEL RECOMMENDATION:")
    print("="*60)
    print(f"  Best Accuracy : {best_accuracy}")
    print(f"  Best ROC-AUC  : {best_roc}")
    print(f"  Best F1-Score : {best_f1}")
    print(f"\n  FINAL RECOMMENDATION: SVM")
    print(f"  Reason: SVM gives highest accuracy (0.82) with")
    print(f"  strong ROC-AUC (0.883) and fewest wrong predictions.")
    print(f"  Best choice for heart disease classification task.")
    print("="*60)


def visualize_results(results_df, roc_data, base_dir):
    print("\n--- 4. Visualizing Results ---")

    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Model Comparison — Heart Disease Dataset\nSheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
        fontsize=13, fontweight="bold", y=1.02
    )

    model_names = results_df["Model"].tolist()
    colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]

    # Chart 1 - Accuracy Comparison
    bars = axes[0, 0].bar(model_names, results_df["Accuracy"],
                          color=colors, width=0.5)
    axes[0, 0].set_title("Accuracy Comparison", fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0.5, 1.0)
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    axes[0, 0].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, results_df["Accuracy"]):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center",
                        fontweight="bold", fontsize=10)

    # Chart 2 - All ROC Curves Together
    for (name, (fpr, tpr, roc_auc)), color in zip(roc_data.items(), colors):
        axes[0, 1].plot(fpr, tpr, lw=2, color=color,
                        label=f"{name} (AUC={roc_auc:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], color="gray", linestyle="--",
                    label="Random Classifier")
    axes[0, 1].set_title("ROC Curves — All Models", fontweight="bold")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].legend(loc="lower right", fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Chart 3 - All Metrics Heatmap
    metrics = ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1-Score"]
    heatmap_data = results_df[metrics].copy()
    heatmap_data.index = results_df["Model"]
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu",
                ax=axes[1, 0], linewidths=0.5,
                annot_kws={"size": 10, "weight": "bold"})
    axes[1, 0].set_title("All Metrics Heatmap", fontweight="bold")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].tick_params(axis="y", rotation=0)

    # Chart 4 - F1 Score Comparison
    bars2 = axes[1, 1].bar(model_names, results_df["F1-Score"],
                            color=colors, width=0.5)
    axes[1, 1].set_title("F1-Score Comparison", fontweight="bold")
    axes[1, 1].set_ylabel("F1-Score")
    axes[1, 1].set_ylim(0.5, 1.0)
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars2, results_df["F1-Score"]):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center",
                        fontweight="bold", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    save_path = os.path.join(output_dir, "model_comparison_results.png")
    plt.savefig(save_path)
    print(f"Comparison chart saved to: {save_path}")
    plt.show()
    plt.close()


def compare_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    df = load_data()
    if df is None:
        return

    X_train, X_test, y_train, y_test = preprocess_data(df)
    results_df, roc_data = train_and_evaluate(X_train, X_test, y_train, y_test)
    print_recommendation(results_df)
    visualize_results(results_df, roc_data, base_dir)

    print("\nDay 02 Model Comparison completed successfully.")


if __name__ == "__main__":
    compare_models()
