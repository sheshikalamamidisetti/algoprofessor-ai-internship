"""
Day 03 — Feature Engineering (Wine Quality Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To perform feature engineering on the Wine Quality dataset —
including feature creation, feature selection, correlation analysis
and identifying the most important features for model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    print("--- 1. Loading Wine Quality Dataset ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "winequality-red.csv")

    if not os.path.exists(file_path):
        print("Error: winequality-red.csv not found!")
        return None, None

    try:
        df = pd.read_csv(file_path, sep=";")
        if len(df.columns) < 5:
            df = pd.read_csv(file_path, sep=",")
    except Exception:
        df = pd.read_csv(file_path, sep=",")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["quality_label"] = (df["quality"] >= 6).astype(int)

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    return df, base_dir


def create_new_features(df):
    print("\n--- 2. Creating New Features ---")
    df["acidity_ratio"]   = df["fixed_acidity"] / (df["volatile_acidity"] + 0.001)
    df["sulfur_ratio"]    = df["free_sulfur_dioxide"] / (df["total_sulfur_dioxide"] + 0.001)
    df["alcohol_density"] = df["alcohol"] / df["density"]
    df["total_acidity"]   = df["fixed_acidity"] + df["volatile_acidity"] + df["citric_acid"]
    print("Created: acidity_ratio, sulfur_ratio, alcohol_density, total_acidity")
    print(f"Total features after engineering: {df.shape[1]}")
    return df


def correlation_analysis(df):
    print("\n--- 3. Correlation Analysis ---")
    numeric_df = df.drop(["quality"], axis=1)
    corr_with_target = numeric_df.corr()["quality_label"].drop("quality_label")
    corr_sorted = corr_with_target.abs().sort_values(ascending=False)
    print("\nTop 5 features most correlated with wine quality:")
    print(corr_sorted.head(5))
    return numeric_df


def select_best_features(df):
    print("\n--- 4. Feature Selection ---")
    X = df.drop(["quality", "quality_label"], axis=1)
    y = df["quality_label"]

    selector = SelectKBest(f_classif, k=5)
    selector.fit(X, y)
    best_features = X.columns[selector.get_support()].tolist()
    print(f"\nSelectKBest — Top 5 features: {best_features}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=5)
    rfe.fit(X_scaled, y)
    rfe_features = X.columns[rfe.support_].tolist()
    print(f"RFE — Top 5 features: {rfe_features}")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)
    print(f"Random Forest — Top 5: {importance_df.head(5)['Feature'].tolist()}")

    return X, y, importance_df, best_features


def compare_with_without_features(X, y):
    print("\n--- 5. Comparing Model With vs Without New Features ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    original_cols = list(range(11))
    rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_original.fit(X_train[:, original_cols], y_train)
    acc_original = accuracy_score(y_test, rf_original.predict(X_test[:, original_cols]))

    rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_all.fit(X_train, y_train)
    acc_all = accuracy_score(y_test, rf_all.predict(X_test))

    print(f"Accuracy WITHOUT new features : {acc_original:.4f}")
    print(f"Accuracy WITH    new features : {acc_all:.4f}")
    print(f"Improvement                   : {(acc_all - acc_original)*100:+.2f}%")
    return acc_original, acc_all


def visualize_results(df, numeric_df, importance_df,
                      best_features, acc_original,
                      acc_all, base_dir):
    print("\n--- 6. Visualizing Results ---")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Use only top 6 features for heatmap to avoid overlap
    top6 = ["alcohol", "sulphates", "volatile_acidity",
            "citric_acid", "alcohol_density", "quality_label"]
    corr_matrix = numeric_df[top6].corr()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "Feature Engineering — Wine Quality Dataset\n"
        "Sheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
        fontsize=14, fontweight="bold"
    )

    # Chart 1 - Correlation Heatmap (only top 6 features — clean)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[0, 0], linewidths=0.5,
                annot_kws={"size": 9},
                xticklabels=True, yticklabels=True)
    axes[0, 0].set_title("Correlation Heatmap (Top Features)", fontweight="bold", pad=10)
    axes[0, 0].tick_params(axis="x", rotation=30, labelsize=8)
    axes[0, 0].tick_params(axis="y", rotation=0, labelsize=8)

    # Chart 2 - Correlation with Target
    corr_target = numeric_df.corr()["quality_label"].drop("quality_label")
    corr_target = corr_target.sort_values(ascending=True)
    colors = ["red" if v < 0 else "blue" for v in corr_target.values]
    axes[0, 1].barh(corr_target.index, corr_target.values, color=colors)
    axes[0, 1].axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 1].set_title("Feature Correlation with Target", fontweight="bold", pad=10)
    axes[0, 1].set_xlabel("Correlation Value")
    axes[0, 1].tick_params(labelsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis="x")

    # Chart 3 - Feature Importance
    top_features = importance_df.head(10).sort_values("Importance", ascending=True)
    axes[1, 0].barh(top_features["Feature"], top_features["Importance"], color="green")
    axes[1, 0].set_title("Top 10 Feature Importance (Random Forest)", fontweight="bold", pad=10)
    axes[1, 0].set_xlabel("Importance Score")
    axes[1, 0].tick_params(labelsize=8)
    axes[1, 0].grid(True, alpha=0.3, axis="x")

    # Chart 4 - With vs Without New Features
    labels = ["Without\nNew Features", "With\nNew Features"]
    values = [acc_original, acc_all]
    bar_colors = ["#E65100", "#1565C0"]
    bars = axes[1, 1].bar(labels, values, color=bar_colors, width=0.4)
    axes[1, 1].set_title("Accuracy: With vs Without New Features", fontweight="bold", pad=10)
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_ylim(0.5, 1.0)
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center",
                        fontweight="bold", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4.0, w_pad=3.0)

    save_path = os.path.join(output_dir, "feature_engineering_results.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Chart saved to: {save_path}")
    plt.show()
    plt.close()


def run_feature_engineering():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df, base_dir = load_data()
    if df is None:
        return
    df = create_new_features(df)
    numeric_df = correlation_analysis(df)
    X, y, importance_df, best_features = select_best_features(df)
    acc_original, acc_all = compare_with_without_features(X, y)
    visualize_results(df, numeric_df, importance_df,
                      best_features, acc_original,
                      acc_all, base_dir)
    print("\nDay 03 Feature Engineering workflow completed successfully.")


if __name__ == "__main__":
    run_feature_engineering()
