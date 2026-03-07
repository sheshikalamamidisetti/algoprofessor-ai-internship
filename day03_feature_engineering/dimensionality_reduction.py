"""
Day 03 — Dimensionality Reduction: PCA, LDA, SVD (Wine Quality Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To implement and compare three dimensionality reduction techniques —
PCA (Principal Component Analysis), LDA (Linear Discriminant Analysis)
and SVD (Singular Value Decomposition) on the Wine Quality dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
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
    print(f"Features: {df.shape[1] - 2}")
    return df, base_dir


def preprocess_data(df):
    print("\n--- 2. Preprocessing Data ---")
    X = df.drop(["quality", "quality_label"], axis=1)
    y = df["quality_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Original features : {X.shape[1]}")
    print("Scaling completed\n")
    return X_scaled, y, X.columns.tolist()


def apply_pca(X_scaled, y):
    print("--- 3. Applying PCA ---")
    pca = PCA()
    pca.fit(X_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Components to explain 95% variance : {n_components_95}")
    print(f"Explained variance per component   :")
    for i, v in enumerate(explained_variance[:5]):
        print(f"  PC{i+1}: {v*100:.2f}%")

    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    print(f"PCA 2D variance explained: {sum(pca_2d.explained_variance_ratio_)*100:.2f}%")

    # Accuracy with PCA components
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Accuracy with PCA (2 components)   : {acc:.4f}")

    return X_pca, explained_variance, cumulative_variance, acc


def apply_lda(X_scaled, y):
    print("\n--- 4. Applying LDA ---")
    lda = LDA(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y)

    explained = lda.explained_variance_ratio_
    print(f"LDA components    : 1 (binary classification max)")
    print(f"Variance explained: {explained[0]*100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X_lda, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Accuracy with LDA : {acc:.4f}")

    return X_lda, acc


def apply_svd(X_scaled, y):
    print("\n--- 5. Applying SVD ---")
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_svd = svd.fit_transform(X_scaled)

    explained = svd.explained_variance_ratio_
    print(f"SVD components    : 2")
    print(f"Variance explained: {sum(explained)*100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X_svd, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Accuracy with SVD : {acc:.4f}")

    return X_svd, acc


def print_comparison(pca_acc, lda_acc, svd_acc):
    print("\n" + "="*50)
    print("   DIMENSIONALITY REDUCTION COMPARISON")
    print("="*50)
    print(f"{'Method':<10} {'Components':<15} {'Accuracy'}")
    print(f"{'-'*40}")
    print(f"{'PCA':<10} {'2':<15} {pca_acc:.4f}")
    print(f"{'LDA':<10} {'1':<15} {lda_acc:.4f}")
    print(f"{'SVD':<10} {'2':<15} {svd_acc:.4f}")
    print("="*50)
    best = max([("PCA", pca_acc), ("LDA", lda_acc), ("SVD", svd_acc)],
               key=lambda x: x[1])
    print(f"Best Method: {best[0]} with accuracy {best[1]:.4f}")
    print("="*50)


def visualize_results(X_scaled, X_pca, X_lda, X_svd,
                      y, explained_variance,
                      cumulative_variance, base_dir):
    print("\n--- 6. Visualizing Results ---")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Dimensionality Reduction: PCA, LDA, SVD — Wine Quality Dataset\n"
        "Sheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
        fontsize=13, fontweight="bold"
    )

    colors = {0: "red", 1: "blue"}
    color_list = [colors[c] for c in y]

    # Chart 1 - PCA Explained Variance
    axes[0, 0].bar(range(1, len(explained_variance)+1),
                   explained_variance * 100, color="blue",
                   alpha=0.7, label="Individual")
    axes[0, 0].plot(range(1, len(explained_variance)+1),
                    cumulative_variance * 100, color="red",
                    marker="o", lw=2, label="Cumulative")
    axes[0, 0].axhline(y=95, color="green", linestyle="--",
                       lw=1.5, label="95% threshold")
    axes[0, 0].set_title("PCA — Explained Variance", fontweight="bold")
    axes[0, 0].set_xlabel("Principal Component")
    axes[0, 0].set_ylabel("Variance Explained (%)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Chart 2 - PCA 2D Scatter
    for label, color, name in [(0, "red", "Bad Wine"), (1, "blue", "Good Wine")]:
        mask = y == label
        axes[0, 1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=color, label=name, alpha=0.5, s=20)
    axes[0, 1].set_title("PCA — 2D Visualization", fontweight="bold")
    axes[0, 1].set_xlabel("Principal Component 1")
    axes[0, 1].set_ylabel("Principal Component 2")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Chart 3 - LDA 1D Distribution
    lda_bad  = X_lda[y == 0].flatten()
    lda_good = X_lda[y == 1].flatten()
    axes[1, 0].hist(lda_bad,  bins=30, color="red",
                    alpha=0.6, label="Bad Wine",  density=True)
    axes[1, 0].hist(lda_good, bins=30, color="blue",
                    alpha=0.6, label="Good Wine", density=True)
    axes[1, 0].set_title("LDA — Class Separation", fontweight="bold")
    axes[1, 0].set_xlabel("LDA Component")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Chart 4 - SVD 2D Scatter
    for label, color, name in [(0, "red", "Bad Wine"), (1, "blue", "Good Wine")]:
        mask = y == label
        axes[1, 1].scatter(X_svd[mask, 0], X_svd[mask, 1],
                           c=color, label=name, alpha=0.5, s=20)
    axes[1, 1].set_title("SVD — 2D Visualization", fontweight="bold")
    axes[1, 1].set_xlabel("SVD Component 1")
    axes[1, 1].set_ylabel("SVD Component 2")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=3.0)

    save_path = os.path.join(output_dir, "dimensionality_reduction_results.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Chart saved to: {save_path}")
    plt.show()
    plt.close()


def run_dimensionality_reduction():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df, base_dir = load_data()
    if df is None:
        return

    X_scaled, y, feature_names = preprocess_data(df)
    X_pca, explained_variance, cumulative_variance, pca_acc = apply_pca(X_scaled, y)
    X_lda, lda_acc = apply_lda(X_scaled, y)
    X_svd, svd_acc = apply_svd(X_scaled, y)
    print_comparison(pca_acc, lda_acc, svd_acc)
    visualize_results(X_scaled, X_pca, X_lda, X_svd,
                      y, explained_variance,
                      cumulative_variance, base_dir)

    print("\nDay 03 Dimensionality Reduction workflow completed successfully.")


if __name__ == "__main__":
    run_dimensionality_reduction()
