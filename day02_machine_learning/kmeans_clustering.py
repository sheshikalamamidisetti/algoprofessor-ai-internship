"""
Day 02 — K-Means Clustering (Heart Disease Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To implement K-Means clustering on a real-world medical dataset
including preprocessing, cluster analysis, evaluation using
silhouette score and visualization of results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load Dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "heart.csv")
df = pd.read_csv(file_path)
print("Dataset loaded successfully\n")

# Explore Data
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Handle Missing Values
df.fillna(df.mode().iloc[0], inplace=True)
print("\nMissing values handled\n")

# Features only (KMeans is unsupervised — no target needed)
X = df.drop("target", axis=1)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Feature scaling completed\n")

# Find Best K using Elbow Method
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

print("Elbow method completed\n")

# Train Final Model with K=2
model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X_scaled)
labels = model.labels_
final_score = silhouette_score(X_scaled, labels)

print(f"Final K: 2")
print(f"Silhouette Score: {final_score:.4f}")
print(f"\nCluster Distribution:")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Cluster {u}: {c} samples")

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("\nPCA completed for visualization\n")

# Visualize AND Save
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "K-Means Clustering — Heart Disease Dataset\nSheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
    fontsize=13, fontweight="bold", y=1.02
)

# Chart 1 - Elbow Curve
axes[0, 0].plot(k_range, inertia, marker="o", color="blue", lw=2)
axes[0, 0].set_title("Elbow Method — Optimal K", fontweight="bold")
axes[0, 0].set_xlabel("Number of Clusters (K)")
axes[0, 0].set_ylabel("Inertia")
axes[0, 0].grid(True, alpha=0.3)

# Chart 2 - Silhouette Scores
axes[0, 1].plot(k_range, silhouette_scores, marker="o",
                color="green", lw=2)
axes[0, 1].set_title("Silhouette Score vs K", fontweight="bold")
axes[0, 1].set_xlabel("Number of Clusters (K)")
axes[0, 1].set_ylabel("Silhouette Score")
axes[0, 1].grid(True, alpha=0.3)

# Chart 3 - Cluster Visualization (PCA)
colors = ["blue", "red"]
for i in range(2):
    mask = labels == i
    axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors[i], label=f"Cluster {i}",
                       alpha=0.6, s=50)
axes[1, 0].set_title("Cluster Visualization (PCA 2D)",
                      fontweight="bold")
axes[1, 0].set_xlabel("PCA Component 1")
axes[1, 0].set_ylabel("PCA Component 2")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Chart 4 - Cluster Distribution
cluster_counts = pd.Series(labels).value_counts().sort_index()
axes[1, 1].bar([f"Cluster {i}" for i in cluster_counts.index],
               cluster_counts.values,
               color=["blue", "red"], width=0.5)
axes[1, 1].set_title("Cluster Distribution", fontweight="bold")
axes[1, 1].set_xlabel("Cluster")
axes[1, 1].set_ylabel("Number of Samples")
axes[1, 1].grid(True, alpha=0.3, axis="y")
for i, count in enumerate(cluster_counts.values):
    axes[1, 1].text(i, count + 1, str(count),
                    ha="center", fontweight="bold", fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(hspace=0.4, wspace=0.3)

save_path = os.path.join(output_dir, "kmeans_results.png")
plt.savefig(save_path)
print(f"Graph saved to: {save_path}")
plt.show()
plt.close()

print("\nDay 02 K-Means workflow completed successfully.")
