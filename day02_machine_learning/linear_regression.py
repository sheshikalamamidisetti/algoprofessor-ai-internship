"""
Day 02 — Linear Regression (Breast Cancer Dataset)

Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti

Objective:
To build a baseline Linear Regression model on the Breast Cancer dataset
by performing data preprocessing, model training, prediction, and evaluation
as part of the machine learning workflow.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load Dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "breast-cancer.csv")
df = pd.read_csv(file_path, header=None)
print("Dataset loaded successfully\n")

# Add Column Names
df.columns = [
    "age",
    "menopause",
    "tumor_size",
    "inv_nodes",
    "node_caps",
    "deg_malig",
    "breast",
    "breast_quad",
    "irradiat",
    "class"
]

# Encode Categorical Data
encoder = LabelEncoder()
for column in df.columns:
    df[column] = encoder.fit_transform(df[column].astype(str))
print("Categorical encoding completed\n")

# Features & Target
X = df.drop("class", axis=1)
y = df["class"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction & Evaluation
predictions = model.predict(X_test)
predictions_rounded = np.round(predictions).astype(int).clip(0, 1)
accuracy = accuracy_score(y_test, predictions_rounded)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Visualize AND Save
output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Linear Regression — Breast Cancer Classification\nSheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
    fontsize=13, fontweight="bold", y=1.02
)

# Chart 1 - Actual vs Predicted
axes[0, 0].scatter(y_test, predictions, color="blue", alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                color="red", linestyle="--", lw=2)
axes[0, 0].set_title("Actual vs Predicted", fontweight="bold")
axes[0, 0].set_xlabel("Actual Values")
axes[0, 0].set_ylabel("Predicted Values")
axes[0, 0].grid(True, alpha=0.3)

# Chart 2 - Residuals Plot
residuals = y_test - predictions
axes[0, 1].scatter(predictions, residuals, color="purple", alpha=0.6)
axes[0, 1].axhline(y=0, color="red", linestyle="--", lw=2)
axes[0, 1].set_title("Residuals Plot", fontweight="bold")
axes[0, 1].set_xlabel("Predicted Values")
axes[0, 1].set_ylabel("Residuals")
axes[0, 1].grid(True, alpha=0.3)

# Chart 3 - Feature Importance
feature_names = X.columns.tolist()
coefficients = model.coef_
sorted_idx = np.argsort(np.abs(coefficients))[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_coefs = coefficients[sorted_idx]
colors = ["blue" if c > 0 else "red" for c in sorted_coefs]
axes[1, 0].barh(sorted_features, sorted_coefs, color=colors)
axes[1, 0].axvline(x=0, color="black", linewidth=0.8, linestyle="--")
axes[1, 0].set_title("Feature Importance (Coefficients)", fontweight="bold")
axes[1, 0].set_xlabel("Coefficient Value")
axes[1, 0].grid(True, alpha=0.3, axis="x")

# Chart 4 - Prediction Distribution
axes[1, 1].hist(predictions, bins=20, color="green",
                edgecolor="white", alpha=0.8)
axes[1, 1].set_title("Prediction Distribution", fontweight="bold")
axes[1, 1].set_xlabel("Predicted Values")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(hspace=0.4, wspace=0.3)

save_path = os.path.join(output_dir, "linear_regression_results.png")
plt.savefig(save_path)
print(f"\nGraph saved to: {save_path}")
plt.show()
plt.close()

print("\nDay 02 ML workflow completed successfully.")
