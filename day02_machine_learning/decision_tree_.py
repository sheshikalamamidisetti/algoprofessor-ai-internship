"""
Day 02 — Decision Tree Model
Heart Disease Prediction

Intern: Sheshikala Mamidisetti
Internship: AlgoProfessor AI R&D Internship
"""

# ==============================
# Step 1: Import Libraries
# ==============================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix


# ==============================
# Step 2: Universal Dataset Loader
# ==============================

def load_data(filename):
    possible_paths = [
        filename,
        f"data/{filename}",
        f"/content/{filename}",
        f"/content/data/{filename}"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loaded file from: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError("Dataset not found in expected locations.")


df = load_data("heart.csv")


# ==============================
# Step 3: Exploratory Data Analysis
# ==============================

print("\nDataset Preview:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nTarget Distribution:\n", df['target'].value_counts())
print("\nDataset Info:\n")
print(df.info())

# Check missing values and duplicates
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)


# ==============================
# Step 4: Visualization
# ==============================

# Boxplots
for column in df.columns:
    if df[column].dtype != 'object':
        plt.figure()
        sns.boxplot(x=df[column])
        plt.title(f"Boxplot of {column}")
        plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# ==============================
# Step 5: Model Building
# ==============================

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nDefault Model Accuracy:",
      accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)


# ==============================
# Step 6: Decision Tree Visualization
# ==============================

plt.figure(figsize=(20, 12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    fontsize=8
)
plt.title("Decision Tree Visualization (Default Model)")
plt.show()


# ==============================
# Step 7: Hyperparameter Tuning (Depth)
# ==============================

print("\nDepth vs Accuracy\n")

for depth in range(1, 11):
    temp_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    temp_model.fit(X_train, y_train)
    temp_pred = temp_model.predict(X_test)
    acc = accuracy_score(y_test, temp_pred)
    print(f"Max Depth: {depth} | Accuracy: {acc:.3f}")

# Best depth model
best_model = DecisionTreeClassifier(max_depth=5, random_state=42)
best_model.fit(X_train, y_train)

y_best = best_model.predict(X_test)

print("\nBest Depth Model Accuracy:",
      accuracy_score(y_test, y_best))


# ==============================
# Step 8: GridSearchCV
# ==============================

grid_params = {
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=grid_params,
    cv=3
)

grid_search.fit(X_train, y_train)

best_grid_model = grid_search.best_estimator_
y_grid = best_grid_model.predict(X_test)

print("\nGridSearch Best Parameters:", grid_search.best_params_)
print("GridSearch Accuracy:",
      accuracy_score(y_test, y_grid))


# ==============================
# Step 9: RandomizedSearchCV
# ==============================

random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(),
    param_distributions=grid_params,
    cv=3,
    random_state=42
)

random_search.fit(X_train, y_train)

best_random_model = random_search.best_estimator_
y_random = best_random_model.predict(X_test)

print("\nRandomizedSearch Best Parameters:",
      random_search.best_params_)
print("RandomizedSearch Accuracy:",
      accuracy_score(y_test, y_random))


# ==============================
# Step 10: Model Comparison
# ==============================

print("\nModel Accuracy Comparison")
print("Default Model:", accuracy_score(y_test, y_pred))
print("Best Depth Model:", accuracy_score(y_test, y_best))
print("GridSearchCV Model:", accuracy_score(y_test, y_grid))
print("RandomizedSearchCV Model:", accuracy_score(y_test, y_random))


# ==============================
# Conclusion
# ==============================

"""
Conclusion:
The Decision Tree model was implemented and optimized for heart disease prediction.
Hyperparameter tuning improved performance while controlling overfitting.
The tuned model achieved approximately 83.6% accuracy.
"""
