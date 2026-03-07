"""
Day 03 — PyTorch Neural Network from Scratch (Wine Quality Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To build a Neural Network completely from scratch using PyTorch
on the Wine Quality dataset — implementing forward pass, backward
propagation, loss calculation and training loop manually.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


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


def preprocess_data(df):
    print("\n--- 2. Preprocessing Data ---")
    X = df.drop(["quality", "quality_label"], axis=1).values
    y = df["quality_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_train_t = torch.FloatTensor(X_train)
    X_test_t  = torch.FloatTensor(X_test)
    y_train_t = torch.FloatTensor(y_train)
    y_test_t  = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print(f"Train set : {X_train.shape[0]} samples")
    print(f"Test  set : {X_test.shape[0]} samples")
    print(f"Features  : {X_train.shape[1]}")
    print("Preprocessing completed\n")

    return train_loader, X_test_t, y_test_t, y_test


# Neural Network Architecture from Scratch
class WineQualityNN(nn.Module):
    def __init__(self, input_size):
        super(WineQualityNN, self).__init__()

        # Layer 1 — Input to Hidden
        self.layer1 = nn.Linear(input_size, 64)
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(0.3)

        # Layer 2 — Hidden to Hidden
        self.layer2 = nn.Linear(64, 32)
        self.relu2  = nn.ReLU()
        self.drop2  = nn.Dropout(0.3)

        # Layer 3 — Hidden to Hidden
        self.layer3 = nn.Linear(32, 16)
        self.relu3  = nn.ReLU()

        # Output Layer
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(self.relu1(self.layer1(x)))
        x = self.drop2(self.relu2(self.layer2(x)))
        x = self.relu3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x


def train_model(train_loader, input_size, epochs=50):
    print("--- 3. Training Neural Network from Scratch ---")
    print(f"Architecture: {input_size} -> 64 -> 32 -> 16 -> 1")
    print(f"Activation  : ReLU (hidden) + Sigmoid (output)")
    print(f"Loss        : Binary Cross Entropy")
    print(f"Optimizer   : Adam (lr=0.001)")
    print(f"Epochs      : {epochs}\n")

    model     = WineQualityNN(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accs   = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct    = 0
        total      = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted   = (outputs >= 0.5).float()
            correct    += (predicted == y_batch).sum().item()
            total      += y_batch.size(0)

        avg_loss = epoch_loss / len(train_loader)
        avg_acc  = correct / total
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"Accuracy: {avg_acc:.4f}")

    return model, train_losses, train_accs


def evaluate_model(model, X_test_t, y_test_t, y_test):
    print("\n--- 4. Evaluating Model ---")
    model.eval()
    with torch.no_grad():
        outputs    = model(X_test_t).squeeze()
        predicted  = (outputs >= 0.5).float().numpy()
        y_prob     = outputs.numpy()

    accuracy = accuracy_score(y_test, predicted)
    print(f"Test Accuracy : {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predicted,
          target_names=["Bad Wine", "Good Wine"]))

    return predicted, y_prob, accuracy


def visualize_results(train_losses, train_accs,
                      y_test, predicted, base_dir):
    print("\n--- 5. Visualizing Results ---")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "PyTorch Neural Network from Scratch — Wine Quality Dataset\n"
        "Sheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
        fontsize=13, fontweight="bold"
    )

    # Chart 1 - Training Loss
    axes[0].plot(train_losses, color="red", lw=2)
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Chart 2 - Training Accuracy
    axes[1].plot(train_accs, color="blue", lw=2)
    axes[1].set_title("Training Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(True, alpha=0.3)

    # Chart 3 - Confusion Matrix
    cm = confusion_matrix(y_test, predicted)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bad Wine", "Good Wine"],
                yticklabels=["Bad Wine", "Good Wine"],
                ax=axes[2])
    axes[2].set_title("Confusion Matrix", fontweight="bold")
    axes[2].set_xlabel("Predicted Label")
    axes[2].set_ylabel("Actual Label")

    plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=3.0)

    save_path = os.path.join(output_dir, "pytorch_nn_results.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Chart saved to: {save_path}")
    plt.show()
    plt.close()


def run_pytorch_nn():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df, base_dir = load_data()
    if df is None:
        return

    train_loader, X_test_t, y_test_t, y_test = preprocess_data(df)
    input_size = X_test_t.shape[1]

    model, train_losses, train_accs = train_model(
        train_loader, input_size, epochs=50
    )
    predicted, y_prob, accuracy = evaluate_model(
        model, X_test_t, y_test_t, y_test
    )
    visualize_results(train_losses, train_accs,
                      y_test, predicted, base_dir)

    print("\nDay 03 PyTorch NN from Scratch completed successfully.")


if __name__ == "__main__":
    run_pytorch_nn()
