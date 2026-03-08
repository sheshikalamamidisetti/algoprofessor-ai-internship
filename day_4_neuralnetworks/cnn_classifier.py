"""
Day 04 — CNN Classifier (MNIST Dataset)
Internship: AlgoProfessor AI R&D Internship
Intern: Sheshikala Mamidisetti
Objective:
To build a Convolutional Neural Network (CNN) for image
classification on the MNIST dataset using PyTorch —
implementing Conv layers, MaxPooling, BatchNorm and
Dropout for regularization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data():
    print("--- 1. Loading MNIST Dataset (Auto Download) ---")
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=os.path.join(base_dir, "data"),
                                   train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=os.path.join(base_dir, "data"),
                                   train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")
    print(f"Image shape   : 1 x 28 x 28")
    return train_loader, test_loader, base_dir


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


def train_model(train_loader, epochs=10):
    print("\n--- 2. Training CNN ---")
    print(f"Architecture  : Conv(32) -> Conv(64) -> Conv(128) -> FC(256) -> 10")
    print(f"Normalization : BatchNorm2d")
    print(f"Pooling       : MaxPool2d")
    print(f"Regularization: Dropout(0.5)")
    print(f"Loss          : CrossEntropyLoss")
    print(f"Optimizer     : Adam (lr=0.001)")
    print(f"Epochs        : {epochs}\n")

    model     = CNNClassifier()
    criterion = nn.CrossEntropyLoss()
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
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == y_batch).sum().item()
            total      += y_batch.size(0)

        avg_loss = epoch_loss / len(train_loader)
        avg_acc  = correct / total
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)

        print(f"Epoch [{epoch+1:2d}/{epochs}] Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

    return model, train_losses, train_accs


def evaluate_model(model, test_loader):
    print("\n--- 3. Evaluating CNN ---")
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs   = model(X_batch)
            predicted = outputs.argmax(dim=1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy : {accuracy:.4f}")
    return all_preds, all_labels, accuracy


def visualize_results(train_losses, train_accs, all_labels, all_preds, base_dir):
    print("\n--- 4. Visualizing Results ---")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "CNN Classifier — MNIST Dataset\n"
        "Sheshikala Mamidisetti | AlgoProfessor AI R&D Internship",
        fontsize=13, fontweight="bold"
    )

    axes[0].plot(train_losses, color="red", lw=2, marker="o")
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, color="blue", lw=2, marker="o")
    axes[1].set_title("Training Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2])
    axes[2].set_title("Confusion Matrix", fontweight="bold")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")

    plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=3.0)
    save_path = os.path.join(output_dir, "cnn_classifier_results.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Chart saved to: {save_path}")
    plt.show()
    plt.close()


def run_cnn_classifier():
    train_loader, test_loader, base_dir = load_data()
    model, train_losses, train_accs = train_model(train_loader, epochs=10)
    all_preds, all_labels, accuracy = evaluate_model(model, test_loader)
    visualize_results(train_losses, train_accs, all_labels, all_preds, base_dir)
    print("\nDay 04 CNN Classifier completed successfully.")


if __name__ == "__main__":
    run_cnn_classifier()
