# Day 04 — Deep Neural Networks (MNIST Dataset)

**Intern:** Sheshikala Mamidisetti
**Internship:** AlgoProfessor AI R&D Internship — Batch 2026
**Phase:** Phase 1 | Week 3 | D11–15
**Date:** March 8–9, 2026
**Milestone:** M2 — Enterprise Knowledge Navigator

---

## Objective

Built deep neural networks with statistical theory grounding on
the MNIST dataset — implementing DNN from scratch, CNN classifier,
production training pipeline and transfer learning using ResNet18.

---

## Dataset

| Property | Details |
|----------|---------|
| Name | MNIST Handwritten Digits |
| Samples | 70,000 (60,000 train + 10,000 test) |
| Classes | 10 (digits 0–9) |
| Image Size | 28 x 28 pixels |
| Source | Auto downloaded via PyTorch torchvision |

---


## 1. Neural Network from Scratch

**File:** neural_network_scratch.py

Built a Deep Neural Network completely from scratch using PyTorch
with statistical theory grounding and linear algebra operations.

| Property | Details |
|----------|---------|
| Architecture | 784 → 512 → 256 → 128 → 64 → 10 |
| Activation | ReLU + BatchNorm + Dropout |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |
| Epochs | 10 |
| Test Accuracy | Results in outputs/ |

Statistical Theory Applied:
- Mean, Std, Variance of pixel values
- Matrix multiplication (Linear Algebra)
- Weight initialization using He initialization

---

## 2. CNN Classifier

**File:** cnn_classifier.py

Built a Convolutional Neural Network for image classification
on MNIST using Conv layers, MaxPooling and BatchNorm.

| Property | Details |
|----------|---------|
| Architecture | Conv(32) → Conv(64) → Conv(128) → FC(256) → 10 |
| Pooling | MaxPool2d |
| Normalization | BatchNorm2d |
| Regularization | Dropout(0.5) |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |
| Epochs | 10 |
| Test Accuracy | Results in outputs/ |

---

## 3. Training Pipeline

**File:** training_pipeline.py

Production-grade training pipeline that:
- Runs all 3 models in sequence (runner)
- Tracks loss, accuracy and epoch time
- Saves results to txt and PNG files
- Uses ReduceLROnPlateau scheduler

---

## 4. Transfer Learning

**File:** transfer_learning.py

Fine-tuned ResNet18 on MNIST using transfer learning.

| Property | Details |
|----------|---------|
| Base Model | ResNet18 |
| Strategy | Freeze all layers, fine tune last block |
| Training Subset | 10,000 samples |
| Epochs | 5 |
| Test Accuracy | Results in outputs/ |

Key Finding: Transfer learning adapts pretrained ImageNet
weights to MNIST classification task.

---

## Key Findings

- CNN achieves higher accuracy than DNN on image data
- BatchNorm stabilizes and speeds up training
- Transfer learning works even with frozen pretrained weights
- PyTorch pipeline approach prevents data leakage

---

## Tools Used

Python | PyTorch | Torchvision | Scikit-learn | Matplotlib | Seaborn


