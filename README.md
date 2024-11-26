# Federated Learning with Private Label Distribution: A Three-Tiered Architecture

## Overview

This repository contains the implementation of our proposed three-tiered architecture designed to tackle challenges in **federated learning** scenarios where clients possess **private class labels**. Our framework ensures effective **knowledge sharing** while maintaining class **privacy**, leveraging a **hierarchical classification** approach:

1. **Known vs. Unknown Classification:** Determines if the input belongs to a client's known label space.
2. **Public vs. Private Classification:** Differentiates between public and private classes for known inputs.
3. **Specific Classification:** Performs detailed classification within the identified label space (public or private).

We evaluated our approach on **MNIST** and **CIFAR-10** datasets, demonstrating its effectiveness in handling private class scenarios while maintaining **model compression** during federated communication.

## Files in This Repository

1. **`MNIST_experiments.ipynb`**:
   - Implements the three-tiered architecture on the MNIST dataset.
   - Simulates federated learning scenarios with shared (0-4) and private (5-9) labels.

2. **`CIFAR_10_experiments.ipynb`**:
   - Extends the framework to the CIFAR-10 dataset.
   - Evaluates performance using MLP-based classification and out-of-distribution detection techniques.

3. **`Resnet_experiments.ipynb`**:
   - Explores the use of **ResNet-based architectures** for novelty assessment.
   - Includes experiments using the CIFAR-10 dataset for more advanced insights.

All files load their respective datasets automatically for seamless experimentation.

## Prerequisites

- Python 3.8 or above
- Libraries:
  - TensorFlow/PyTorch
  - NumPy
  - Matplotlib
  - Scikit-learn
- Datasets: MNIST, CIFAR-10 (automatically loaded in the notebooks)

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ary1002/Federated-Learning-Setup-WIth-Private-Labels.git
   cd your-repo-name
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open any of the experiment files in Jupyter Notebook or any compatible IDE:
   ```bash
   jupyter notebook MNIST_experiments.ipynb
   ```

