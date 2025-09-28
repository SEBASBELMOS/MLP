# Vehicle Silhouettes Classification using MLP

This project implements a Multi-Layer Perceptron (MLP) neural network for classifying vehicle silhouettes using the Statlog Vehicle Silhouettes dataset from the UCI Machine Learning Repository.

## Dataset

The Statlog Vehicle Silhouettes dataset contains 18 features extracted from vehicle silhouettes and 4 classes:
- BUS
- VAN
- SAAB
- OPEL

## Features

- **Data Loading**: Automatically downloads and combines the dataset from UCI
- **Data Preprocessing**: Normalization, label encoding, and train/validation/test split
- **MLP Architecture**: 4-layer neural network with dropout for regularization
- **Training**: Uses early stopping and learning rate reduction callbacks
- **Evaluation**: Comprehensive metrics including accuracy, classification report, and confusion matrix
- **Visualization**: Training history plots and confusion matrix heatmap

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python vehicle_classification_mlp.py
```

## Model Architecture

The MLP consists of:
- Input layer: 18 features
- Hidden layer 1: 128 neurons with ReLU activation + Dropout (0.3)
- Hidden layer 2: 64 neurons with ReLU activation + Dropout (0.3)
- Hidden layer 3: 32 neurons with ReLU activation + Dropout (0.2)
- Output layer: 4 neurons with Softmax activation

## Training Features

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: Early Stopping, Learning Rate Reduction
- **Validation Split**: 20% of training data

## Output

The script generates:
1. Training history plots (accuracy and loss curves)
2. Confusion matrix visualization
3. Classification report with precision, recall, and F1-score
4. Final test accuracy

## Files Generated

- `training_history.png`: Training and validation accuracy/loss curves
- `confusion_matrix.png`: Confusion matrix heatmap

