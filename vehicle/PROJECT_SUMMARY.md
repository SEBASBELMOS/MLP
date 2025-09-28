# Vehicle Silhouettes Classification using MLP - Project Summary

## Project Overview
This project implements a Multi-Layer Perceptron (MLP) neural network for classifying vehicle silhouettes using the **Statlog Vehicle Silhouettes dataset** from the UCI Machine Learning Repository.

## Dataset Information
- **Source**: UCI Machine Learning Repository
- **Dataset**: Statlog (Vehicle Silhouettes)
- **Total Samples**: 846
- **Features**: 18 numerical features extracted from vehicle silhouettes
- **Classes**: 4 vehicle types
  - Bus: 218 samples
  - Saab: 217 samples  
  - Opel: 212 samples
  - Van: 199 samples

## Implementation Details

### Technology Stack
- **Language**: Python 3.11
- **ML Framework**: Scikit-learn (MLPClassifier)
- **Data Processing**: Pandas, NumPy
- **Preprocessing**: StandardScaler, LabelEncoder
- **Evaluation**: Classification metrics from scikit-learn

### Model Architecture
- **Type**: Multi-Layer Perceptron (MLP)
- **Hidden Layers**: 3 layers (128, 64, 32 neurons)
- **Activation Function**: ReLU
- **Optimizer**: Adam
- **Regularization**: L2 (alpha=0.001)
- **Early Stopping**: Enabled (patience=10)

### Data Preprocessing
1. **Data Loading**: Downloaded from 9 separate files from UCI repository
2. **Data Cleaning**: Removed rows with missing values
3. **Label Encoding**: Converted string labels to numerical values
4. **Feature Scaling**: StandardScaler normalization
5. **Data Splitting**: 
   - Training: 540 samples (64%)
   - Validation: 136 samples (16%)
   - Test: 170 samples (20%)

## Results

### Model Performance
- **Training Accuracy**: 91.30%
- **Validation Accuracy**: 75.74%
- **Test Accuracy**: 78.82%
- **Training Iterations**: 36 (early stopping)

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support | Accuracy |
|-------|-----------|--------|----------|---------|----------|
| Bus   | 1.00      | 1.00   | 1.00     | 44      | 100.00%  |
| Opel  | 0.57      | 0.60   | 0.58     | 42      | 59.52%   |
| Saab  | 0.63      | 0.59   | 0.61     | 44      | 59.09%   |
| Van   | 0.95      | 0.97   | 0.96     | 40      | 97.50%   |

### Confusion Matrix
```
[[44  0  0  0]   # Bus: Perfect classification
 [ 0 25 15  2]   # Opel: Some confusion with Saab
 [ 0 18 26  0]   # Saab: Some confusion with Opel  
 [ 0  1  0 39]]  # Van: Excellent classification
```

## Key Findings

### Strengths
1. **Excellent Bus Classification**: 100% accuracy for bus identification
2. **Strong Van Classification**: 97.5% accuracy for van identification
3. **Good Overall Performance**: 78.82% test accuracy
4. **Efficient Training**: Early stopping at 36 iterations

### Challenges
1. **Opel vs Saab Confusion**: These two classes show significant confusion (15-18 misclassifications)
2. **Overfitting**: Training accuracy (91.30%) is much higher than validation accuracy (75.74%)
3. **Class Imbalance**: Slight imbalance in class distribution

### Model Behavior Analysis
- **Bus and Van**: Easily distinguishable features, leading to near-perfect classification
- **Opel and Saab**: Similar silhouette characteristics causing classification challenges
- **Generalization**: Model shows some overfitting but maintains reasonable test performance

## Technical Implementation

### Files Created
1. `final_vehicle_mlp.py` - Main implementation script
2. `vehicle_classification_sklearn.py` - Extended version with visualizations
3. `vehicle_classification_mlp.py` - TensorFlow version (requires TensorFlow installation)
4. `requirements.txt` - Dependencies
5. `README.md` - Usage instructions
6. `PROJECT_SUMMARY.md` - This summary document

### Key Features
- **Automatic Data Download**: Downloads dataset directly from UCI repository
- **Comprehensive Evaluation**: Multiple metrics and detailed analysis
- **Reproducible Results**: Fixed random seeds for consistent results
- **Error Handling**: Robust error handling and data validation
- **Detailed Logging**: Step-by-step progress reporting

## Conclusions

The MLP successfully classifies vehicle silhouettes with 78.82% accuracy. The model excels at distinguishing buses and vans but struggles with the similar Opel and Saab classes. The implementation demonstrates proper machine learning practices including data preprocessing, model validation, and comprehensive evaluation.

### Recommendations for Improvement
1. **Feature Engineering**: Extract additional features to better distinguish Opel and Saab
2. **Data Augmentation**: Increase training data for underrepresented classes
3. **Regularization**: Adjust L2 regularization to reduce overfitting
4. **Architecture Tuning**: Experiment with different layer sizes and activation functions
5. **Ensemble Methods**: Combine multiple models for better performance

## Usage Instructions

To run the project:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the main script
python final_vehicle_mlp.py
```

The script will automatically download the dataset, train the model, and display comprehensive results including accuracy metrics, classification report, and confusion matrix.

