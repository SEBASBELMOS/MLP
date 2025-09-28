# Bike Sharing Demand Prediction using MLP Regression

This project implements a Multi-Layer Perceptron (MLP) neural network for predicting bike rental demand using the [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) from the UCI Machine Learning Repository. The implementation uses scikit-learn's MLPRegressor for better compatibility and stability.

## Dataset Information

The Bike Sharing dataset contains hourly and daily count of rental bikes between years 2011 and 2012 in Capital bikeshare system with corresponding weather and seasonal information.

### Key Features:
- **Total Instances**: 17,389 hourly records
- **Features**: 13 features including temporal, weather, and seasonal information
- **Target Variable**: `cnt` (total rental bikes count)
- **Time Period**: 2011-2012 Capital bikeshare system
- **Data Quality**: No missing values

### Features Used:
- **Temporal**: season, year, month, hour, weekday, workingday, holiday
- **Weather**: weathersit, temp, atemp, hum, windspeed
- **Target**: cnt (total rental bikes)

## Model Architecture

The MLP regression model uses scikit-learn's MLPRegressor with the following architecture:
- **Input Layer**: 12 features (after preprocessing)
- **Hidden Layers**: 128 → 64 → 32 → 16 neurons
- **Output Layer**: 1 neuron (regression)
- **Activation**: ReLU
- **Regularization**: L2 (α=0.001)
- **Optimizer**: Adam
- **Early Stopping**: Enabled

## Features

- **Data Loading**: Automatically downloads dataset from UCI repository
- **Data Preprocessing**: Feature scaling, train/validation/test split
- **MLP Architecture**: Deep neural network with dropout regularization
- **Training**: Uses early stopping and learning rate reduction callbacks
- **Evaluation**: Comprehensive regression metrics (MSE, RMSE, MAE, R², MAPE)
- **Visualization**: Training history, prediction plots, residual analysis

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook bike_sharing_regression.ipynb
```

## Expected Results

The model typically achieves:
- **R² Score**: > 0.8 (explains >80% of variance)
- **RMSE**: Low root mean squared error
- **MAE**: Low mean absolute error
- **MAPE**: < 20% mean absolute percentage error

## Key Insights

1. **Temporal Patterns**: Hour of day and season are strong predictors
2. **Weather Impact**: Temperature and weather conditions significantly affect demand
3. **Working Days**: Different patterns for working vs non-working days
4. **Model Performance**: Neural networks effectively capture complex interactions

## Files

- `bike_sharing_regression.ipynb`: Main Jupyter notebook with complete analysis
- `requirements.txt`: Required Python packages
- `README.md`: This documentation file

## Educational Value

This project demonstrates:
- Regression with neural networks using scikit-learn MLPRegressor
- Proper data preprocessing for time series data
- Overfitting detection and prevention techniques
- Comprehensive regression evaluation metrics
- Visualization techniques for regression analysis
- Real-world application of ML in transportation systems
