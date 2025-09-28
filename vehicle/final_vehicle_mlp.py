"""
Final Working Vehicle Classification MLP using Scikit-learn
Dataset: Statlog (Vehicle Silhouettes) from UCI Machine Learning Repository
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import urllib.request
import sys

def load_vehicle_data():
    """Load the vehicle dataset from UCI"""
    print("="*60)
    print("LOADING STATLOG VEHICLE SILHOUETTES DATASET")
    print("="*60)
    
    # URLs for all parts of the dataset
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaa.dat",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xab.dat",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xac.dat",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xad.dat",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xae.dat",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaf.dat",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xag.dat",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xah.dat",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xai.dat"
    ]
    
    all_data = []
    for i, url in enumerate(urls):
        try:
            print(f"Downloading part {i+1}/9...")
            response = urllib.request.urlopen(url)
            data = response.read().decode('utf-8')
            lines = data.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 19:  # 18 features + 1 label
                        all_data.append(parts)
        except Exception as e:
            print(f"Error downloading part {i+1}: {e}")
            continue
    
    print(f"Total samples loaded: {len(all_data)}")
    
    # Convert to DataFrame
    columns = [f'feature_{i}' for i in range(18)] + ['class']
    df = pd.DataFrame(all_data, columns=columns)
    
    # Convert features to numeric
    for col in columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with NaN values
    df = df.dropna()
    
    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Class distribution:")
    print(df['class'].value_counts())
    
    return df

def preprocess_data(df):
    """Preprocess the dataset"""
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # Last column (class labels)
    
    # Encode labels to numeric values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Original classes: {le.classes_}")
    print(f"Encoded classes: {np.unique(y_encoded)}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Further split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Validation set shape: {X_val_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, le, scaler

def build_and_train_mlp(X_train, y_train, X_val, y_val):
    """Build and train the MLP model"""
    print("\n" + "="*60)
    print("BUILDING AND TRAINING MLP MODEL")
    print("="*60)
    
    # Create MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10
    )
    
    print("Model parameters:")
    print(f"Hidden layers: {mlp.hidden_layer_sizes}")
    print(f"Activation: {mlp.activation}")
    print(f"Solver: {mlp.solver}")
    print(f"Max iterations: {mlp.max_iter}")
    print(f"Early stopping: {mlp.early_stopping}")
    
    print("\nTraining model...")
    mlp.fit(X_train, y_train)
    
    # Get training and validation scores
    train_score = mlp.score(X_train, y_train)
    val_score = mlp.score(X_val, y_val)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Validation accuracy: {val_score:.4f}")
    print(f"Number of iterations: {mlp.n_iter_}")
    
    return mlp, train_score, val_score

def evaluate_model(mlp, X_test, y_test, le):
    """Evaluate the model"""
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    # Make predictions
    y_pred = mlp.predict(X_test)
    y_pred_proba = mlp.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(le.classes_):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
            print(f"{class_name}: {class_accuracy:.4f}")
    
    return y_pred, y_pred_proba, accuracy, cm

def main():
    """Main function"""
    print("VEHICLE SILHOUETTES CLASSIFICATION USING MLP")
    print("Dataset: Statlog (Vehicle Silhouettes) from UCI")
    print("Implementation: Scikit-learn MLPClassifier")
    
    try:
        # Load data
        df = load_vehicle_data()
        
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = preprocess_data(df)
        
        # Build and train model
        mlp, train_score, val_score = build_and_train_mlp(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        y_pred, y_pred_proba, test_accuracy, cm = evaluate_model(mlp, X_test, y_test, le)
        
        # Final results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Validation Accuracy: {val_score:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Number of training iterations: {mlp.n_iter_}")
        
        # Model summary
        print(f"\nModel Summary:")
        print(f"Total parameters: {mlp.coefs_[0].size + mlp.coefs_[1].size + mlp.coefs_[2].size + mlp.coefs_[3].size}")
        print(f"Input features: {X_train.shape[1]}")
        print(f"Output classes: {len(le.classes_)}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

