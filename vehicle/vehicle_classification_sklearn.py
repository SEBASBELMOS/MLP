"""
Multi-Layer Perceptron (MLP) for Vehicle Silhouettes Classification using Scikit-learn
Dataset: Statlog (Vehicle Silhouettes) from UCI Machine Learning Repository
Author: AI Assistant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import urllib.request
import os

# Set random seeds for reproducibility
np.random.seed(42)

class VehicleClassificationMLP:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """
        Load the Statlog Vehicle Silhouettes dataset from UCI
        """
        print("Loading Statlog Vehicle Silhouettes dataset...")
        
        # URL for the dataset
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
        
        # Download and combine all parts
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
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset: separate features and labels, encode labels, split data
        """
        print("\nPreprocessing data...")
        
        # Separate features and labels
        X = df.iloc[:, :-1].values  # All columns except the last one
        y = df.iloc[:, -1].values   # Last column (class labels)
        
        # Encode labels to numeric values
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Original classes: {self.label_encoder.classes_}")
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
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Validation set shape: {X_val_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def build_model(self):
        """
        Build the MLP model using scikit-learn
        """
        print(f"\nBuilding MLP model using scikit-learn...")
        
        # Create MLPClassifier with similar architecture to TensorFlow version
        self.model = MLPClassifier(
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
        print(f"Hidden layers: {self.model.hidden_layer_sizes}")
        print(f"Activation: {self.model.activation}")
        print(f"Solver: {self.model.solver}")
        print(f"Max iterations: {self.model.max_iter}")
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train the MLP model
        """
        print(f"\nTraining model...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        print("Training completed!")
        
        # Get training and validation scores
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Validation accuracy: {val_score:.4f}")
        
        return train_score, val_score
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model and calculate metrics
        """
        print("\nEvaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return y_pred, y_pred_proba, accuracy, cm
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Vehicle Classification')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.savefig('confusion_matrix_sklearn.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, X_train, feature_names=None):
        """
        Plot feature importance (using the first layer weights)
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        # Get the weights from the first layer
        first_layer_weights = self.model.coefs_[0]
        
        # Calculate feature importance as the sum of absolute weights
        feature_importance = np.sum(np.abs(first_layer_weights), axis=1)
        
        # Normalize
        feature_importance = feature_importance / np.sum(feature_importance)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        indices = np.argsort(feature_importance)[::-1]
        
        plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance (First Layer Weights)')
        plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("="*60)
        print("VEHICLE SILHOUETTES CLASSIFICATION USING MLP (Scikit-learn)")
        print("="*60)
        
        # Load data
        df = self.load_data()
        
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(df)
        
        # Build model
        self.build_model()
        
        # Train model
        train_score, val_score = self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        y_pred, y_pred_proba, accuracy, cm = self.evaluate_model(X_test, y_test)
        
        # Plot results
        self.plot_confusion_matrix(cm)
        self.plot_feature_importance(X_train)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'train_score': train_score,
            'val_score': val_score,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'model': self.model
        }

def main():
    """
    Main function to run the vehicle classification analysis
    """
    # Create and run the analysis
    mlp_classifier = VehicleClassificationMLP()
    results = mlp_classifier.run_complete_analysis()
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {results['train_score']:.4f}")
    print(f"Validation Accuracy: {results['val_score']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print("Results saved as 'confusion_matrix_sklearn.png' and 'feature_importance.png'")

if __name__ == "__main__":
    main()

