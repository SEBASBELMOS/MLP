"""
Multi-Layer Perceptron (MLP) for Vehicle Silhouettes Classification
Dataset: Statlog (Vehicle Silhouettes) from UCI Machine Learning Repository
Author: AI Assistant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import urllib.request
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class VehicleClassificationMLP:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        
    def load_data(self):
        """
        Load the Statlog Vehicle Silhouettes dataset from UCI
        """
        print("Loading Statlog Vehicle Silhouettes dataset...")
        
        # URL for the dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaa.dat"
        url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xab.dat"
        url3 = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xac.dat"
        url4 = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xad.dat"
        url5 = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xae.dat"
        url6 = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xaf.dat"
        url7 = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xag.dat"
        url8 = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xah.dat"
        url9 = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xai.dat"
        
        urls = [url, url2, url3, url4, url5, url6, url7, url8, url9]
        
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
    
    def build_model(self, input_dim, num_classes):
        """
        Build the MLP model architecture
        """
        print(f"\nBuilding MLP model...")
        print(f"Input dimension: {input_dim}")
        print(f"Number of classes: {num_classes}")
        
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        model.summary()
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the MLP model
        """
        print(f"\nTraining model for {epochs} epochs...")
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model and calculate metrics
        """
        print("\nEvaluating model...")
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
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
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("="*60)
        print("VEHICLE SILHOUETTES CLASSIFICATION USING MLP")
        print("="*60)
        
        # Load data
        df = self.load_data()
        
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(df)
        
        # Build model
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        self.model = self.build_model(input_dim, num_classes)
        
        # Train model
        self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        y_pred, y_pred_proba, accuracy, cm = self.evaluate_model(X_test, y_test)
        
        # Plot results
        self.plot_training_history()
        self.plot_confusion_matrix(cm)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'history': self.history
        }

def main():
    """
    Main function to run the vehicle classification analysis
    """
    # Create and run the analysis
    mlp_classifier = VehicleClassificationMLP()
    results = mlp_classifier.run_complete_analysis()
    
    print(f"\nFinal Test Accuracy: {results['accuracy']:.4f}")
    print("Results saved as 'training_history.png' and 'confusion_matrix.png'")

if __name__ == "__main__":
    main()

