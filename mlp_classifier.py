"""
Stock Price Movement Prediction using Multi-Layer Perceptron (MLP) Classifier

This module builds upon the PCA-transformed features created by pca_baseline.py to predict
significant stock price movements. The workflow is as follows:

1. Data Input:
   - Reads the PCA-transformed features from 'pca_features.parquet'
   - These features were created from raw stock data and reduced using PCA
   - The PCA transformation maintains 95% of the original variance

2. Target Creation:
   - Converts the multi-class labels to binary classification
   - Class 1 (positive): Large upward price movements (>1.5% increase)
   - Class 0 (negative): Small movements or decreases

3. Model Architecture:
   - Uses scikit-learn's MLPClassifier
   - Three hidden layers: 128 → 64 → 32 neurons
   - ReLU activation functions
   - Adam optimizer with adaptive learning rate
   - Early stopping to prevent overfitting

4. Training Approach:
   - Chronological train-test split (80-20)
   - No shuffling to maintain time series integrity
   - Validation fraction: 0.2 of training data
   - Early stopping with 10 epochs patience

5. Evaluation Metrics:
   - Accuracy: Overall prediction accuracy
   - Precision: Accuracy of positive predictions
   - Recall: Ability to find all positive cases
   - F1-score: Harmonic mean of precision and recall
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from config import SCRIPT_DIR, OUTPUT_DIR

# Set plot style
plt.style.use('default')
sns.set_style("whitegrid")

def load_data():
    """Load the enriched dataset with PCA and clustering features"""
    input_path = os.path.join(OUTPUT_DIR, "pca_features.parquet")
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            "\nERROR: PCA features file not found at: {}"
            "\n\nThis module requires the output from pca_baseline.py."
            "\nPlease ensure:"
            "\n1. The data/ directory exists"
            "\n2. pca_baseline.py has been run"
            "\n3. pca_features.parquet was generated successfully"
            .format(input_path)
        )
    
    try:
        df = pd.read_parquet(input_path)
        return df
    except Exception as e:
        raise Exception(
            "\nError reading PCA features file: {}"
            "\nPlease ensure pca_baseline.py completed successfully."
            .format(str(e))
        )

def create_model():
    """Create the MLPClassifier model"""
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        verbose=True
    )

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Calculate and return evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    # Detailed classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return metrics, report

def save_results(metrics, report, model, X_test, y_test, y_pred):
    """Save all results to a dedicated folder in the data directory"""
    # Create output directory
    output_folder = os.path.join(OUTPUT_DIR, "mlp_classifier_output")
    os.makedirs(output_folder, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_folder, "metrics.csv"), index=False)
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_folder, "classification_report.csv"))
    
    # Save predictions with probabilities
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted': y_pred,
        'probability': model.predict_proba(X_test)[:, 1]
    })
    predictions_df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False)
    
    # Save model parameters
    params = {
        'hidden_layer_sizes': model.hidden_layer_sizes,
        'activation': model.activation,
        'solver': model.solver,
        'alpha': model.alpha,
        'batch_size': model.batch_size,
        'learning_rate': model.learning_rate,
        'max_iter': model.max_iter,
        'early_stopping': model.early_stopping,
        'validation_fraction': model.validation_fraction,
        'n_iter_no_change': model.n_iter_no_change
    }
    pd.DataFrame([params]).to_csv(os.path.join(output_folder, "model_parameters.csv"), index=False)
    
    # Save learning curves
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, label='Training Loss')
        plt.title('Learning Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, "learning_curve.png"))
        plt.close()
    
    # Save validation scores
    if hasattr(model, 'validation_scores_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.validation_scores_, label='Validation Score')
        plt.title('Validation Scores During Training')
        plt.xlabel('Iterations')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, "validation_scores.png"))
        plt.close()
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
    plt.close()

def train_and_evaluate():
    """Main function to train and evaluate the MLP model"""
    print("Loading and preparing data...")
    df = load_data()
    
    # Display dataset info
    print("\nDataset Overview:")
    print("-" * 50)
    print(f"Shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Time period coverage: {df.shape[0] // 500:.1f} trading days")

    # Show label distribution
    dist = df['label'].value_counts(normalize=True)
    print("\nClass distribution:")
    print(f"  Big negative moves (<-1.5%): {dist.get(0, 0):.1%}")
    print(f"  Small/No moves (-1.5% to 1.5%): {dist.get(1, 0):.1%}")
    print(f"  Big positive moves (>1.5%): {dist.get(2, 0):.1%}")
    
    # Prepare features and target
    X = df.drop('label', axis=1)
    # Convert multi-class to binary (1 for up, 0 for down/neutral)
    y = (df['label'] == 2).astype(int)
    
    # Split data chronologically
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Keep chronological order
    )
    
    print("\nTraining set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    print("\nCreating and training MLP model...")
    model = create_model()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics, report = evaluate_model(y_test, y_pred)
    
    # Print metrics
    print("\nPerformance Metrics:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric:>10}: {value:.4f}")
    
    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(report)
    
    # Save all results to data directory
    save_results(metrics, report, model, X_test, y_test, y_pred)
    print("\nAll results have been saved to the data directory.")
    
    print("\nEvaluating model performance...")
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics, report = evaluate_model(y_test, y_pred)
    
    print("\n====== MLP Classifier Metrics ======")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    
    print("\nDetailed Classification Report:")
    print(report)
    
    return metrics, report, model

if __name__ == "__main__":
    metrics, report, history = train_and_evaluate()
