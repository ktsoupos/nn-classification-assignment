"""
Data loading and preprocessing functions
"""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_cifar10(test_size=0.4, random_state=42):
    """
    Load and preprocess CIFAR-10 dataset
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Loading CIFAR-10...")
    X, y = fetch_openml('CIFAR_10', version=1, return_X_y=True, parser='auto')
    
    # Normalize
    X = np.array(X, dtype=float) / 255.0
    y = np.array(y)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test