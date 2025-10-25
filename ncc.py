import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score


# Load CIFAR-10
print("Loading CIFAR-10...")
X, y = fetch_openml('CIFAR_10', version=1, return_X_y=True, parser='auto')

# Normalize pixel values to [0, 1]
X = np.array(X, dtype=float) / 255.0
y = np.array(y)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Split: 60% train, 40% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")


# ============================================
# 2. NEAREST CENTROID CLASSIFIER
# ============================================

print("\n" + "="*60)
print("NEAREST CENTROID CLASSIFIER")
print("="*60)

print("Training Nearest Centroid...")
start = time.time()
ncc = NearestCentroid()
ncc.fit(X_train, y_train)
ncc_train_time = time.time() - start
print(f"Training time: {ncc_train_time:.2f}s")

print("Predicting...")
start = time.time()
ncc_predictions = ncc.predict(X_test)
ncc_test_time = time.time() - start
print(f"Prediction time: {ncc_test_time:.2f}s ({ncc_test_time/60:.2f} min) for {X_test.shape[0]} samples")

ncc_accuracy = accuracy_score(y_test, ncc_predictions)
print(f"Nearest Centroid Accuracy: {ncc_accuracy * 100:.2f}%")