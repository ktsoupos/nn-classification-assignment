import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
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

# Train KNN with parallel processing
print("\nTraining KNN (k=1)...")
start = time.time()
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
knn.fit(X_train, y_train)
print(f"Training: {time.time() - start:.2f}s")

# Predict
print("Predicting...")
start = time.time()
predictions = knn.predict(X_test)
test_time = time.time() - start
print(f"Prediction: {test_time:.2f}s for {X_test.shape[0]} samples ")

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"\n Accuracy: {accuracy * 100:.2f}%")


print("\nTraining KNN (k=5)...")
start = time.time()
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(X_train, y_train)
print(f"Training: {time.time() - start:.2f}s")

# Predict
print("Predicting...")
start = time.time()
predictions = knn.predict(X_test)
test_time = time.time() - start
print(f"Prediction: {test_time:.2f}s for {X_test.shape[0]} samples ")

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"\n Accuracy: {accuracy * 100:.2f}%")
