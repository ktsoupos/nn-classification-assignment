import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


# Load CIFAR-10
print("Loading CIFAR-10...")
X, y = fetch_openml('CIFAR_10', version=1, return_X_y=True, parser='auto')

# Convert to numpy arrays and normalize
X = np.array(X, dtype=float) / 255.0  # Normalize to 0-1
y = np.array(y)

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Split into train/test (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.4,      # 40% for testing
    random_state=42     # For reproducibility
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, new_points):
        predictions = [self._predict(new_point) for new_point in new_points]
        return np.array(predictions)

    def _predict(self, new_point):
        # Compute distances between new_point and all examples in the training set
        distances = [euclidean_distance(new_point, point) for point in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]



knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"\nKNN Classifier Accuracy: {accuracy * 100:.2f}%")


