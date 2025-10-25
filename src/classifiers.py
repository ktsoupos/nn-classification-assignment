"""
Custom classifier implementations
"""
import numpy as np
from collections import Counter

class CustomKNN:
    """
    Custom K-Nearest Neighbors implementation
    """
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X_test):
        """Predict for all test samples"""
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        """Predict for single sample"""
        # Vectorized distance calculation
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


class CustomNearestCentroid:
    """
    Custom Nearest Centroid Classifier implementation
    """
    def __init__(self):
        self.centroids = {}
        self.classes = None
    
    def fit(self, X, y):
        """Calculate centroid for each class"""
        self.classes = np.unique(y)
        
        for class_label in self.classes:
            class_samples = X[y == class_label]
            centroid = np.mean(class_samples, axis=0)
            self.centroids[class_label] = centroid
    
    def predict(self, X_test):
        """Predict by finding nearest centroid"""
        predictions = []
        
        for x in X_test:
            distances = {}
            for class_label, centroid in self.centroids.items():
                dist = np.sqrt(np.sum((x - centroid) ** 2))
                distances[class_label] = dist
            
            nearest_class = min(distances, key=distances.get)
            predictions.append(nearest_class)
        
        return np.array(predictions)