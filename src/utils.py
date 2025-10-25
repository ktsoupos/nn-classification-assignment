"""
Utility functions for evaluation and visualization
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def evaluate_classifier(clf, X_train, y_train, X_test, y_test, name="Classifier"):
    """
    Train and evaluate a classifier
    
    Returns:
        dict with train_time, test_time, accuracy, predictions
    """
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    # Train
    print("Training...")
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Training time: {train_time:.2f}s")
    
    # Predict
    print("Predicting...")
    start = time.time()
    predictions = clf.predict(X_test)
    test_time = time.time() - start
    print(f"Prediction time: {test_time:.2f}s ({test_time/60:.2f} min)")
    
    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return {
        'train_time': train_time,
        'test_time': test_time,
        'accuracy': accuracy,
        'predictions': predictions
    }


def plot_comparison(results, save_path='results/plots/comparison.png'):
    """
    Plot comparison between classifiers
    
    Args:
        results: dict with classifier names as keys
    """
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, alpha=0.7, edgecolor='black', linewidth=2)
    
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Classifier Comparison on CIFAR-10', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.axhline(y=10, color='red', linestyle='--', label='Random (10%)', linewidth=2)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{acc:.2f}%', ha='center', va='bottom', 
                 fontsize=12, fontweight='bold')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def save_results(results, filepath='results/metrics.txt'):
    """Save results to file"""
    with open(filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CIFAR-10 CLASSIFICATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for name, result in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Training Time: {result['train_time']:.2f}s\n")
            f.write(f"  Prediction Time: {result['test_time']:.2f}s\n")
            f.write(f"  Accuracy: {result['accuracy']*100:.2f}%\n")
            f.write(f"  vs Random: {result['accuracy']/0.1:.1f}x better\n\n")
    
    print(f"✓ Saved results to: {filepath}")