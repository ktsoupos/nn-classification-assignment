"""
Main script to compare KNN (k=1, k=3) and Nearest Centroid on CIFAR-10
"""
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from src.data_loader import load_cifar10
from src.utils import evaluate_classifier

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_cifar10(test_size=0.4)
    
    # Store results
    results = {}
    
    # 1. KNN with k=1
    knn1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    results['KNN (k=1)'] = evaluate_classifier(
        knn1, X_train, y_train, X_test, y_test,
        name="K-NEAREST NEIGHBORS (k=1)"
    )
    
    # 2. KNN with k=3
    knn3 = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    results['KNN (k=3)'] = evaluate_classifier(
        knn3, X_train, y_train, X_test, y_test,
        name="K-NEAREST NEIGHBORS (k=3)"
    )
    
    # 3. Nearest Centroid
    ncc = NearestCentroid()
    results['Nearest Centroid'] = evaluate_classifier(
        ncc, X_train, y_train, X_test, y_test,
        name="NEAREST CENTROID CLASSIFIER"
    )
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    print(f"\n{'Classifier':<25} {'Train Time (s)':<15} {'Test Time (s)':<15} {'Accuracy (%)':<15}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<25} {result['train_time']:<15.2f} "
              f"{result['test_time']:<15.2f} {result['accuracy']*100:<15.2f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nRandom baseline: 10.00%")
    for name, result in results.items():
        improvement = result['accuracy'] / 0.1
        print(f"{name}: {result['accuracy']*100:.2f}% ({improvement:.1f}x better than random)")
    
    # Find best accuracy
    best_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_acc = results[best_name]['accuracy']
    print(f"\nBest accuracy: {best_name} with {best_acc*100:.2f}%")
    
    # Speed comparison
    fastest_name = min(results.items(), key=lambda x: x[1]['test_time'])[0]
    fastest_time = results[fastest_name]['test_time']
    print(f"Fastest prediction: {fastest_name} with {fastest_time:.2f}s")

if __name__ == "__main__":
    main()