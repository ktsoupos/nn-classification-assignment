# KNN vs Nearest Centroid on CIFAR-10

Comparing K-Nearest Neighbors and Nearest Centroid classifiers on CIFAR-10 dataset.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

First run downloads CIFAR-10 (~170MB). Subsequent runs use cached data.

## Classifiers

1. **KNN (k=1)** - Predicts using single nearest neighbor
2. **KNN (k=3)** - Predicts using 3 nearest neighbors  
3. **Nearest Centroid** - Predicts using class centroids

## Results

| Classifier | Accuracy | Prediction Time |
|------------|----------|-----------------|
| KNN (k=1) | ~35% | ~25s |
| KNN (k=3) | ~34% | ~24s |
| Nearest Centroid | ~28% | ~1s |

Random baseline: 10%

## Dataset

CIFAR-10: 60,000 images (32×32 pixels) in 10 classes
- Training: 36,000 (60%)
- Testing: 24,000 (40%)

## Structure
```
├── main.py              # Run experiments
├── src/
│   ├── data_loader.py  # Load CIFAR-10
│   ├── classifiers.py  # Custom implementations
│   └── utils.py        # Evaluation functions
└── results/            # Output files
```

## Key Findings

- KNN is more accurate but 25x slower than Nearest Centroid
- All classifiers perform 3x better than random guessing
- Deep learning achieves 90%+ on this dataset
