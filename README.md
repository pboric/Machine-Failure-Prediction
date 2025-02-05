# Machine Failure Prediction System

A machine learning system that predicts equipment failures using sensor data through PCA and Random Forest classification.

## Overview
- Predicts failures with 87.3% accuracy
- Uses 9 sensor inputs transformed into 5 principal components
- Implements Random Forest with optimized hyperparameters
- ROC AUC score: 0.943

## Features
- Data preprocessing and PCA transformation
- Tuned Random Forest classifier
- Real-time monitoring capabilities
- Balanced precision/recall for failure detection

## Requirements
```python
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

## Installation
```bash
git clone https://github.com/[username]/machine-failure-prediction
cd machine-failure-prediction
pip install -r requirements.txt
```

## Usage
```python
from joblib import load
model = load('rf_model_tuned.joblib')

# Predict failure probability
def predict_failure(data):
    return model.predict_proba(data)[:, 1]
```

## Model Architecture
- Input: 9 sensor features
- PCA transformation: 5 components
- Random Forest parameters:
  - n_estimators: 200
  - max_depth: 5
  - min_samples_split: 10
  - class_weight: balanced

## Performance
- Accuracy: 87.3%
- Precision: 0.87
- Recall: 0.87
- F1-score: 0.87
- ROC AUC: 0.943

## Contributing
Pull requests welcome. For major changes, open an issue first.

## License
[MIT](https://choosealicense.com/licenses/mit/)
