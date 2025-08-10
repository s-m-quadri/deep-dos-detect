# `DeepDoSDetect`: DoS Attack Detection using XGBoost on UNSW-NB15 Dataset

## Overview

This project implements a Denial of Service (DoS) attack detection system using the UNSW-NB15 dataset.
The pipeline includes data preprocessing, feature encoding, scaling, handling class imbalance with SMOTE, model training with XGBoost, threshold tuning for optimal F1-score, and model evaluation with multiple metrics and visualizations.

The goal is to classify each network record as either **DoS** or **Non-DoS**.

## Dataset

* Source: [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
* Files:

  * `UNSW_NB15_training-set.csv`
  * `UNSW_NB15_testing-set.csv`
* Features: Combination of categorical and numeric network traffic attributes.
* Labels: Multi-class original labels, but converted into **binary label**:

  * `1` → DoS attack
  * `0` → Non-DoS

## Steps Performed

### Data Loading and Preprocessing

* Combined training and testing datasets for consistent preprocessing.
* Encoded categorical features using `LabelEncoder`.
* Scaled numeric features with `StandardScaler` for better convergence.
* Created a binary target column for DoS detection.

### Class Imbalance Handling

* Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance classes in the training data.
* Balanced data helps avoid bias toward the majority class.

### Model Training

* Used **XGBoost Classifier** for its efficiency and handling of imbalanced datasets.
* Tuned hyperparameters for improved generalization and reduced overfitting.
* Adjusted decision threshold to maximize F1-score.

### Evaluation Metrics

* **Precision**: Correct DoS detections out of all predicted DoS.
* **Recall**: Correctly detected DoS out of all actual DoS.
* **F1-score**: Harmonic mean of precision and recall.
* **ROC-AUC**: Probability that the model ranks a random positive higher than a random negative.

### Visualizations

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Feature Importance Plot
* Class balance before and after SMOTE
* Distribution of predicted probabilities per class
* Precision and Recall vs Threshold plot
* Feature correlation heatmap

## Requirements

* Python 3.8+
* Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib imbalanced-learn
```

## Running the Script

1. Place the dataset CSV files in the working directory.
2. Update `train_path` and `test_path` in the script if needed.
3. Run the script in Python or Jupyter Notebook.
4. The model and best threshold will be saved as `dos_detector_model.pkl`.

## Example Outputs

### Classification Report

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Non-DoS | 0.98      | 0.97   | 0.98     | 10000   |
| DoS     | 0.96      | 0.97   | 0.97     | 8000    |

*(Values are placeholders. Replace with actual results after running.)*

### Confusion Matrix

Placeholder for heatmap output:
`[Image: Confusion Matrix plot]`

### ROC Curve

Placeholder for line plot output:
`[Image: ROC Curve with AUC score]`

### Precision-Recall Curve

Placeholder for PR plot output:
`[Image: Precision-Recall curve]`

### Feature Importance

Placeholder for bar plot output:
`[Image: Top 15 features with highest importance]`

### Additional Visualizations

* Class distribution before and after SMOTE
  `[Image: Bar chart of class counts before/after balancing]`
* Distribution of predicted probabilities for DoS vs Non-DoS
  `[Image: Histogram of prediction probabilities]`
* Precision & Recall vs Threshold
  `[Image: Line plot showing precision and recall trends]`
* Feature correlation heatmap
  `[Image: Heatmap of numeric feature correlations]`

## Model Saving and Loading

* The trained model and chosen threshold are saved in `dos_detector_model.pkl`.
* They can be reloaded without retraining for inference.
