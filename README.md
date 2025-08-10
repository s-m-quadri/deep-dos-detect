<div align="center">
  <h1><b><code>DeepDoSDetect</code> DoS Attack Detection using XGBoost on the UNSW-NB15 dataset with SMOTE balancing, threshold tuning, and detailed evaluation visualizations.</b></h1>
  <p>This project implements a Denial of Service (DoS) attack detection system using the UNSW-NB15 dataset.
The pipeline includes data preprocessing, feature encoding, scaling, handling class imbalance with SMOTE, model training with XGBoost, threshold tuning for optimal F1-score, and model evaluation with multiple metrics and visualizations. The goal is to classify each network record as either DoS or Non-DoS.</p>

  <p>
    <a href="https://s-m-quadri.me/projects/deep-dos-detect">Homepage</a> ·
    <a href="https://github.com/s-m-quadri/deep-dos-detect">Repository</a> ·
    <a href="https://github.com/s-m-quadri/deep-dos-detect/discussions/new?category=q-a">Ask Question</a> ·
    <a href="mailto:dev.smq@gmail.com">Contact</a>
  </p>

  <a href="https://github.com/s-m-quadri/deep-dos-detect/releases">
         <img src="https://custom-icon-badges.demolab.com/github/v/tag/s-m-quadri/deep-dos-detect?label=Version&labelColor=302d41&color=f2cdcd&logoColor=d9e0ee&logo=tag&style=for-the-badge" alt="Release Version"/>
  </a>
  <a href="https://www.codefactor.io/repository/github/s-m-quadri/deep-dos-detect"><img src="https://img.shields.io/codefactor/grade/github/s-m-quadri/deep-dos-detect?label=CodeFactor&labelColor=302d41&color=8bd5ca&logoColor=d9e0ee&logo=codefactor&style=for-the-badge" alt="GitHub Readme Profile Code Quality"/></a>
  <a href="https://github.com/s-m-quadri/deep-dos-detect/issues">
    <img src="https://custom-icon-badges.demolab.com/github/issues/s-m-quadri/deep-dos-detect?label=Issues&labelColor=302d41&color=f5a97f&logoColor=d9e0ee&logo=issue&style=for-the-badge" alt="Issues"/>
  </a>
  <a href="https://github.com/s-m-quadri/deep-dos-detect/pulls">
    <img src="https://custom-icon-badges.demolab.com/github/issues-pr/s-m-quadri/deep-dos-detect?label=PRs&labelColor=302d41&color=ddb6f2&logoColor=d9e0ee&logo=git-pull-request&style=for-the-badge" alt="Pull Requests"/>
  </a>
  <a href="https://github.com/s-m-quadri/deep-dos-detect/graphs/contributors">
    <img src="https://custom-icon-badges.demolab.com/github/contributors/s-m-quadri/deep-dos-detect?label=Contributors&labelColor=302d41&color=c9cbff&logoColor=d9e0ee&logo=people&style=for-the-badge" alt="Contributors"/>
  </a>
</div>

<center><img width="894" height="800" alt="image" src="https://github.com/user-attachments/assets/515ac016-9b70-4a92-a5a6-3034401c577b" /></center>

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

<img width="527" height="414" alt="image" src="https://github.com/user-attachments/assets/6ba82322-6d12-4617-ae24-8af8df32eb90" />

## Model Saving and Loading

* The trained model and chosen threshold are saved in `dos_detector_model.pkl`.
* They can be reloaded without retraining for inference.
