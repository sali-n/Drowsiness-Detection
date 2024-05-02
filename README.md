# Drowsiness-Detection

## Description

This repository contains code and models for a drowsiness detection system. Below is a brief description of the files and folders:

### Files:

- `main_multi.py`: Main drowsiness code.
- `eye.py`: Helper class used for eye-based feature calculation, used in tandem with `main_multi.py`.
- `final.pkl`: XGBoost model for drowsiness detection.

### Folders:

- `deep_learning/`: Folder containing deep learning models and associated training scripts.
- `feature_selection/`: Folder containing various files for feature selection:
  - `feature_visualization.py`: Visualization of features graph with their output.
  - `feature_combination.py`: Combination of features tested for various models.
  - `feature_selection.py`: Various feature selection methods.

- `hyperparameter_tuning/`: Folder containing hyperparameter tuning scripts:
  - `random_search_cv.py`: Hyperparameter tuning using RandomSearchCV.
  - `hyperopt.py`: Hyperparameter tuning using Hyperopt.
  - `stepwise_xgboost.py`: Stepwise approach using XGBoost for hyperparameter tuning (GROUP 1: max_depth, min_child_weight, GROUP 2: subsample, colsample_bytree, GROUP 3: learning_rate, num_boost_round).

## Dataset:

The dataset used was the UTARLDD found here: https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset

## Running the Code

To run the code, execute `main_multi.py`. The first 200 frames (corresponds to ~ 5 - 10 seconds depending on the system) are used for calibrating. During this time, stare into the camera normally and blink a maximum of one time. 

## Results

The following table displays the accuracy of various models:

| Model               | Accuracy |
|---------------------|----------|
| XGBoost             | 90%      |
| Bagging Classifier  | 89%      |
| Decision Tree       | 87%      |
| LGM Classifier      | 87%      |
| Gradient Boost      | 84%      |
| Random Forest       | 81%      |
| K Neighbours        | 80%      |
| Extra Tree          | 78%      |
| AdaBoost            | 72%      |
| Naïve Bayes         | 41%      |
| SVM                 | 41%      |
| Logistic Regression | 33%      |
| LSTM                | 72%      |
| LSTM - Stacked      | 75%      |
| RNN                 | 69%      |
| RNN - Stacked       | 72%      |
