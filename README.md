# Drowsiness-Detection

Drowsiness detection algorithm that uses xgboost.

Each files purpose is as follows:
- **main_multi.py**: Main code that is used for the drowsiness detection and extracting features from dataset. It uses multiprocessing to speed up the process for extracting features.
- **Eyes.py**: A helper class used for eye based feature calculation used in tandem with main_multi.
- **feature_comb.py**: Get the feature combinations for a wide variety of models.
- **feature_selection**: A range of feature selection methods are used to find which features have the most impact.
- **hyperopt_hyp.py**: Hyperparameter tuning using hyperopt.
- **training.py**: Hyperparameter tuning using RandomCV.
- **xgboost_hyp.py**: Hyperparamter tuning using a stepwise approach (GROUP 1: max_depth , min_child_weight, GROUP 2: subsample, colsample_bytree, GROUP 3: learning_rate, num_boost_round).

The dataset used was the UTARLDD dataset found here: https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset

Overall accuracy was 87%.
