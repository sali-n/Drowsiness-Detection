"""
File used for RandomCV tuning for miscellaneous models.
"""

# File Handling Imports.
import csv
import pickle
import os

#classifer imports
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier

# Training Imports.
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import classification_report
from scipy.stats import randint, uniform
import numpy as np

X = []
Y = []
X_test = []
y_test = []

def train(folder, features_selected):
    """Get X and Y features from csv files."""
    global X, Y
    for filename in os.listdir(folder):
        if filename.startswith('drowsy'):
            ylabel = 1
        else:
            ylabel = 0
        with open(os.path.join(folder,filename),'r') as f:
            re = csv.reader(f,delimiter=',')
            for row in re:
                a = []
                i = 0
                for x in row:
                    if i in features_selected:
                        a.append(float(x))
                    i += 1
                X.append(a)
                Y.append(ylabel)

features_selected = [1,3,5,7]  # 0,1,3,5,7  # 1,3,6,7
for i in [1,2,3,4]:
    train(f'Data2\\Fold{i}', features_selected)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, stratify = Y)
print('file part done!\tX_train = {}\tX_valid = {}'.format(len(X), len(X_test)))

### randomized search
model_list = [KNeighborsClassifier, RandomForestClassifier, BaggingClassifier, LGBMClassifier, LinearSVC, ExtraTreesClassifier, LogisticRegression]
random_grid_li = {
    KNeighborsClassifier:{'n_neighbors': list(range(1, 10, 2)),
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                        'p': [1, 2]},
    RandomForestClassifier: {'n_estimators': list(range(25, 100, 5)),
                              'max_depth': list(range(3, 10, 2)),
                              'min_samples_leaf': np.linspace(0,0.5,11).tolist(),
                              'min_samples_split': np.linspace(0,0.7,11).tolist(),
                              'max_features': ['sqrt', 'log2', None],
                              'verbose': [1]},                   
    BaggingClassifier: {'bootstrap_features': [False, True],
                        'max_features': [0.5, 0.7, 1.0],
                        'max_samples': [0.5, 0.7, 1.0],
                        'n_estimators': list(range(1, 50, 4))},
    LGBMClassifier: {'reg_alpha': [0.1, 0.5],
                    'reg_lambda': [0, 1, 10],
                    'n_estimators': list(range(50, 250, 50)),
                    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
                    'max_depth': [3, 5, 7, 9]},
    LinearSVC: {'C': [0.1, 1, 10, 100],
                'penalty': ['l1','l2'],
                'kernel': ['linear', 'poly', 'rbf'],
                'degree': [2, 3, 5, 7, 9],
                'dual': [False]},
    ExtraTreesClassifier: { 'n_estimators': randint(100, 1000),  
                            'max_depth': randint(1, 100),        
                            'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5],  
                            'min_samples_leaf': randint(1, 20),    
                            'max_features': ['sqrt', 'log2', None],  
                            'bootstrap': [True, False]},
    LogisticRegression: {'penalty': ['l1', 'l2'],           
                        'C': uniform(0.1, 10),             
                        'solver': ['liblinear', 'saga'],  
                        'max_iter': [100, 200, 300, 400]   
    }


}

# model = GaussianNB()
# model.fit(X_train, y_train)
# pickle.dump(model, open('naive.pkl','wb'))
# print(classification_report(y_test, model.predict(X_test)))

cross_val = StratifiedShuffleSplit(n_splits = 2, random_state = 42)
for curr in model_list:
    model = curr(n_jobs = -1)
    random_sea = RandomizedSearchCV(estimator = model, param_distributions = random_grid_li[curr], n_iter = 5, cv = cross_val, n_jobs = 6, verbose = 2)
    random_sea.fit(X_train, y_train)
    print('trained')
    print(random_sea.best_params_)
    model = random_sea.best_estimator_
    pickle.dump(model, open("logisiticregre.pkl", "wb"))
    print(classification_report(y_test, model.predict(X_test)))
