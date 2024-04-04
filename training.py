"""
File used for RandomCV tuning for miscellaneous models.
"""

# File Handling Imports.
import csv
import pickle
import os

#classifer imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Training Imports.
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import classification_report

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

features_selected = [1,3,6,7]  # 0,1,3,5,7
for i in [1,2,3,4]:
    train(f'Data2\\Fold{i}', features_selected)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, stratify = Y)
print('file part done!\tX_train = {}\tX_valid = {}'.format(len(X), len(X_test)))

### randomized search
model_list = [KNeighborsClassifier, RandomForestClassifier]
random_grid_li = {
    KNeighborsClassifier: {'n_neighbors': list(range(1, 10, 2)),
                           'weights': ['uniform', 'distance'],
                           'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                           'leaf_size': list(range(1, 25, 1)),
                           'p': [1, 2]},
    RandomForestClassifier: {'n_estimators': list(range(25, 100, 5)),
                              'max_depth': list(range(3, 10, 2)),
                              'min_samples_leaf': list(range(1, 5, 1)),
                              'min_samples_split': list(range(2, 6, 1)),
                              'max_features': ['sqrt', 'log2', None],
                              'verbose': [1]},
}
cross_val = StratifiedShuffleSplit(n_splits = 4, random_state = 42)
for curr in model_list[1:]:
    model = curr(n_jobs = -1)
    random_sea = RandomizedSearchCV(estimator = model, param_distributions = random_grid_li[curr], n_iter = 5, cv = cross_val, n_jobs = 6, verbose = 2)
    random_sea.fit(X_train, y_train)
    print('trained')
    print(random_sea.best_params_)
    model = random_sea.best_estimator_
    pickle.dump(model, open("rf1new.pkl", "wb"))
    print(classification_report(y_test, model.predict(X_test)))
