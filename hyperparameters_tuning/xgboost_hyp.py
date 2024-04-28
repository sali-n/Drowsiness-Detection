"""
File used for hyperparamter tuning for xgboost.
"""

# File Handling Imports.
import csv
import pickle
import os
from random import sample
from matplotlib import pyplot

# Model Imports.
from xgboost import XGBClassifier

# Training Imports.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, brier_score_loss, log_loss
from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error

X = []
Y = []
X_test = []
y_test = []
X_valid = []
y_valid = []

def train(folder, features_selected, valid, test):
    """Get X and Y features from csv files."""
    global X, Y
    for filename in os.listdir(folder):
        if int(filename[-6:-4]) in valid:
            train_valid(folder, filename, features_selected)
            continue
        elif int(filename[-6:-4]) in test:
            train_test(folder, filename, features_selected)
            continue
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

def train_test(folder, filename, features_selected):
    """Get X and Y features from csv files."""
    global X_test, y_test
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
            X_test.append(a)
            y_test.append(ylabel)

def train_valid(folder, filename, features_selected):
    """Get X and Y features from csv files."""
    global X_valid, y_valid
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
            X_valid.append(a)
            y_valid.append(ylabel)

features_selected = [1,3,6,7]  # 0,1,3,5,7
valid = sample(range(1,49), 14)
print(valid)
test = sample([num for num in range(1,47) if num not in valid], 1) # 49
valid = [49]
test = [49]
for i in [1,2,3,4]:
    train(f'Data2\\Fold{i}', features_selected, valid, test)
print('file part done!\tX_train = {}\tX_valid = {}'.format(len(X), len(X_valid)))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
X, Y = shuffle(X,Y)
model = XGBClassifier(n_jobs = -1, max_depth = 4, n_estimators = 17, subsample = 0.5, colsample_bytree = 0.7)
# model = XGBClassifier(n_jobs=-1, max_depth = 9, min_child_weight = 100, subsample = 0.5, colsample_bytree = 0.7, learning_rate = 0.001, reg_lambda = 1, reg_alpha = 100, n_estimators = 5)# early_stopping_rounds = 30, reg_lambda = 1000, gamma = 4, colsample_bytree = 0.8, min_child_weight = 200) #early_stopping_rounds=30, max_depth = 9, n_estimators = 250) 
model.fit(X_train,y_train, eval_set=[(X_train, y_train)], eval_metric=["error", "logloss"], verbose = True)
print('trained')
x = pickle.load(open('xgb_80.pkl', "rb"))
if accuracy_score(y_valid, model.predict(X_valid)) > accuracy_score(y_valid, x.predict(X_valid)):
    pickle.dump(model, open("xgb_80.pkl", "wb"))
print('Training Accuracy = ', accuracy_score(Y, model.predict(X)))
print(classification_report(y_valid, model.predict(X_valid)))


# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()

### randomized search
# model = XGBClassifier(n_jobs = -1)
# random_grid = {'max_depth': [5,7,9,11,13],
#                'early_stopping_rounds': [int(x) for x in np.linspace(0, 50, num = 50)],
#                'reg_lambda': [0, 10, 100, 1000, 10000],
#                'gamma': [2,4,6,8,10,12,14],
#                'colsample_bytree': [float(x) for x in np.linspace(0, 1, num = 100)]}
# model2 = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 50, cv = 5, n_jobs = -1)
# model2.fit(X,Y)
# print('trained')
# print(model2.best_params_)
# model = model2.best_estimator_
# x = pickle.load(open('xgb_80.pkl', "rb"))
# if accuracy_score(y_valid, model.predict(X_valid)) > accuracy_score(y_valid, x.predict(X_valid)):
#     pickle.dump(model, open("xgb_89.pkl", "wb"))
# print('Training Accuracy = ', accuracy_score(Y, model.predict(X)))
# print(classification_report(y_valid, model.predict(X_valid)))
# pickle.dump(model, open("fold3xgb.pkl", "wb"))
