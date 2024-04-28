"""
For hyperopt based hyperparamter tuning.
"""

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from hyperopt.pyll.base import scope

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm

#classifer imports
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import csv
import os
import numpy as np

X = []
Y = []
X_test = []
y_test = []

def train(folder):
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
                    if i < 8 and (i != 2 and i != 3): #i < 7
                        a.append(float(x))
                    i += 1
                X.append(a)
                Y.append(ylabel)

def train_test(folder):
    """Get X and Y features from csv files."""
    global X_test, y_test
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
                    if i < 8 and (i != 2 and i != 3): #i < 7
                        a.append(float(x))
                    i += 1
                X_test.append(a)
                y_test.append(ylabel)

train(r'Data\Fold1')
train(r'Data\Fold2')
train(r'Data\Fold3')
train_test(r'Data\Fold4')
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
X_train = X
y_train = Y

space={'max_depth': scope.int(hp.quniform("max_depth", 3, 18, 1)),
        'gamma': hp.uniform ('gamma', 0,4),
        'reg_alpha' : scope.int(hp.quniform('reg_alpha', 0,10,1)),
        'reg_lambda' : hp.uniform('reg_lambda', 1,4),
        'colsample_bytree' : scope.int(hp.uniform('colsample_bytree', 0, 1)),
        'min_child_weight' : scope.int(hp.quniform('min_child_weight', 0, 10, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators',200,800,100)),
        'learning_rate': hp.uniform('learning_rate',0,1),
        'n_jobs' : -1
    }

space_rf={'n_estimators':scope.int(hp.quniform('n_estimators',25,150,1)),
           'max_depth':scope.int(hp.quniform('max_depth',5,20,1)),
           'min_samples_leaf':scope.int(hp.quniform('min_samples_leaf',1,5,1)),
           'min_samples_split':scope.int(hp.quniform('min_samples_split',2,6,1)),
           'max_features':hp.choice('max_features',['sqrt','log2',None]),
           'n_jobs' : 4
    }

dt_params = {'criterion': hp.choice('criterion', ['gini', 'entropy']),
             'splitter': hp.choice('splitter', ['best', 'random']),
             'max_depth': scope.int(hp.quniform('max_depth', 3, 50, 1)),
             'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 50, 1)),
             'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 50, 1)),
             'max_features': hp.choice('max_features', ['auto', 'log2', None]),
             'n_jobs' : 4
}

sv_params = {'C': hp.uniform('C', 0.1, 2.0),
             'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
             'degree': scope.int(hp.quniform('degree', 2, 5, 1)),
             'gamma': hp.choice('gamma', ['auto', 'scale']),
             'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-2)),
             'max_iter': scope.int(hp.quniform('max_iter', -1, 100, 1))
}

kn_params = {'n_neighbors': scope.int(hp.quniform('n_neighbors',1,100,2)),
             'weights': hp.choice('weights', ['uniform','distance']),
             'algorithm': hp.choice('algorithm',['ball_tree','kd_tree','brute']),
             'leaf_size': scope.int(hp.quniform('leaf_size',1,50,1)),
             'p': hp.choice('p',[1, 2]),
             'n_jobs' : 4
}

def bayes_tuning(estimator,space):
    #Define objective function
    def obj_function(params):
        model = estimator(**params)

        model.fit(X_train, y_train,)

        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        print ("SCORE:", accuracy)
        print(f"Paramaters = {params}")
        return {'loss': 1-accuracy, 'status': STATUS_OK }
    
    
    #Perform tuning
    hist = Trials()
    best_hyperparams = fmin(fn = obj_function,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 50,
                        trials = hist)
    
    return best_hyperparams


models = [XGBClassifier,DecisionTreeClassifier, SVC, RandomForestClassifier, KNeighborsClassifier ]
model_params = [space,dt_params, sv_params, space_rf, kn_params]

models = [XGBClassifier, KNeighborsClassifier]
model_params = [space, kn_params]

for m, par in zip(models, model_params):
    param = bayes_tuning(m,par)
    print(space_eval(param, par))

#extra algorithms::-------------------------------

# def objective_rf(space):
    # pass
    # clf=RandomForestClassifier(
    #     n_estimators=space['n_estimators'],max_depth=space['max_depth'], min_samples_leaf=space['min_samples_leaf'],
    #     min_samples_split=space['min_samples_split'], max_features=space['max_features']
    # )
    
    # evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    # clf.fit(X_train, y_train,
    #         eval_set=evaluation, eval_metric="auc",
    #         early_stopping_rounds=10,verbose=False)
    

    # pred = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, pred>0.5)
    # print ("SCORE:", accuracy)
    # return {'loss': -accuracy, 'status': STATUS_OK }

# trials = Trials()

# def objective_xgb(space):
    # clf=XGBClassifier(
    #                 n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
    #                 reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
    #                 colsample_bytree=int(space['colsample_bytree']))
    
    # evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    # clf.fit(X_train, y_train,
    #         eval_set=evaluation, eval_metric="auc",
    #         early_stopping_rounds=10,verbose=False)
    

    # pred = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, pred>0.5)
    # print ("SCORE:", accuracy)
    # return {'loss': -accuracy, 'status': STATUS_OK }
