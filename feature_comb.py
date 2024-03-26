"""Iterate to find the best feature combinations."""
# File Handling Imports. 
import csv
import os
from random import sample

# Model Imports.
from sklearn.linear_model import SGDClassifier

# Training Imports.
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
# from itertools import combinations

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

features_selected = [0,1,2,3,4,5,6,7,8]
valid = sample(range(1,49), 4)
print(valid)
test = [49] # sample([num for num in range(1,47) if num not in valid], 1)
valid = [1, 5, 19, 7]
for i in [1,2,3,4]:
    train(f'Data2\\Fold{i}', features_selected, valid, test)
X_train = np.array(X)
y_train = np.array(Y)
X_test = np.array(X_valid)
print('file part done!\tX_train = {}\tX_valid = {}'.format(len(X), len(X_valid)))

# feature_combinations = [c for i in range(1, len(features_selected) + 1) for c in combinations(features_selected, i)]
feature_combinations = []
results = []
with open('features_lgm.csv','r') as f:
    re = csv.reader(f,delimiter=',')
    for row in re:
        a = []
        for i in row:
            a.append(int(i))
        feature_combinations.append(a)

# Iterate over each feature combination
for features in feature_combinations:
    classifier = SGDClassifier(n_jobs = 6)
    classifier.fit(X_train[:, features], y_train)  # only select appropiate features.

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, classifier.predict(X_test[:, features]))
    with open('results_sgd.csv', 'a', newline='') as fil:
        writer = csv.writer(fil)
        writer.writerow([accuracy,features])
    results.append((features, accuracy))
    print(f'accuracy: {accuracy:.2f}\tFeatures: {features}')

# Sort the results by accuracy in descending order
results.sort(key=lambda x: x[1], reverse=True)

# Display the top results
for features, accuracy in results[:15]:
    print(f"Features: {features}, Accuracy: {accuracy:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(range(len(results)), [acc for _, acc in results], tick_label=[str(features) for features, _ in results], color='blue')
plt.xlabel('Feature Combination')
plt.ylabel('Accuracy')
plt.title('Accuracy for Different Feature Combinations (SGD)')
plt.xticks(rotation=45, ha='right')
plt.show()
