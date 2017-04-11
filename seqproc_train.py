'''
Module: 03_analyse.ipynb
Version: 0.2
Python Version: 2.7.13
Authors: Hakim Khalafi, <>
Description:

    This module takes the windowed master data file created by 02_split.ipynb and performs classification on it.
    Some information is printed such as confusion matrices, F2 scores, cross validations, data size.
    Finally the model is saved for re-use by sequence processor API.

'''


## Imports

import os
from pprint import pprint
import csv
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import fbeta_score
import pickle
import time


## Configurations

test_split = 0.2
n_estimators = 500
N_window = 5
cv = 5

script_folder = os.path.realpath('.')
datafile = '/master_data/N_master_windowed/md_reshaped_n' + str(N_window) + '.csv'
total_path = script_folder + datafile
model_folder = '/models/'

# Custom methods for scoring

def f2scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    score = fbeta_score(y, y_pred, beta=2)
    return score

def test_model(chosen_model, X_train, y_train, X_test, y_test, cv):
    clf = chosen_model.fit(X_train, y_train)

    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print("Crossval F2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Accuracy scoring on test set
    y_pred = clf.predict(X_test)
    score = fbeta_score(y_test, y_pred, beta=2)
    model_name = str(type(chosen_model).__name__)
    print(model_name + ' F2 score: ' + str(score))
    return clf, score


## Read file

df = pd.read_csv(total_path, header=None)
print("Data points, features: " + str(df.shape))

df = df.reset_index(drop=True)

## Split set

cols = df.shape[1]
X = df.ix[:,:cols-2]
Y = df[cols-1].astype(int)
print("dig: "+  str(Y.value_counts()[1]) + ", nodig: " + str(Y.value_counts()[0]))
# Split to train and test set

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_split)


# Model testing

#test_model(GaussianNB(), X_train, y_train, X_test, y_test, cv)
#test_model(RandomForestClassifier(n_estimators=n_estimators), X_train, y_train, X_test, y_test, cv)
clf,score = test_model(AdaBoostClassifier(n_estimators=n_estimators), X_train, y_train, X_test, y_test, cv)


# Confusion matrix plot, function from scikit learn examples

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

y_pred = clf.predict(X_test)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['No dig','Dig'],
                      title='Confusion matrix')

plt.show()


# Save model to disk
timestr = time.strftime("%Y%m%d-%H%M%S")

filename = str(type(clf).__name__) + '_N' + str(N_window) + '_t' + timestr + '_F2_' + str(score*1000)[0:3] + '.pkl'
print("Saved model to: /models/" + filename)
pickle.dump(clf,open(script_folder + model_folder + filename, "wb" ) , protocol=2)


# ToDo: Probably want to make the output that looks like this:
# Crossval F2: 0.96 (+/- 0.00)
# AdaBoostClassifier F2 score: 0.99153705398
# Automatically save to a .txt with the same model name as saved above
# For now, saves F2 score in filename..    