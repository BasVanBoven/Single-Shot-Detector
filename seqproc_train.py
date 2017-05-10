#!/usr/bin/python
# seqproc_train.py - processes input data for training a Sequence Processor


# imports
import json
import os
import pickle
import time
import itertools
import csv
import argparse
import shutil
import random
import warnings
import subprocess
import signal
import numpy as np
import matplotlib.pyplot as plt
import seqproc_common as sp
import seqproc_featureno as spf
from random import shuffle
from pprint import pprint
from scipy.stats import mode
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,fbeta_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# handle input arguments
parser = argparse.ArgumentParser(description='Train a Sequence Processor.')
parser.add_argument('-d', '--debug', default=False, action='store_true', help='print debug output')
parser.add_argument('-b', '--balance', default=True, action='store_true', help='balance training set')
parser.add_argument('-r', '--random', default=False, action='store_true', help='perform random classification')
parser.add_argument('-a', '--augment', default=False, action='store_true', help='use augmented data for training')
parser.add_argument('-e', '--estimator', type=int, default=3, help='estimator that is to be used')
parser.add_argument('-f', '--beta', type=int, default=1, help='beta score to be reported')
parser.add_argument('-w', '--window', type=int, default=5, help='window size to be used, needs to be an odd number')
parser.add_argument('-c', '--crossval', type=int, default=10, help='number of cross validation splits')
parser.add_argument('-n', '--noserv', default=False, action='store_true', help='do not start serving after training')
args = parser.parse_args()
# window size needs to be uneven to make the majority vote function correctly
assert(args.window % 2 != 0)


# global variables and general pathing
warnings.filterwarnings('ignore')
estimators = 500
rootdir = os.getcwd()
model_folder = os.path.join(rootdir, 'seqproc', '04_model')
model_file = os.path.join(model_folder, 'model.pkl')
traintest_folder = os.path.join(rootdir, 'seqproc', '03_traintest')
testset = np.genfromtxt(os.path.join(traintest_folder, 'test.csv'), delimiter=',')
if args.augment:
    trainset = np.genfromtxt(os.path.join(traintest_folder, 'train_augmented.csv'), delimiter=',')
else:
    trainset = np.genfromtxt(os.path.join(traintest_folder, 'train.csv'), delimiter=',')


# initialize output directories
if (os.path.exists(model_folder)):
    shutil.rmtree(model_folder)
os.makedirs(model_folder, 0755)


# trains and tests a generated model
def train_test(chosen_model, X_train, y_train, X_test, y_test):
    classifier = chosen_model.fit(X_train, y_train)
    scores = cross_val_score(classifier, X_train, y_train, cv=args.crossval)
    print('Crossval F'+str(args.beta)+': %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    # accuracy scoring on test set
    y_pred = classifier.predict(X_test)
    score = fbeta_score(y_test, y_pred, beta=args.beta)
    model_name = str(type(chosen_model).__name__)
    print(model_name + ' F'+str(args.beta)+' score: ' + str(score))
    # print feature importances if applicable
    if (args.debug and args.solver in [3,4,5]):
        importances = chosen_model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in chosen_model.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")
        for f in range(X_train.shape[1]):
            print("%d. feature %d, %s (%f)" % (f + 1, indices[f], spf.featureno(indices[f], args.window), importances[indices[f]]))
    # return result
    return classifier, score


# prints a conusion matrix
def print_confusion_matrix(cm, classes, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix')
    print(cm)


# balance training set
if args.balance:
    np.random.shuffle(trainset)
    cutoff = min([np.count_nonzero(trainset[:,0]),trainset.shape[0] - np.count_nonzero(trainset[:,0])])
    count_dig = 0
    count_nodig = 0
    # walk over trainset backwards to not mess up indexing
    for i in range(trainset.shape[0]-1, -1, -1):
        if int(trainset[i,0]) == 1:
            count_dig += 1
            if count_dig >= cutoff:
                trainset = np.delete(trainset, i, 0)
        else:
            count_nodig +=1
            if count_nodig >= cutoff:
                trainset = np.delete(trainset, i, 0)
# split data from classification
X_train = trainset[:,1:]
X_test = testset[:,1:]
y_train = trainset[:,0].astype(int)
y_test = testset[:,0].astype(int)
# perform random classification, to validate training effect
if args.random:
    np.random.shuffle(y_train)
    np.random.shuffle(y_test)
# print debug output
if args.debug:
    print ('Train - shape: ' + str(X_train.shape) + ' ' + str(y_train.shape))
    print ('Test - shape: ' + str(X_test.shape) + ' ' + str(y_test.shape))
    print('Train - data points, features: ' + str(trainset.shape))
    print('Test  - data points, features: ' + str(testset.shape))
    print('Train - digging windows: '+  str(np.count_nonzero(y_train)) + ', nodig windows: ' + str(y_train.shape[0] - np.count_nonzero(y_train)))
    print('Test  - digging windows: '+  str(np.count_nonzero(y_test)) + ', nodig windows: ' + str(y_test.shape[0] - np.count_nonzero(y_test)))
# save model log
model_log = np.array([ ['Window Size','Augmentation','Cross Validation Splits','Balanced Dataset','Random Classification'] , [args.window,args.augment,args.crossval,args.balance,args.random] ])
with open(os.path.join(model_folder, 'model.log'), 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(model_log)
# train and test a model
print 'Model training started...'
# select the right classifier
assert(args.estimator < 6)
if args.estimator == 0:
    classifier, score = train_test(GaussianNB(), X_train, y_train, X_test, y_test)
elif args.estimator == 1:
    classifier, score = train_test(MLPClassifier(max_iter=2000), X_train, y_train, X_test, y_test)
elif args.estimator == 2:
    classifier, score = train_test(SVC(), X_train, y_train, X_test, y_test)
elif args.estimator == 3:
    classifier, score = train_test(AdaBoostClassifier(n_estimators=estimators), X_train, y_train, X_test, y_test)
elif args.estimator == 4:
    classifier, score = train_test(ExtraTreesClassifier(n_estimators=estimators), X_train, y_train, X_test, y_test)
elif args.estimator == 5:
    classifier, score = train_test(RandomForestClassifier(n_estimators=estimators), X_train, y_train, X_test, y_test)
# predict on test set
y_pred = classifier.predict(X_test)
# print confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print_confusion_matrix(cnf_matrix, classes=['No dig', 'Dig'])
# save model to disk
pickle.dump(classifier, open(model_file, 'wb'), protocol=2)

# append log to output file
with open('3dplot.csv', 'a') as output:
    output.write(str(args.window)+','+str(args.estimator)+','+str(score)+','+str(cnf_matrix[0,0])+','+str(cnf_matrix[0,1])+','+str(cnf_matrix[1,0])+','+str(cnf_matrix[1,1])+'\n')


# start serving, old server should have stopped automatically due to changed files
if args.noserv == False:
    # kill existing server, if running (does not work yet)
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    for line in out.splitlines():
        if 'seqproc_serv' in line:
            pid = int(line.split(None, 1)[0])
            os.kill(pid, signal.SIGKILL)
    # start new server
    cmd = 'nohup python '+rootdir+'/seqproc_serv.py &'
    print 'Started serving the trained model in the background...'
    os.system(cmd)