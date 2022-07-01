#!/usr/bin/env python
# coding: utf-8

pip install -U imbalanced-learn

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import cross_validate
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.svm import LinearSVC
from imblearn.under_sampling import NearMiss
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from collections import Counter
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

dataset = pd.read_csv("creditcard.csv")

X = dataset.iloc[:,0:30]
y = dataset.iloc[:,30]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dataset

print(Counter(y))

#Random Forest

RF_pipe1 = make_pipeline(RandomForestClassifier(random_state=0, n_jobs=-1))
RF_pipe2 = make_pipeline(RandomForestClassifier(random_state=0, class_weight = 'balanced', n_jobs=-1))
RF_pipe3= make_pipeline(NearMiss(version = 1),
                        RandomForestClassifier(random_state=0, class_weight = 'balanced', n_jobs=-1))
RF_pipe4= make_pipeline(NearMiss(version = 2),
                        RandomForestClassifier(random_state=0, class_weight = 'balanced', n_jobs=-1))
RF_pipe5= make_pipeline(NearMiss(version = 3),
                        RandomForestClassifier(random_state=0, class_weight = 'balanced', n_jobs=-1))
RF_pipe6= make_pipeline(SMOTE(random_state=0, k_neighbors=6, n_jobs=-1),
                        RandomForestClassifier(random_state=0, class_weight = 'balanced', n_jobs=-1))
RF_pipe7 = make_pipeline(EasyEnsembleClassifier(base_estimator=RandomForestClassifier(random_state=0, n_jobs=-1),n_jobs=-1, random_state=42))


RF_pipe1.fit(X_train,y_train)
RF_pipe2.fit(X_train,y_train)
RF_pipe3.fit(X_train,y_train)
RF_pipe4.fit(X_train,y_train)
RF_pipe5.fit(X_train,y_train)
RF_pipe6.fit(X_train,y_train)
RF_pipe7.fit(X_train, y_train)

scoring = ['accuracy', 'balanced_accuracy']

print("Random Forest")
scores = cross_validate(RF_pipe1, X, y, scoring = scoring, cv = 10, return_train_score = False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print("\nRandom Forest with balanced weights")
scores = cross_validate(RF_pipe2, X, y, scoring = scoring, cv = 10, return_train_score = False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))


print("\nRandom Forest with NearMiss 1")
scores = cross_validate(RF_pipe3, X, y, scoring = scoring, cv = 10, return_train_score = False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
    
    
print("\nRandom Forest with NearMiss 2")
scores = cross_validate(RF_pipe4, X, y, scoring = scoring, cv = 10, return_train_score = False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print("\nRandom Forest with NearMiss 3")
scores = cross_validate(RF_pipe5, X, y, scoring = scoring, cv = 10, return_train_score = False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
    
print("\nRandom Forest with SMOTE")
scores = cross_validate(RF_pipe6, X, y, scoring = scoring, cv = 10, return_train_score = False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
    
print("\nRandom Forest with EasyEnsemble")
scores = cross_validate(RF_pipe7, X, y, scoring = scoring, cv = 10, return_train_score = False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))


#SVM

SVM_pipe1 = make_pipeline(LinearSVC(random_state=0))
SVM_pipe2 = make_pipeline(LinearSVC(random_state=0, class_weight = 'balanced'))
SVM_pipe3= make_pipeline(NearMiss(version = 1),
                        LinearSVC(random_state=0, class_weight = 'balanced'))
SVM_pipe4= make_pipeline(NearMiss(version = 2),
                            LinearSVC(random_state=0, class_weight = 'balanced'))
SVM_pipe5= make_pipeline(NearMiss(version = 3),
                        LinearSVC(random_state=0, class_weight = 'balanced'))
SVM_pipe6= make_pipeline(SMOTE(random_state=0, k_neighbors=6, n_jobs=-1),
                        LinearSVC(random_state=0, class_weight = 'balanced'))
SVM_pipe7 = make_pipeline(EasyEnsembleClassifier(base_estimator=LinearSVC(random_state=0, class_weight = 'balanced'),n_jobs=-1, random_state=42))


SVM_pipe1.fit(X_train,y_train)
SVM_pipe2.fit(X_train,y_train)
SVM_pipe3.fit(X_train,y_train)
SVM_pipe4.fit(X_train,y_train)
SVM_pipe5.fit(X_train,y_train)
SVM_pipe6.fit(X_train,y_train)
SVM_pipe7.fit(X_train,y_train)

scoring = ['accuracy', 'balanced_accuracy']

print("SVM")
scores = cross_validate(RF_pipe1, X, y, scoring = scoring, cv = 10, return_train_score = False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print("\nSVM with balanced weights")
scores = cross_validate(RF_pipe2, X, y, scoring = scoring, cv = 10, return_train_score =  False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))


print("\nSVM with NearMiss 1")
scores = cross_validate(RF_pipe3, X, y, scoring = scoring, cv = 10, return_train_score =  False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
    
    
print("\nSVM with NearMiss 2")
scores = cross_validate(RF_pipe4, X, y, scoring = scoring, cv = 10, return_train_score =  False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print("\nSVM with NearMiss 3")
scores = cross_validate(RF_pipe5, X, y, scoring = scoring, cv = 10, return_train_score =  False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
    
print("\nSVM with SMOTE")
scores = cross_validate(RF_pipe6, X, y, scoring = scoring, cv = 10, return_train_score =  False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
    
print("\nSVM with EasyEnsemble")
scores = cross_validate(SVM_pipe7, X, y, scoring = scoring, cv = 10, return_train_score = False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

#Naive Bayes

NB_pipe1 = make_pipeline(GaussianNB())
NB_pipe2= make_pipeline(NearMiss(version = 1),
                        GaussianNB())
NB_pipe3= make_pipeline(NearMiss(version = 2),
                            GaussianNB())
NB_pipe4= make_pipeline(NearMiss(version = 3),
                        GaussianNB())
NB_pipe5= make_pipeline(SMOTE(random_state=0, k_neighbors=6, n_jobs=-1),
                        GaussianNB())
NB_pipe6 = make_pipeline(EasyEnsembleClassifier(base_estimator=GaussianNB(),n_jobs=-1, random_state=42))


NB_pipe1.fit(X_train,y_train)
NB_pipe2.fit(X_train,y_train)
NB_pipe3.fit(X_train,y_train)
NB_pipe4.fit(X_train,y_train)
NB_pipe5.fit(X_train,y_train)
NB_pipe6.fit(X_train,y_train)

scoring = ['accuracy', 'balanced_accuracy']

print("Naive Bayes")
scores = cross_validate(RF_pipe1, X, y, scoring = scoring, cv = 10, return_train_score =  False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print("\nNaive Bayes with NearMiss 1")
scores = cross_validate(RF_pipe2, X, y, scoring = scoring, cv = 10, return_train_score =  False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
    
print("\nNaive Bayes with NearMiss 2")
scores = cross_validate(RF_pipe3, X, y, scoring = scoring, cv = 10, return_train_score =  False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print("\nNaive Bayes with NearMiss 3")
scores = cross_validate(RF_pipe4, X, y, scoring = scoring, cv = 10, return_train_score =  False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
    
print("\nNaive Bayes with SMOTE")
scores = cross_validate(RF_pipe5, X, y, scoring = scoring, cv = 10, return_train_score =  False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
    
print("\nNaive Bayes with EasyEnsemble")
scores = cross_validate(NB_pipe6, X, y, scoring = scoring, cv = 10, return_train_score =  False)
for s in scoring:
    print("%s: %0.2f (+/- %0.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
