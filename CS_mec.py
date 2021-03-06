# -*- coding: utf-8 -*-
"""PA_MEC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1v_9ZP9-d0WWrEsowcnfE-JiVuGfQEb7P
"""
#COST SENSITIVE LEARNING - MINIMIZING THE EXPECTED COST

!pip install scikit-learn==0.22.2.post1
!pip install costcla

import pandas as pd
import joblib
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from costcla.metrics import cost_loss
from sklearn.calibration import CalibratedClassifierCV
from costcla.models import BayesMinimumRiskClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

data = np.loadtxt('heart.dat', unpack = True)
a = data.transpose()
header_list = ["1","2","3","4","5","6","7","8","9","10","11","12","13","target"]
np.savetxt('heart.csv', a, delimiter=',')

dataset = pd.read_csv("heart.csv", names=header_list)
dataset.target.replace({1:0, 2:1}, inplace = True)

dataset

frame = pd.DataFrame(dataset.target, columns=['target'])
print(frame['target'].value_counts())

X = dataset.iloc[:,0:13]
y = dataset.iloc[:,13]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=dataset.target)

fp = np.full((y_test.shape[0],1), 1)
fn = np.full((y_test.shape[0],1), 5)
tp = np.zeros((y_test.shape[0],1))
tn = np.zeros((y_test.shape[0],1))
cost_matrix = np.hstack((fp, fn, tp, tn))

cost_m = [[0 , 5], [1, 0]]
print("\nCost matrix: ", cost_m)

#Random Forest

print("Random Forest with no cost minimization")
RF_clf = RandomForestClassifier(random_state=0, n_estimators=100)
model = RF_clf.fit(X_train, y_train)
pred_test = model.predict(X_test)
conf_m = confusion_matrix(y_test, pred_test).T
print(f"Loss = {int(np.sum(conf_m*cost_m))}\n")
print(confusion_matrix(y_test,pred_test).T)

print("\nRandom Forest with sigmoid calibration")
RF_clf = RandomForestClassifier(random_state=0, n_estimators=100)
cc = CalibratedClassifierCV(RF_clf, method="sigmoid", cv=6)
model = cc.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
conf_m = confusion_matrix(y_test, pred_test).T
print(f"Loss = {int(np.sum(conf_m*cost_m))}\n")
print(confusion_matrix(y_test,pred_test).T)

#SVM

print("SVM with no cost minimization")
SVM_clf = SVC(kernel='linear', random_state=0, probability=True)
model = SVM_clf.fit(X_train, y_train)
pred_test = model.predict(X_test)
conf_m = confusion_matrix(y_test, pred_test).T
print(f"Loss = {int(np.sum(conf_m*cost_m))}\n")
loss = cost_loss(y_test, pred_test, cost_matrix)
print(confusion_matrix(y_test,pred_test).T)

print("\nSVM with sigmoid calibration")
SVM_clf = SVC(kernel='linear', C=1)
cc = CalibratedClassifierCV(SVM_clf, method="sigmoid", cv=3)
model = cc.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
conf_m = confusion_matrix(y_test, pred_test).T
print(f"Loss = {int(np.sum(conf_m*cost_m))}\n")
print(confusion_matrix(y_test,pred_test).T)

#Naive Bayes

print("Naive Bayes with no cost minimization")
NB_clf = GaussianNB()
model = NB_clf.fit(X_train, y_train)
pred_test = model.predict(X_test)
conf_m = confusion_matrix(y_test, pred_test).T
print(f"Loss = {int(np.sum(conf_m*cost_m))}\n")
loss = cost_loss(y_test, pred_test, cost_matrix)
print(confusion_matrix(y_test,pred_test).T)

print("\nNaive Bayes with sigmoid calibration")
NB_clf = GaussianNB()
cc = CalibratedClassifierCV(NB_clf, method="sigmoid", cv=3)
model = cc.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
conf_m = confusion_matrix(y_test, pred_test).T
print(f"Loss = {int(np.sum(conf_m*cost_m))}\n")
print(confusion_matrix(y_test,pred_test).T)
