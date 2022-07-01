#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -U imbalanced-learn


# In[2]:


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


# In[3]:


dataset = pd.read_csv("creditcard.csv")

X = dataset.iloc[:,0:30]
y = dataset.iloc[:,30]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dataset


# In[4]:


print(Counter(y))


# In[5]:


#Παρακάτω εκπαιδεύονται 7 μοντέλα για τους Classifiers Random Forest και SVM και 6 μοντέλα για τον ταξινομητή Naive Bayes
#Στα μοντέλα αυτά εφαρμόζονται 3 διαφορετικές τεχνικές για Class Imbalance με σκοπό να αξιολογηθούν οι επιδόσεις των 3 αυτών
#αλγορίθμων πριν και μετά την εφαρμογή της εκάστοτε τεχνικής πάνω στο συγκεκριμένο dataset. Οι τεχνικές που εφαρμόζονται για 
#κάθε ταξινομήτη είναι οι NearMiss, SMOTE και EasyEnsemble. Στις μετρικές παρουσιάζονται οι accuracy και balanced_accuracy. 
#Οι τιμές της πρώτης σε προβλήματα τέτοιου τύπου συνήθως ειναι παραπλανητικές, συνεπώς η αξιολόγηση των μοντέλων θα γίνει με 
#βάση την balanced_accuracy.


# In[6]:


#Random Forest


# In[7]:


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


# In[8]:


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


# In[9]:


#Σύμφωνα με τα αποτελέσματα, παρατηρούμε ότι ο Random Forest στο συγκεκριμένο
#σύνολο δεδομένων, πετυχαίνει ακρίβεια (balanced_accuracy) 83%, ενώ με balanced
#weights βελτιώνεται ελαφρώς (85%). Με την εφαρμογή της τεχνικής NearMiss η 
#ακρίβεια του Random Forest κυμαίνεται από 76% έως 93% με χειρότερες επιδόσεις
#στη version2 (68%) και καλύτερη την version3. Eπιπλέον, με την εφαρμογή του
#SMOTE ο αλγόριθμος βελτιώνει την ακρίβεια του κατά 6% από την αρχική (από 83% 
#σε 89%). Τέλος, παρατηρείται ότι ο συγκεκριμένος ταξινομητής πετυχαίνει την
#βέλτιστη ακρίβεια του στο συγκεκριμένο dataset με την εφαρμογή της τεχνικής
#EasyEnsemble όπου φτάνει το 94%!
#Best balanced_accuracy score: 94% - EasyEnsemble
#Worst balanced_accuracy score: 68% - NearMiss version2


# In[10]:


#SVM


# In[11]:


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


# In[12]:


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


# In[13]:


#Σύμφωνα με τα αποτελέσματα, παρατηρούμε ότι ο SVM πετυχαίνει ακριβώς τα ίδια
#επίπεδα ακρίβειας με τον Random Forest σε όλα τα στάδια, με εξαίρεση την
#εφαρμογή της EasyEnsemble όπου παρατηρείται χαμηλότερη επίδοση. Πιο 
#συγκεκριμένα με την εφαρμογ΄ή της EasyEnsemble, η επίδοση του αλγορίθμου στο
#συγκεκριμένου dataset φαινεται να χειροτερεύει, καθώς η ακρίβεια πέφτει από
#το αρχικό 83% στο 70%.
#Best balanced_accuracy score: 93% - NearMiss version3
#Worst balanced_accuracy score: 68% - NearMiss version2


# In[14]:


#Naive Bayes


# In[15]:


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


# In[16]:


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


# In[17]:


#Για τον Naive Bayes εκπαιδεύτηκε ένα μοντέλο λιγότερο, καθώς δεν χρησιμοποιή-
#θηκε κάποια μέθοδος για balanced_weights. Σύμφωνα με τα αποτελέσματα, ο Naive
#Bayes πετυχαίνει επίσης ακρίβεια (balanced_accuracy) 83% χωρις την εφαρμογή 
#κάποιας τεχνικής για class imbalance. Με την εφαρμοφή της NearMiss μεθόδου, 
#η ακρίβεια του ταξινομητή, κυμαίνεται από 68% εως 85%. Εδώ μάλιστα τα πράγματα
#φαίνεται να αντιστρέφονται συγκριτικά με τους άλλους δυο αλγορίθμους. Στο 
#NearMiss version3 ο Naive Bayes πετυχαίνει την χειρότερη απόδοση του, ενώ
#με την εφαρμογή του SMOTE η ακρίβεια βελτιστοποιείται στο 93%. Τέλος, η
#EasyEnsemble φαίνεται να βελτιώνει την επίδοση του Naive Bayes κατα 3% από
#την αρχική επίδοση (από 83% σε 86%).
#Best balanced_accuracy score: 93% - SMOTE
#Worst balanced_accuracy score: 68% - NearMiss version3


# In[18]:


#Συνολικά, παρατηρούμε για τους 3 αλγορίθμους ότι η εφαρμοφή κάποιας εκ των 3
#τεχνικών μπορεί να βελτιώσει σημαντικά την επίδοση τους. Και οι 3 αλγόριθμοι
#με βάση τη μετρική balanced_accuracy πετυχαίνουν αρχικά ακρίβεια της τάξης του
#83% ενώ με την εφαρμογή των τεχνικών και την επιλογή της καταλληλότερης για 
#τον κάθε αλγόριθμο η ακρίβεια βελτιώνεται κατά 10% στους SVM και Naive Bayes
#και κατά 11% στον Random Forest. Αυτή μάλιστα είναι και η καλύτερη επίδοση 
#(94%) που παρατηρήθηκε μεταξύ των 20 μοντέλων που εκπαιδεύσαμε και επιτεύχθηκε
#με την εφαρμογή της EasyEnsemble. Η χειρότερη επίδοση και στους 3 αλγορίθμους
#ήταν 68% και παρατηρήθηκε σε κάποια από τις εκδοχές της NearMiss, πράγμα που
#υποδηλώνει ότι για την συγκεκριμένη τεχνική οι επιδόσεις μεταξύ των
#διαφόρων εκδοχών της μπορεί να διαφέρουν από αλγόριθμο σε αλγόριθμο.

