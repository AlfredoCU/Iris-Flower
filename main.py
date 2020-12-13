###############################################################################
###############################################################################
###############################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 12:10:45 2020

@author: alfredocu
"""

###############################################################################

# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

###############################################################################

# Load dataset
data = pd.read_csv("IRIS.csv")

###############################################################################

# Shape
print(data.shape)

'''
(150, 5)
'''

###############################################################################

# Information
print(data.info())

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object 
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
None
'''

###############################################################################

# Show data
print(data)

'''
     sepal_length  sepal_width  petal_length  petal_width         species
0             5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
3             4.6          3.1           1.5          0.2     Iris-setosa
4             5.0          3.6           1.4          0.2     Iris-setosa
..            ...          ...           ...          ...             ...
145           6.7          3.0           5.2          2.3  Iris-virginica
146           6.3          2.5           5.0          1.9  Iris-virginica
147           6.5          3.0           5.2          2.0  Iris-virginica
148           6.2          3.4           5.4          2.3  Iris-virginica
149           5.9          3.0           5.1          1.8  Iris-virginica

[150 rows x 5 columns]
'''

###############################################################################

# Descriptions
print(data.describe())

'''
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
'''

###############################################################################

# Species distribution
print(data.groupby("species").size())

'''
species
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
dtype: int64
'''

###############################################################################
###############################################################################
###############################################################################

# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.plot()
# plt.savefig("DA.eps", format="eps")

###############################################################################

# data.hist()
# plt.plot()
# plt.savefig("HV.eps", format="eps")

###############################################################################

# pd.plotting.scatter_matrix(data)
# plt.plot()
# plt.savefig("GRAFICS.eps", format="eps")

###############################################################################
###############################################################################
###############################################################################

# Split-out validation dataset
array = data.values

X = array[:,0:4]
Y = array[:,4]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

###############################################################################

# Spot Check Algorithms
models = []
models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma='auto')))
models.append(("MLP", MLPClassifier(alpha = 1, max_iter = 1000)))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
	results.append(cv_results)
	names.append(name)
	print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

'''
LR: 0.941667 (0.065085)
LDA: 0.975000 (0.038188)
KNN: 0.958333 (0.041667)
CART: 0.941667 (0.053359)
NB: 0.950000 (0.055277)
SVM: 0.983333 (0.033333)
MLP: 0.975000 (0.038188)
'''

###############################################################################

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title("Algorithm Comparison")
plt.show()
# plt.savefig("AC.eps", format="eps")

###############################################################################
###############################################################################
###############################################################################

# Make predictions on validation dataset
model = SVC(gamma="auto")
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

###############################################################################

# Evaluate predictions
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

###############################################################################

'''
0.9666666666666667

[[11  0  0]
 [ 0 12  1]
 [ 0  0  6]]
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        11
Iris-versicolor       1.00      0.92      0.96        13
 Iris-virginica       0.86      1.00      0.92         6

       accuracy                           0.97        30
      macro avg       0.95      0.97      0.96        30
   weighted avg       0.97      0.97      0.97        30
'''

###############################################################################

# Score
score_train = model.score(X_train, Y_train)
score_test = model.score(X_test, Y_test)

print(score_train)
print(score_test)

'''
Train: 0.9833333333333333
Test: 0.9666666666666667
'''

###############################################################################
