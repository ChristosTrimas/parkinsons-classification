import pandas as pd
import os , sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Setup Ready!")

"""
I will try to classify the parkinson dataset with 
1)Support Vector Classifier
2)Naive Bayes
3)Logistic Regression
4)Stochastic Gradient Descent
5)K-Nearest Neighbors
6)Random Forest Classifier
"""

"""
Lets check the dataset first
"""

ps = pd.read_csv('/home/broly/Desktop/parkinsons.data')

features = ps.loc[:,ps.columns!='status'].values[:,1:]
labels = ps.loc[:,'status'].values
group_names = ['no_park','park']
ps['status'] = pd.cut(ps['status'],bins=2,labels=group_names)
ps['status'].unique()
print(ps['status'])


"""
Splitiing the data to train and test data
"""

X_train, X_test,y_train,y_test = train_test_split(features,labels, test_size = 0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Random Forest --------------------------------------------
model_1 = RandomForestClassifier(n_estimators=200)
model_1.fit(X_train,y_train)
m1_prediction = model_1.predict(X_test)
# print(m1_prediction[:20])


# performance
print(classification_report(y_test,m1_prediction))
print(confusion_matrix(y_test,m1_prediction))

# -----------------------------------------------


