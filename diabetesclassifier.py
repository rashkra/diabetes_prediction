# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
diabetes=pd.read_csv('C:/Users\Rashmi-PC\downloads\\diabetes.csv')
diabetes.head()
diabetes.shape
diabetes.info()
print(diabetes.keys())
import seaborn as sns
sns.countplot(x='Outcome',data=diabetes,palette='hls')
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
x_train,x_test,y_train,y_test=train_test_split(diabetes.drop('Outcome',1),diabetes.Outcome,test_size=0.3,random_state=21)
LR=LogisticRegression()
LR.fit(x_train,y_train)
y_predict=LR.predict(x_test)
y_prob=LR.predict_proba(x_test)[:,1]# only keep the first column which is a pos value
from sklearn.metrics import confusion_matrix
c=metrics.confusion_matrix(y_test,y_predict)
c
sensitivity=c[0][0]/(c[0][0]+c[0][1])
specificity=c[1][1]/(c[1][1]+c[1][0])
print('sensitivity=',sensitivity)
print('specificity=',specificity)
fpr,tpr,threshold=metrics.roc_curve(y_test,y_prob)
plt.plot(fpr,tpr)
plt.title('ROC Curve for Diabetes Classifier')
plt.xlabel('false positive rate(1-specificity)')
plt.ylabel('True positive rate(sensitivity)')
plt.grid(True)
metrics.auc(fpr,tpr)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(6)
knn.fit(diabetes.drop('Outcome',1),diabetes.Outcome)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
import numpy as np
n=np.arange(1,9)
test_accuracy=np.empty(len(n))
for i ,k in enumerate(n):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    test_accuracy[i]=knn.score(x_test,pred)
plt.title('k-NN:varying no. of neighbors')
plt.plot(n,test_accuracy,label='testing accuracy')
plt.legend()
plt.xlabel('no. of neighbors')
plt.ylabel('accuracy')
plt.grid()
plt.show()
