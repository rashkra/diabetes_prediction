
import pandas as pd

#importing the data using pandas's read_csv module

diabetes=pd.read_csv('../input/pima-indians-diabetes-database//diabetes.csv')

#Exploring the data

diabetes.head()
diabetes.shape
diabetes.info() # this will tell us about the features' type and available(not NaN) values
print(diabetes.keys())

# Visualizing the data

import seaborn as sns
sns.countplot(x='Outcome',data=diabetes,palette='hls')

#importing libraries for generating the model.

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#model selection and validation

x_train,x_test,y_train,y_test=train_test_split(diabetes.drop('Outcome',1),diabetes.Outcome,test_size=0.3,random_state=21)
LR=LogisticRegression()
LR.fit(x_train,y_train)
y_predict=LR.predict(x_test)
y_prob=LR.predict_proba(x_test)[:,1]# only keep the first column which is a pos value

#checing the accuracy of the model
from sklearn.metrics import confusion_matrix
c=metrics.confusion_matrix(y_test,y_predict)
c # printing the confusion matrix

#considering positive to be not having diabetes and negative to be having diabetes.

sensitivity=c[0][0]/(c[0][0]+c[0][1])
specificity=c[1][1]/(c[1][1]+c[1][0])
print('sensitivity=',sensitivity)
print('specificity=',specificity)

#checking accuracy through Reciever operating characteristic curve and
#calculating area under curve which provides the accuracy of the model.

fpr,tpr,threshold=metrics.roc_curve(y_test,y_prob) #fpr=false positive rate, tprtrue positive rate
plt.plot(fpr,tpr)
plt.title('ROC Curve for Diabetes Classifier')
plt.xlabel('false positive rate(1-specificity)')
plt.ylabel('True positive rate(sensitivity)')
plt.grid(True)
metrics.auc(fpr,tpr)

