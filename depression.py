import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import copy
from sklearn import preprocessing

mydataset = pd.read_excel("C:\\Users\\Vyborg\\Desktop\\MyProject\\corrected\\run\\student_data1000.xlsx")
print(mydataset.head(20))
print(" ")
cat_students = mydataset.select_dtypes(include=['object']).copy()
print(cat_students.head(20))
print(" ")
mydataset['school']         = mydataset['school'].astype('category')
mydataset['sex']         = mydataset['sex'].astype('category')
mydataset['address']     = mydataset['address'].astype('category')
mydataset['famsize']     = mydataset['famsize'].astype('category')
mydataset['Pstatus']     = mydataset['Pstatus'].astype('category')
mydataset['Mjob']        = mydataset['Mjob'].astype('category')
mydataset['Fjob']        = mydataset['Fjob'].astype('category')
mydataset['reason']      = mydataset['reason'].astype('category')
mydataset['guardian']    = cat_students['guardian'].astype('category')
mydataset['schoolsup']   = cat_students['schoolsup'].astype('category')
mydataset['famsup']      = cat_students['famsup'].astype('category')
mydataset['paid']        = cat_students['paid'].astype('category')
mydataset['activities']  = cat_students['activities'].astype('category')
mydataset['nursery']     = cat_students['nursery'].astype('category')
mydataset['higher']      = cat_students['higher'].astype('category')
mydataset['internet']    = cat_students['internet'].astype('category')
mydataset['romantic']    = cat_students['romantic'].astype('category')
mydataset['school']      =mydataset['school'].cat.codes
mydataset['sex']         = mydataset['sex'].cat.codes
mydataset['address']     = mydataset['address'].cat.codes
mydataset['famsize']     = mydataset['famsize'].cat.codes
mydataset['Pstatus']     = mydataset['Pstatus'].cat.codes
mydataset['Mjob']        = mydataset['Mjob'].cat.codes
mydataset['Fjob']        = mydataset['Fjob'].cat.codes
mydataset['reason']      = mydataset['reason'].cat.codes
mydataset['guardian']    = mydataset['guardian'].cat.codes
mydataset['schoolsup']   = mydataset['schoolsup'].cat.codes
mydataset['famsup']      = mydataset['famsup'].cat.codes
mydataset['paid']        = mydataset['paid'].cat.codes
mydataset['activities']  = mydataset['activities'].cat.codes
mydataset['nursery']     = mydataset['nursery'].cat.codes
mydataset['higher']      = mydataset['higher'].cat.codes
mydataset['internet']    = mydataset['internet'].cat.codes
mydataset['romantic']    = mydataset['romantic'].cat.codes
print(" ")
print(mydataset.head(20))
print("")
print(mydataset.shape)
print(" ")
print(mydataset.groupby("failures").size())
print(" ")
summary = mydataset.describe()
print(summary)
mydataset = mydataset.values
print(" ")
print(mydataset)
print(" ")
X = mydataset[:,:-1] #Features
Y = mydataset[:,-1]  #Class_Labels

print(X.shape) 
print(Y.shape)
print("")
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state = 75)
print(x_test.shape)
print(x_train.shape)
print(y_train.shape)
print(y_test.shape)
print(" ")
print(" This is x_test")
print(x_test)
print(" ")
print(" This is x_train")
print(x_train)
print(" ")
clf1 = GaussianNB()
y_train=y_train.astype('int')
x_train=x_train.astype('int')
clf1.fit(x_train,y_train)
x_test=x_test.astype('int')
y_test=y_test.astype('int')
predict1 = clf1.predict(x_test)
percent1 = accuracy_score(y_test,predict1)
percent = percent1 * 100
print(" ")
print("Naive_Bayes score: ", percent1, "or", int(percent),"%")
print(" ")
print("confusion_matrix:\n", confusion_matrix(y_test,predict1))
print(" ")
print("classification_report:\n", classification_report(y_test,predict1))

"""
A Naive bayes approach to the prediction of depression using 
"""

