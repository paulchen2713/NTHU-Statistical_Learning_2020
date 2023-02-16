# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:34:53 2019

@author: chih-hsuan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy.linalg as lg
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


def Del_J(x,y,beta,n):
    delj = np.zeros((1025, 1))
    
    for i in range(n):
        xi=x[i,:]
        xi = xi.reshape(1025,1)
        numerator = (2*y[i]-1)*(xi) #分子
        exponent = (-1)*np.dot((2*y[i]-1)*xi.T,beta)
        denominator = 1+np.exp(exponent)
        #print(numerator)
        #print(denominator)
        #print(exponent)
        #print(np.exp(exponent))
        #print(denominator)
        #print(numerator/denominator)
        delj = (numerator/denominator+delj)
    delj = (-1)*delj
    #print(numerator) 
    return delj

def confusion_matrix(true, pred):
    result = np.zeros((2, 2))
    for i in range(len(true)):
        if true[i]==1:
            if pred[i]==1:
                result[0][0]=result[0][0]+1
            else:
                result[1][0]=result[1][0]+1
        else:
            if pred[i]==2:
                result[1][1]=result[1][1]+1
            else:
                result[0][1]=result[0][1]+1
    return result
    
    
n=960
data1 = pd.read_csv('csvTrainImages 13440x1024.csv',header = None)
label1 = pd.read_csv('csvTrainLabel 13440x1.csv',header = None)

for i in range(480):
    if i==0:
        x = data1.iloc[0:16, :].values  # take the feature value we want
        y = label1.iloc[0:16, 0].values  # take the feature value we want
    else:
        k=i*224
        x_data = data1.iloc[k:k+16, :].values  # take the feature value we want
        y_data = label1.iloc[k:k+16, 0].values  # take the feature value we want
        x = np.r_[x,x_data]
        y = np.r_[y,y_data]
y=y-np.ones(n)
y = y.reshape(n,1)

#x = preprocessing.scale(x)
x = x/255

Trainingset = np.c_[np.ones(n),x]
beta = np.zeros(1025)
beta = beta.reshape(1025,1)
eta = 0.0001

#delj=Del_J(Trainingset,y,beta,n)
#print(delj)
#print(Del_J(Trainingset,y,beta,n))
#print(max(delj))

for i in range(1000):
    delj=Del_J(Trainingset,y,beta,n)
    #print(delj) 
    beta = beta-eta*delj
    #beta = beta-eta*Del_J(Trainingset,y,beta,n)
    
 
# Testing
data2 = pd.read_csv('csvTestImages 3360x1024.csv',header = None)
label2 = pd.read_csv('csvTestLabel 3360x1.csv',header = None)

for i in range(120):
    if i==0:
        x_test = data2.iloc[0:4, :].values  # take the feature value we want
        y_test = label2.iloc[0:4, 0].values  # take the feature value we want
    else:
        k=i*56
        x_data = data2.iloc[k:k+4, :].values  # take the feature value we want
        y_data = label2.iloc[k:k+4, 0].values  # take the feature value we want
        x_test = np.r_[x_test,x_data]
        
        y_test = np.r_[y_test,y_data]
x_test = x_test/255
    
n=240
Testingset = np.c_[np.ones(n),x_test]
y_p = np.zeros(n)
y_es = np.zeros(n)
for i in range(n):
    xi=Testingset[i,:]
    xi = xi.reshape(1025,1)
    z = np.dot(xi.T,beta)
    #y_p[i] = float(np.exp(-1.0*z) / float((1.0 + np.exp(-1.0*z))))
    y_p[i] = 1 / float((1.0 + np.exp(-1.0*z)))
    if y_p[i]>0.5:
        y_es[i]=2
    else:
        y_es[i]=1
        
confusion=confusion_matrix(y_test,y_es)
print(confusion)
print(accuracy_score(y_test,y_es))