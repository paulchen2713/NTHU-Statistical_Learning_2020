# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:04:48 2019

@author: chih-hsuan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy.linalg as lg
import seaborn as sns
from sklearn import preprocessing


def confusion_matrix(true, pred):
    result = np.zeros((3, 3))
    for i in range(len(true)):
        if true[i]==1:
            if pred[i]==1:
                result[0,0]=result[0,0]+1
            elif pred[i]==2:
                result[1,0]=result[1,0]+1
            else:
                result[2,0]=result[2,0]+1
        elif true[i]==2:
            if pred[i]==1:
                result[0,1]=result[0,1]+1
            elif pred[i]==2:
                result[1,1]=result[1,1]+1
            else:
                result[2,1]=result[2,1]+1
        else:
            if pred[i]==1:
                result[0,2]=result[0,2]+1
            elif pred[i]==2:
                result[1,2]=result[1,2]+1
            else:
                result[2,2]=result[2,2]+1
    return result
#
#def param_calcu(X_train,y_train):
#    nclass = len(np.unique(y_train))
#    u_kclass = []
#    all_num = len(y_train)
#    var = np.zeros((X_train.shape[1],X_train.shape[1]))
#    phi = np.zeros(nclass)
#    for i,element in enumerate(np.unique(y_train)) :
#        num_class = np.sum(y_train == np.unique(y_train)[i]) 
#        tmp = X_train[y_train == np.unique(y_train)[i],:]
#        u_kclass.append((1/num_class)*np.sum(tmp,axis=0))
##        var = var + 1/(all_num-num_class)*np.sum((tmp-u_kclass[i])**2)#p=1
#        for j in range(0,tmp.shape[0]):
#            var = var + (1/(all_num-nclass))*np.dot((tmp[j]-u_kclass[i]).reshape((X_train.shape[1],1)),np.transpose((tmp[j]-u_kclass[i]).reshape((X_train.shape[1],1))))#P>1
#        phi[i] = num_class/all_num
#    return u_kclass,phi,var

def param_calcu(X_train,y_train):
    mu=np.zeros((3,len(X_train[1])))
    n=np.zeros(3)
    phi=np.zeros(3)
    var=np.zeros((len(X_train[1]),len(X_train[1])))
          
    for i in range(len(y_train)):
            #print(y_train[i])
        if y_train[i]==1:
            mu[0]=mu[0]+X_train[i]
            n[0]=n[0]+1
        elif y_train[i]==2:
            mu[1]=mu[1]+X_train[i]
            n[1]=n[1]+1
        else:
            mu[2]=mu[2]+X_train[i]
            n[2]=n[2]+1
    
    num=n[0]+n[1]+n[2] 
    
    for j in range(3):
        mu[j]=mu[j]/n[j]
        phi[j]=n[j]/len(y_train)
        #print(j)
        
        for i in range(len(y_train)):
            if y_train[i]==j+1:
                a1=X_train[i]
                a2=mu[j]
                a3=(X_train[i]-mu[j]).reshape(len(X_train[1]),1)
                var=var+np.dot(a3,a3.T)
                #var=var+np.dot((X_train[i]-mu[j]),(X_train[i]-mu[j]).T)
#        
#        tmp = X_train[y_train == np.unique(y_train)[j],:]
#        for i in range(0,tmp.shape[0]):
#            var = var + np.dot((tmp[i]-mu[j]).reshape((X_train.shape[1],1)),np.transpose((tmp[i]-mu[j]).reshape((X_train.shape[1],1))))#P>1

    #print(num)# should be 1440
    var=var/(num-3)

    return mu,phi,var
    

def delta_fun(x,k,mu,phi,var):
    u = mu[k].reshape(len(mu[k]),1)
    #print(x)
    #np.dot(np.dot(x.T,lg.pinv(var)),u)
    #np.dot(np.dot(u.T,lg.pinv(var)),u)
    delta=np.dot(np.dot(x.T,lg.pinv(var)),u)-0.5*np.dot(np.dot(u.T,lg.pinv(var)),u)+math.log(phi[k])
    return delta

n=1440
data1 = pd.read_csv('csvTrainImages 13440x1024.csv',header = None)
label1 = pd.read_csv('csvTrainLabel 13440x1.csv',header = None)

for i in range(480):
    if i==0:
        x = data1.iloc[0:24, :].values  # take the feature value we want
        y = label1.iloc[0:24, 0].values  # take the feature value we want
    else:
        k=i*224
        x_data = data1.iloc[k:k+24, :].values  # take the feature value we want
        y_data = label1.iloc[k:k+24, 0].values  # take the feature value we want
        x = np.r_[x,x_data]
        y = np.r_[y,y_data]

mu,phi,var = param_calcu(x,y)

data2 = pd.read_csv('csvTestImages 3360x1024.csv',header = None)
label2 = pd.read_csv('csvTestLabel 3360x1.csv',header = None)
for i in range(120):
    if i==0:
        x_test = data2.iloc[0:6, :].values  # take the feature value we want
        y_test = label2.iloc[0:6, 0].values  # take the feature value we want
    else:
        k=i*56
        x_data = data2.iloc[k:k+6, :].values  # take the feature value we want
        y_data = label2.iloc[k:k+6, 0].values  # take the feature value we want
        x_test = np.r_[x_test,x_data]
        y_test = np.r_[y_test,y_data]
        
n=360
y_es = np.zeros(n)
pred=np.zeros(3)
for i in range(n):
    print(i)
    for j in range(3):
        pred[j] = delta_fun(x_test[i].reshape(len(x_test[i]),1),j,mu,phi,var)
        
    y_es[i]=np.argmax(pred)+1
    
    
confusion=confusion_matrix(y_test,y_es)
print(confusion)
