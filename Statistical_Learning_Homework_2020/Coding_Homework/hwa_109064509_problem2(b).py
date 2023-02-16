import csv
import numpy as np
import math
import pandas
from matplotlib import pyplot as plt 

def D(x,mu,Sigma):
    return np.dot(np.dot(x,np.linalg.pinv(Sigma)),mu.transpose())-0.5*np.dot(np.dot(mu,np.linalg.pinv(Sigma)),mu.transpose())
def max_index(a,b,c):
    if (a>=b)&(a>=c):
        return 0
    elif (b>=a)&(b>=c):
        return 1
    else:
        return 2

with open('train.csv') as f:
    rows=csv.reader(f)
    next(rows)
    train=np.array(list(rows),dtype=float)
with open('test.csv') as f:
    rows=csv.reader(f)
    next(rows)
    test=np.array(list(rows),dtype=float)   
X0_train=train[0:60000:10,1:]
X1_train=train[1:60000:10,1:]
X2_train=train[2:60000:10,1:]
X0_test=test[0:10240:10,1:]
X1_test=test[1:10240:10,1:]
X2_test=test[2:10240:10,1:]
mu0=np.zeros([1,784])
mu1=np.zeros([1,784])
mu2=np.zeros([1,784])
Sigma=np.zeros([784,784])
for i in range(0,6000):
    mu0+=X0_train[i,:]/6000
    mu1+=X1_train[i,:]/6000
    mu2+=X2_train[i,:]/6000
for i in range(0,6000):
    print(i)
    temp0=X0_train[i,:]-mu0
    temp1=X1_train[i,:]-mu1
    temp2=X2_train[i,:]-mu2
    Sigma+=np.dot(temp0.transpose(),temp0)/(6000-3)+np.dot(temp1.transpose(),temp1)/(6000-3)+np.dot(temp2.transpose(),temp2)/(6000-3)
print(Sigma)
print(np.linalg.pinv(Sigma))
print(np.shape(mu0),np.shape(Sigma))
confusion_matrix=np.zeros([3,3])
for i in range(0,1024):
    print(i)
    y0_es=max_index(D(X0_test[i,:],mu0,Sigma),D(X0_test[i,:],mu1,Sigma),D(X0_test[i,:],mu2,Sigma))
    y1_es=max_index(D(X1_test[i,:],mu0,Sigma),D(X1_test[i,:],mu1,Sigma),D(X1_test[i,:],mu2,Sigma))
    y2_es=max_index(D(X2_test[i,:],mu0,Sigma),D(X2_test[i,:],mu1,Sigma),D(X2_test[i,:],mu2,Sigma))
    confusion_matrix[y0_es][0]+=1
    confusion_matrix[y1_es][1]+=1
    confusion_matrix[y2_es][2]+=1
print(f'   Confusion   |True label')
print(f'    matrix     |  1     2     3')
print(f'_______________________________\n')
print(f' predicted   1 | {confusion_matrix[0][0]}   {confusion_matrix[0][1]}   {confusion_matrix[0][2]}')
print(f'   label     2 | {confusion_matrix[1][0]}  {confusion_matrix[1][1]}   {confusion_matrix[1][2]}')
print(f'             3 | {confusion_matrix[2][0]}  {confusion_matrix[2][1]}   {confusion_matrix[2][2]}')