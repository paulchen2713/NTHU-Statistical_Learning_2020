import csv
import numpy as np
import math
import pandas
from matplotlib import pyplot as plt 


with open('train.csv') as f:
    rows=csv.reader(f)
    next(rows)
    train=np.array(list(rows),dtype=float)
with open('test.csv') as f:
    rows=csv.reader(f)
    next(rows)
    test=np.array(list(rows),dtype=float)   
X1_train=np.append(np.ones([6000,1]),train[1:60000:10,1:],axis=1)
X2_train=np.append(np.ones([6000,1]),train[2:60000:10,1:],axis=1)
X1_test=np.append(np.ones([1024,1]),test[1:10240:10,1:],axis=1)
X2_test=np.append(np.ones([1024,1]),test[2:10240:10,1:],axis=1)

beta=np.zeros(785)


print('training start')
#iterations=0
step_size=0.1
while 1 :   
    gradient=np.zeros(785)
    for i in range(0,6000):
        gradient+=-1*X1_train[i,:]/(1+np.exp(-1*np.dot(X1_train[i,:],beta)))
        gradient+=X2_train[i,:]/(1+np.exp(np.dot(X2_train[i,:],beta)))
    gradient=-gradient
    beta-=step_size*gradient
    #print(iterations,np.linalg.norm(gradient),gradient.sum(),np.linalg.norm(beta),beta.sum())
    #iterations=iterations+1
    if np.linalg.norm(gradient)<=10000 :
        break


error1=0
error2=0
for i in range(0,1024):
    temp1=np.exp(np.dot(X1_test[i,:],beta))
    temp2=np.exp(np.dot(X2_test[i,:],beta))

    if math.isinf(temp1):
        y1_es=2
    else:
        y1_es=(temp1/(1+temp1)>0.5)+1

    if math.isinf(temp2):
        y2_es=2
    else:
        y2_es=(temp2/(1+temp2)>0.5)+1
    error1+=(1!=y1_es)
    error2+=(2!=y2_es)
confusion_matrix=np.array([[1024-error1,error2],[error1,1024-error2]])

print(f'   Confusion   |True label')
print(f'    matrix     |  1     2')
print(f'_______________________________\n')
print(f' predicted   1 | {confusion_matrix[0][0]}   {confusion_matrix[0][1]}')
print(f'   label     2 | {confusion_matrix[1][0]}  {confusion_matrix[1][1]}\n')
print(f'error of label 1 is {error1/1024}')
print(f'error of label 2 is {error2/1024}')
print(f'average error is {(error1+error2)/2048}')