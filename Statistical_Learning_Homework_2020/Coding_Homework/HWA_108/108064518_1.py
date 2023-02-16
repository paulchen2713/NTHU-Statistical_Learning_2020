# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy.linalg as lg
import seaborn as sns
from scipy.stats import t
from mpl_toolkits.mplot3d import Axes3D


def coefficient(x,y):
    return np.dot(np.dot(lg.inv(np.dot(x.T,x)),x.T),y)

def std_error(x,y,y_es,n,p):    
    #y_es = np.dot(x,beta)
    sigma_square = np.dot((y-y_es).T,(y-y_es).T)/(n-p-1)
    a=lg.inv(np.dot(x.T,x))*sigma_square
    ##std_error
    #std_error = np.array(([math.sqrt(a[0,0])],[math.sqrt(a[1,1])],[math.sqrt(a[2,2])],[math.sqrt(a[3,3])],[math.sqrt(a[4,4])],[math.sqrt(a[5,5])])) 
    
    std_error = np.zeros(p+1)
    for i in range(0,p+1):
        std_error[i] = math.sqrt(a[i,i])
        #std_error1 = np.r_[std_error1,math.sqrt(a[i,i])]
    #std_error1 = np.array(([math.sqrt(a[i,i])] for i in range(0,p) ))
    
    return std_error

def t_statistic(beta,std_err):
    #np.true_divide: division with floating-point number reserved
    t = np.true_divide(beta.T,std_err.T) 
    return t.T

def p_value(t_st,n,p):
    # cdf(x, df, loc=0, scale=1)
    # cdf of the t-distribution at each of the values x with degrees of freedom=df
    p_val=(1-t.cdf(abs(t_st),n-p-1))*2
    return p_val

def R2_statistic(y,y_es,n,p):
    rss = np.dot((y-y_es).T,(y-y_es).T)
    y_mean = np.ones(n)*np.mean(y)
    tss = np.dot((y-y_mean).T,(y-y_mean).T)
    r_2 = (tss-rss)/tss
    return r_2

## 1-a

dataset = pd.read_excel("Real estate valuation data set.xlsx")
n=414
p1=5
data = dataset.iloc[0:n, 2:7].values  # take the feature value we want
y = dataset.iloc[0:n, 7].values
#print(data)

x = np.c_[np.ones(n),data] # add ones to the first column of data
del data #free memories
#print(x)

# matrix multiplication: np.dot() 
# inverse: np.linalg.inv->lg,inv
#beta = np.dot(np.dot(lg.inv(np.dot(x.T,x)),x.T),y) ##coefficient
beta = coefficient(x,y)

y_es = np.dot(x,beta)
##std_error
#y_es = np.dot(x,beta)
#sigma_square = np.dot(y.T-y_es,y.T-y_es)/(n-p-1)
#a=lg.inv(np.dot(x.T,x))*sigma_square
#std_error = np.array(([math.sqrt(a[0,0])],[math.sqrt(a[1,1])],[math.sqrt(a[2,2])],[math.sqrt(a[3,3])],[math.sqrt(a[4,4])],[math.sqrt(a[5,5])])) 
std_err = std_error(x,y,y_es,n,p1)
t_st = t_statistic(beta,std_err)
p_val = p_value(t_st,n,p1)


## 1-b
rss = np.zeros(10)
index = np.zeros((2,10))
p2=2
k=0
for i in range(1,5):
    for j in range(i+1,6):
        x_1 = x[:,[0,i,j]]
        beta_1 = coefficient(x_1,y)
        y_es_1 = np.dot(x_1,beta_1)
        rss[k] = np.dot((y-y_es_1).T,(y-y_es_1).T)
        index[:,k] = [i,j]
        k=k+1
min_choice=index[:,np.argmin(rss)]
print("The choice that minimize RSS is:")
print("X",int(min_choice[0]),int(min_choice[1]))
x_1 = x[:,[0,int(min_choice[0]),int(min_choice[1])]]
beta_1 = coefficient(x_1,y)
y_es_1 = np.dot(x_1,beta_1)
std_err_1 = std_error(x_1,y,y_es_1,n,p2)
t_st_1 = t_statistic(beta_1,std_err_1)
p_val_1 = p_value(t_st_1,n,p2)

rse_a = math.sqrt(np.dot((y-y_es).T,(y-y_es).T)/(n-p1-1))
rse_b = math.sqrt(np.dot((y-y_es_1).T,(y-y_es_1).T)/(n-p2-1))
r_2a = R2_statistic(y,y_es,n,p1)
r_2b = R2_statistic(y,y_es_1,n,p2)

## 1-c

X=np.arange(0,6400)
Y=np.arange(0,10)
B1, B2 = np.meshgrid(X, Y)
Z = np.zeros((Y.size, X.size))

for (i,j),v in np.ndenumerate(Z):
    Z[i,j] = (beta_1[0] + B1[i,j]*beta_1[1] + B2[i,j]*beta_1[2])

fig = plt.figure(figsize=(10,6))
#fig.suptitle('Regression: house price/area ~ distance to the nearest MRT station + number of convenience stores', fontsize=20)
ax = Axes3D(fig)
## alpha: tranparency,rstride:Array row stride(step size),cstride:Array column stride(step size)
ax.plot_surface(B1, B2 ,Z , alpha=0.4)
ax.scatter3D(x_1[:,1],x_1[:,2],y, c='r')
ax.set_xlabel('distance to the nearest MRT station')
ax.set_ylabel('number of convenience stores')
ax.set_zlabel('house price of unit area')

## 1-d
x_2 = x[:,[0,int(min_choice[0])]]
beta_2 = coefficient(x_2,y)
y_es_2 = np.dot(x_2,beta_2)

plt.figure()
sns.regplot(y_es_2, y-y_es_2, lowess=True, line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'None', 'edgecolors':'k', 'alpha':0.5})
plt.hlines(0, xmin=min(y_es_2), xmax=max(y_es_2), linestyles='dotted')

plt.title("Residual Plot for Linear Fit")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()

plt.figure()
sns.regplot(y_es_1, y-y_es_1, lowess=True, line_kws={'color':'r', 'lw':1},
            scatter_kws={'facecolors':'None', 'edgecolors':'k', 'alpha':0.5})
plt.hlines(0, xmin=min(y_es_1), xmax=max(y_es_1), linestyles='dotted')
plt.title("Residual Plot for Quadratic Fit")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()
