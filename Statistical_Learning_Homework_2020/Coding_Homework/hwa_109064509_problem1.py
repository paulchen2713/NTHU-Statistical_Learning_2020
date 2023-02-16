import csv
import numpy as np
import math
from matplotlib import pyplot as plt 


'''
problem 1(a) start from line 13
problem 1(b) start from line 45
problem 1(c) start from line 77
'''

### problem 1(a)
N=160
X=np.ones([N,7])
y=np.ones(N)
with open('covid-19.csv') as f:
    rows = csv.reader(f)
    next(rows)
    i=0
    for row in rows :
        X[i,1:7]=row[1:7]
        y[i]=row[7]
        i+=1
beta=np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()),y)
#print(beta)

y_es=np.dot(X,beta)
RSS=pow(y_es-y,2).sum()
var=RSS/(N-2)
#print(var)
temp=np.linalg.inv(np.dot(X.transpose(),X))*var
#print(temp)
SE=np.zeros(7)
t=np.zeros(7)
for i in range(0,7):
    SE[i]=math.sqrt(temp[i][i])
    t[i]=beta[i]/SE[i]
    #print('SE_',i,'=',SE[i])
    #print('t_',i,'=',t[i])



### problem 1(b)

#Find the two predictors that yield the smallest RSS
RSS2=pow(y-y.sum()/160,2).sum()
mi=0
mj=0
for i in range(1,7):
    for j in range(i+1,7):
        X2=X[:,[0,i,j]]
        beta2=np.dot(np.dot(np.linalg.inv(np.dot(X2.transpose(),X2)),X2.transpose()),y)
        y2_es=np.dot(X2,beta2)
        m=pow(y2_es-y,2).sum()
        if m<RSS2:           
            RSS2=m
            mi=i
            mj=j
X2=X[:,[0,mi,mj]]
beta2=np.dot(np.dot(np.linalg.inv(np.dot(X2.transpose(),X2)),X2.transpose()),y)
#print(beta2)
y2_es=np.dot(X2,beta2)
var2=RSS2/(N-2)
#print(var2)
temp2=np.linalg.inv(np.dot(X2.transpose(),X2))*var2
SE2=np.zeros(7)
t2=np.zeros(7)
for i in range(0,3):
    SE2[i]=math.sqrt(temp2[i][i])
    t2[i]=beta2[i]/SE2[i]
    #print('SE_',i,'=',SE2[i])
    #print('t_',i,'=',t2[i])


### problem 1(c)

lam=[]
for i in range(0,17):
    lam.append(pow(10,-5+i*0.5))

D1=X[0:32,1:]
D2=X[32:64,1:]
D3=X[64:96,1:]
D4=X[96:128,1:]
D5=X[128:160,1:]
Xc=np.zeros([N,6])
Xc1=np.zeros([32,6])
Xc2=np.zeros([32,6])
Xc3=np.zeros([32,6])
Xc4=np.zeros([32,6])
Xc5=np.zeros([32,6])
yc=np.zeros(N)
yc1=np.zeros(32)
yc2=np.zeros(32)
yc3=np.zeros(32)
yc4=np.zeros(32)
yc5=np.zeros(32)
for i in range(0,6):
    Xc[:,i]=X[:,i+1]-X[:,i+1].sum()/N
    Xc1[:,i]=D1[:,i]-D1[:,i].sum()/32
    Xc2[:,i]=D2[:,i]-D2[:,i].sum()/32
    Xc3[:,i]=D3[:,i]-D3[:,i].sum()/32
    Xc4[:,i]=D4[:,i]-D4[:,i].sum()/32
    Xc5[:,i]=D5[:,i]-D5[:,i].sum()/32
yc=y-y.sum()/N
yc1=y[0:32]-y[0:32].sum()/32
yc2=y[32:64]-y[32:64].sum()/32
yc3=y[64:96]-y[64:96].sum()/32
yc4=y[96:128]-y[96:128].sum()/32
yc5=y[128:160]-y[128:160].sum()/32
beta_D1=np.zeros([6,17])
beta_D2=np.zeros([6,17])
beta_D3=np.zeros([6,17])
beta_D4=np.zeros([6,17])
beta_D5=np.zeros([6,17])
y_D1=np.zeros([32,17])
y_D2=np.zeros([32,17])
y_D3=np.zeros([32,17])
y_D4=np.zeros([32,17])
y_D5=np.zeros([32,17])
I=np.identity(6)
CVE=[]
for i in range(0,17):
    beta_D1[:,i]=np.dot(np.dot(np.linalg.inv(np.dot(Xc1.transpose(),Xc1)+lam[i]*I),Xc1.transpose()),yc1)
    beta_D2[:,i]=np.dot(np.dot(np.linalg.inv(np.dot(Xc2.transpose(),Xc2)+lam[i]*I),Xc2.transpose()),yc2)
    beta_D3[:,i]=np.dot(np.dot(np.linalg.inv(np.dot(Xc3.transpose(),Xc3)+lam[i]*I),Xc3.transpose()),yc3)
    beta_D4[:,i]=np.dot(np.dot(np.linalg.inv(np.dot(Xc4.transpose(),Xc4)+lam[i]*I),Xc4.transpose()),yc4)
    beta_D5[:,i]=np.dot(np.dot(np.linalg.inv(np.dot(Xc5.transpose(),Xc5)+lam[i]*I),Xc5.transpose()),yc5)
    y_D1[:,i]=np.dot(Xc1,beta_D1[:,i])
    y_D2[:,i]=np.dot(Xc2,beta_D2[:,i])
    y_D3[:,i]=np.dot(Xc3,beta_D3[:,i])
    y_D4[:,i]=np.dot(Xc4,beta_D4[:,i])
    y_D5[:,i]=np.dot(Xc5,beta_D5[:,i])
    CVE.append((pow(y_D1[:,i]-yc1,2).sum()+pow(y_D2[:,i]-yc2,2).sum()+pow(y_D3[:,i]-yc2,2).sum()+pow(y_D4[:,i]-yc4,2).sum()+pow(y_D5[:,i]-yc5,2).sum())/160)
    
#Choose lambda = 10^-1.5, find lambda that minimize CVE
m=CVE[0]
for i in range(0,17):
    if CVE[i]<=m:
        lbd=lam[i]
        m=CVE[i]
        mi=i

#Find the two predictors that yield the smallest RSS
I=np.identity(2)
RSSr2=pow(yc-yc.sum()/160,2).sum()
mi=0
mj=0
for i in range(0,6):
    for j in range(i+1,6):
        Xr2=Xc[:,[i,j]]
        betar2=np.dot(np.dot(np.linalg.inv(np.dot(Xr2.transpose(),Xr2)+lbd*I),Xr2.transpose()),yc)
        yr2_es=np.dot(Xr2,betar2)
        m=pow(yr2_es-yc,2).sum()
        if m<RSSr2:           
            RSSr2=m
            mi=i
Xr2=Xc[:,[mi,mj]]
betar2=np.dot(np.dot(np.linalg.inv(np.dot(Xr2.transpose(),Xr2)+lbd*I),Xr2.transpose()),yc)
#print(betar2)

#plot CVE v.s. lambda
for i in range(0,17):
    CVE[i]=10*math.log(CVE[i],10)
    lam[i]=10*math.log(lam[i],10)

plt.title('CV v.s. tuning parameter v')
plt.xlabel('v(dB)')
plt.ylabel('CV(dB)')
plt.plot(lam,CVE) 
plt.show()