# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 19:30:03 2018

@author: Yun
"""

import numpy as np
x = np.array([[1, 5], [1, 5],[1, 3]])
y = np.array([[8], [8],[7]])
ans = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(x),x)),np.transpose(x)),y)

# =============================================================================
# B= 1.737、1.5
# =============================================================================
x1 = np.array([[1, 3], [1, 5],[1,10]])
ans2 = np.dot(np.linalg.pinv(np.dot(np.transpose(x1),x1)),2.02)
#1.907是我算出來的RSS=3.64除以(3-1-1)後開根號得1.907
#ans2 的對角元素=分別為B0 B1的std err 
x3 =  np.array([[10], [3],[5]])
y3 = np.array([[15], [7],[8]])
import statsmodels.api as sm
x4 = sm.add_constant(x3)
est = sm.OLS(y3, x4)
est2 = est.fit()
print(est2.summary())###########...ANS