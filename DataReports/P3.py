#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:41:42 2021

@author: aragaom
"""
#loading data and importing libs
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model


data = np.genfromtxt('/Users/aragaom/Desktop/Lab5/kepler.txt')


#%%
# 21) There are concerns that the IQ test is not culturally fair, 
# privileging members of higher castes. You look for evidence of 
# this and determine that the correlation between caste membership and IQ is


#%% 2) Simple linear regression between IQ and income
import os,sys
from scipy import stats
import numpy as np
from io import StringIO  
import pandas as pd

# 0. Load:
d = pd.read_table("/Users/aragaom/Desktop/NYU Classes/Lab5/kepler.txt", sep="\t", header= None)
data = d[[0, 4]].to_numpy(dtype='float')

# D1 = np.mean(data,axis=0) # take mean of each column
# D2 = np.median(data,axis=0) # take median of each column
# D3 = np.std(data,axis=0) # take std of each column
# D4 = np.corrcoef(data[:,0],data[:,1]) # pearson r = 0.44

# # 2. Visualize:
# import matplotlib.pyplot as plt 
# plt.plot(data[:,0],data[:,1],'o',markersize=.75) # Plot IQ against income
# m, b = np.polyfit(data[:,0], data[:,1], 1)
# plt.plot(data[:,0], m*data[:,0] + b)
# plt.xlabel('IQ') 
# plt.ylabel('Brain mass')  
#%%
def simple_linear_regress_func(data):
    # Load numpy:
    import numpy as np
    # Initialize container:
    temp_cont = np.empty([len(data),5]) # initialize empty container, n x 5
    temp_cont[:] = np.NaN # convert to NaN
    # Store data in container:
    for i in range(len(data)):
        temp_cont[i,0] = data[i,0] # x 
        temp_cont[i,1] = data[i,1] # y 
        temp_cont[i,2] = data[i,0]*data[i,1] # x*y
        temp_cont[i,3] = data[i,0]**2 #x^2
        temp_cont[i,4] = data[i,1]**2 # y^2
    # Compute numerator and denominator for m:
    m_numer = len(data)*sum(temp_cont[:,2]) - sum(temp_cont[:,0])*sum(temp_cont[:,1])
    m_denom = len(data)*sum(temp_cont[:,3]) - (sum(temp_cont[:,0]))**2
    m = m_numer/m_denom
    # Compute numerator and denominator for b:
    b_numer = sum(temp_cont[:,1]) - m * sum(temp_cont[:,0])
    b_denom = len(data)
    b = b_numer/b_denom
    # Compute r^2:
    temp = np.corrcoef(data[:,0],data[:,1]) # pearson r
    r_sqr = temp[0,1]**2 # L-7 weenie it (square)
    # Output m, b & r^2:
    output = np.array([m, b, r_sqr])
    return output 

#%% 4) Partial correlation
# We want to know the relationship between income and violent incidents.
# Lead levels are a known confounding variable, so we want to control for this.
# Logic: do 2 simple linear regressions for lead levels vs. income and lead levels vs.
# violent incidents, then correlate the residuals; this is the partial correlation

# 0. Load:
data = d[[0, 1,3]].to_numpy(dtype='float')# income, violent incidents, lead levels

# 1. Descriptives: 
D1 = np.mean(data,axis=0) # take mean of each column
D2 = np.median(data,axis=0) # take median of each column
D3 = np.std(data,axis=0) # take std of each column

# 2. Correlation between income and violence:
r = np.corrcoef(data[:,0],data[:,1]) # pearson r = -0.49

# 3. Do the partial correlation via 2 simple linear regressions:
inputData = np.transpose([data[:,2],data[:,0]]) # lead levels, income
output = simple_linear_regress_func(inputData) # output returns m,b,r^2
y = data[:,0] # actual income
yHat = output[0]*data[:,2] + output[1] # predicted income
residuals1 = y - yHat # compute residuals

inputData = np.transpose([data[:,2],data[:,1]]) # lead levels, violent incidents
output2 = simple_linear_regress_func(inputData) # output returns m,b,r^2
y = data[:,1] # actual violent incidents
yHat = output[0]*data[:,2] + output[1] # predicted violent incidents
residuals2 = y - yHat # compute residuals

# 4. Correlate the residuals:
part_corr = np.corrcoef(residuals1,residuals2) # r = -0.09
rsquered = part_corr*part_corr
#%%
import numpy as np
from sklearn import linear_model # library for multiple linear regression
import matplotlib.pyplot as plt 
# .5134 
data = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Lab5/kepler.txt')

X = np.transpose([data[:,0],data[:,3],data[:,4]]) # IQ, hours worked, years education
Y = data[:,1] # income
regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 
rSqr = regr.score(X,Y) # 0.48 - realistic - life is quite idiosyncratic
betas = regr.coef_ # m
yInt = regr.intercept_  # b
y_hat = betas[0]*data[:,1] + betas[1]*data[:,3] +betas[2]*data[:,4]+ yInt
plt.plot(y_hat,data[:,0],'o',markersize=.75) # y_hat, income
plt.xlabel('Prediction from model') 
plt.ylabel('Actual income')  
plt.title('R^2: {:.3f}'.format(rSqr)) 
#%%
# 3. Regression:
# By hand - also saved as a custom function that we will use later
regressContainer = np.empty([len(data),5]) # initialize empty container, 1000x5
regressContainer[:] = np.NaN # convert to NaN
for i in range(len(data)):
    regressContainer[i,0] = data[i,0] # IQ
    regressContainer[i,1] = data[i,1] # income
    regressContainer[i,2] = data[i,0]*data[i,1] # IQ * income
    regressContainer[i,3] = data[i,0]**2 # IQ squared
    regressContainer[i,4] = data[i,1]**2 # income squared
    
# Compute m ("slope"):
mNumerator = len(data)*sum(regressContainer[:,2]) - sum(regressContainer[:,0])*sum(regressContainer[:,1])
mDenominator = len(data)*sum(regressContainer[:,3]) - (sum(regressContainer[:,0]))**2
m = mNumerator/mDenominator

# Compute b ("y-intercept):
bNumerator = sum(regressContainer[:,1]) - m * sum(regressContainer[:,0])
bDenominator = len(data)
b = bNumerator/bDenominator
rSquared = D4[0,1]**2 

# 4. Add regression line to visualization:
yHat = m*data[:,0] + b # slope-intercept form, y = mx + b
plt.plot(data[:,0],yHat,color='orange',linewidth=0.5) # orange line for the fox
plt.title('R^2: {:.3f}'.format(rSquared)) # add title, r-squared rounded to nearest thousandth

