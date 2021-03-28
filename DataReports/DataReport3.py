#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:03:27 2021

@author: aragaom
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn import linear_model # library for multiple linear regression
from scipy import stats

data = pd.read_csv('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Data Science code/movieRatingsDeidentified.csv')
titanic = data['Titanic (1997)'].to_numpy(dtype='float')
starWars1 = data['Star Wars: Episode I - The Phantom Menace (1999)'].to_numpy(dtype='float')
# starWars2 = data['Star Wars: Episode II - Attack of the Clones (2002)'].to_numpy(dtype='float')
#%%
import numpy as np
import pandas as pd
from sklearn import linear_model # library for multiple linear regression
from scipy import stats
data = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Data Science code/movieRatingsDeidentified.csv',missing_values=None,usecols=np.arange(0,209),delimiter=',')
titanic = data[:,197]
starWars1 = data[:,156]
starWars2 = data[:,157]
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
def normalizedError(array,flag):
    y = array[:,0]
    y_pred =array[:,1]
    abs_sum = 0
    if flag == 1:    
        # the mean absolute error, 
        # MAE = (1/n) * Σ|yi – xi|
        for i in range(len(array[0])):
            abs_sum += abs(array[0,i] - array[1,i])
        result = abs_sum/len(array[0])
    if flag == 2:
        # 2 for the RMSE, 
        for i in range(len(array[0])):
            abs_sum += abs(array[0,i] - array[1,i])**2
        result = (((abs_sum)/len(array[0]))**(1/2))
    if flag == 3:
       # cubic root mean cubed error and so on.
       for i in range(len(array[0])):
           abs_sum += abs(array[0,i] - array[1,i])**3 
       result = (((abs_sum)/len(array[0]))**(1/3))
    return result
#%%
# correto
mask = ~np.isnan(starWars1) & ~np.isnan(titanic) 
res = stats.linregress(starWars1[mask],titanic[mask])
#1
rsquare = res.rvalue**2  #0.06249403072076677
#2
beta_titanic = res.slope #0.23552863613359326
#7
size=titanic.size
counter=0
for i in range(size):
    if not np.isnan(titanic[i]) and not np.isnan(starWars1[i]):
        counter+=1# 1625
sum_of_rating1 = counter

#8
array1 = np.array([titanic[mask],starWars1[mask]])
rmse1 = normalizedError(array1,2)
#1.4478362610149247
#%%
mask = ~np.isnan(starWars2) & ~np.isnan(starWars1)
res2 = stats.linregress(starWars2[mask],starWars1[mask])
#3
r_value = res2.rvalue
rsquare_star = res2.rvalue**2 #0.6524118588448808
#4
# # counter = 0
# # for i in range(0,size):
# #     if not np.isnan(starWars2[i]) and not np.isnan(starWars1[i]):
# #         counter+=1

# sum_of_rating2 = counter
# 1625
#5
beta_starwars1 = res2.slope #0.8082746072425062

#%%
mask = ~np.isnan(starWars1) & ~np.isnan(titanic) 
inputData = np.transpose([starWars1[mask],titanic[mask]]) # lead levels, income
output = simple_linear_regress_func(inputData) # output returns m,b,r^2
y = titanic[mask] # actual income
yHat = output[0]*starWars1[mask] + output[1] # predicted income
residuals1 = y - yHat
array2 = np.array([yHat,y])
rmse_titanic = normalizedError(array2,2)
#8) 1.05
#%%
mask = ~np.isnan(starWars2) & ~np.isnan(starWars1)
inputData = np.transpose([starWars2[mask],starWars1[mask]]) # lead levels, income
output = simple_linear_regress_func(inputData) # output returns m,b,r^2
y =starWars1[mask] # actual income
yHat = output[0]*starWars2[mask] + output[1] # predicted income
residuals1 = y - yHat
array = np.array([yHat,y])
rmse_starwar = normalizedError(array,2)
# 0.6670924466723792
#%%

def big_Pascualian_Function(x,y):
    counter=0
    for i in range(len(x)):
        if not np.isnan(x[i]) and not np.isnan(y[i]):
            counter+=1# 1625
    mask = ~np.isnan(x) & ~np.isnan(y)
    inputData = np.transpose([x[mask],y[mask]]) # lead levels, income
    output = simple_linear_regress_func(inputData) # output returns m,b,r^2
    Y = y[mask]# actual income
    yHat = output[0]*x[mask] + output[1] # predicted income
    array = np.array([yHat,Y])
    rmse = normalizedError(array,2)
    return counter,output[0],output[2],rmse

output2 = big_Pascualian_Function(starWars2,starWars1)
output1 = big_Pascualian_Function(starWars1,titanic)
