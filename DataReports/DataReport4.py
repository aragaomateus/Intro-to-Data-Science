#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:55:51 2021

@author: aragaom
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn import linear_model # library for multiple linear regression
from scipy import stats
data = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Data Science code/movieRatingsDeidentified.csv',missing_values=None,usecols=np.arange(0,209),delimiter=',')
kill_bill_1 = data[:,109]
kill_bill_2 = data[:,110]
pulp_fiction = data[:,136]

temp = np.array([np.isnan(kill_bill_1),np.isnan(kill_bill_2),np.isnan(pulp_fiction)],dtype=bool)
temp2 = temp*1 # convert boolean to int
temp2 = sum(temp2) # take sum of each participant
missingData = np.where(temp2>0) # find participants with missing data
kill_bill_1 = np.delete(kill_bill_1,missingData) # delete missing data from array
kill_bill_2 = np.delete(kill_bill_2,missingData) # delete missing data from array
pulp_fiction = np.delete(pulp_fiction,missingData)
#%%
# Doing an independent samples t-test between ratings from Kill Bill 1 and Pulp Fiction, what is the p-value?

ttest_KB1_PF = stats.ttest_ind(kill_bill_1, pulp_fiction)
# p value = 4.083848614270736e-17

#%%
# Doing a paired samples t-test between ratings from Kill Bill 1 and Kill Bill 2, what is the p-value? 

ttest_paired_KB1_KB2 = stats.ttest_rel(kill_bill_1, kill_bill_2)
# p-value: 5.646322558291671e-06

#%%
# What are the degrees of freedom for a *paired samples* t-test between Kill Bill 1 and Kill Bill 2?
df_KB1_KB2_paired = len(kill_bill_1) - 1
# p-value: 1310

#%%
# Doing an independent samples t-test between ratings from Kill Bill 1 and Kill Bill 2, what is the t-value?  

ttest_KB1_KB2 = stats.ttest_ind(kill_bill_1, kill_bill_2)

# t-value:1.9356232385154668

#%%
# When using a paired samples t-test, is there a significant difference between the ratings for Kill Bill 1 and Kill Bill 2?

# C. Yes, we can reject the null hypothesis

#%%
# What is the median rating of Kill Bill 1 in this population (of people who have rated all 3 movies)?  
median_KB2 = np.median(kill_bill_1)
# 3.5

#%%
# Doing an independent samples t-test between ratings from Kill Bill 1 and Pulp Fiction, what is the absolute value of the t-value? 

# t-vaalue = 8.4683711289626

#%%
# When using an independent samples t-test, is there a significant difference between the ratings for Kill Bill 1 and Pulp Fiction?
# A. Yes, we can reject the null hypothesis

#%%
# When using an independent samples t-test, is there a significant difference between the ratings for Kill Bill 1 and Kill Bill 2?
# B. No, we fail to reject the null hypothesis

#%%
# What are the degrees of freedom for an *independent samples* t-test between Kill Bill 1 and Kill Bill 2?  
df_KB1_KB2 =  len(kill_bill_1) + len(kill_bill_2) - 2
# df =2620

#%%
# Doing an independent samples t-test between ratings from Kill Bill 1 and Kill Bill 2, what is the p-value?  

# p-value = 0.053021327459578244

#%%
# Doing a paired samples t-test between ratings from Kill Bill 1 and Kill Bill 2, what is the t-value? 
# t-value = 4.558001060041254
#%%
# How many users have rated all 3 of these movies?  
howMany= (len(kill_bill_1) + len(kill_bill_2) + len(pulp_fiction))/3
# 1311.0
#%%
# When using a Mann-Whitney U-test, is there a significant difference between the ratings for Kill Bill 2 and Pulp Fiction?
u_test = stats.mannwhitneyu(kill_bill_2,pulp_fiction)
# D. Yes, we can reject the null hypothesis
#%%
# What is the mean rating of Pulp Fiction in this population (of people who have rated all 3 movies)?  
mean_pulpF = np.mean(pulp_fiction)
# 3.403203661327231