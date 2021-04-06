#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:24:14 2021

@author: aragaom
"""

import numpy as np
from scipy import stats as stat



sadex1 = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Sadex Analysis/Sadex1.txt')
sadex2 = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Sadex Analysis/Sadex2.txt')
sadex3 = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Sadex Analysis/Sadex3.txt')
sadex4 = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Sadex Analysis/Sadex4.txt')


#%%
#21 to 25
sadex_experiment = []
sadex_experiment = np.array(sadex_experiment)

sadex_control = []
sadex_control = np.array(sadex_control)

for i in range(len(sadex1)):
    if (sadex1[i][1] ==  1):
        sadex_experiment = np.append(sadex_experiment, sadex1[i][0])
    if (sadex1[i][1] ==  0):
        sadex_control = np.append(sadex_control, sadex1[i][0])
    
t_test = stat.ttest_ind(sadex_control, sadex_experiment)

mean_of_control = sum(sadex_control)/len(sadex_control)
mean_of_experiment = sum(sadex_experiment)/len(sadex_experiment)
#%%
#26 e 28
sadex_before= []
sadex_before = np.array(sadex_before)

sadex_after = []
sadex_after = np.array(sadex_after)

for i in range(len(sadex2)):
    sadex_before = np.append(sadex_before, sadex2[i][0])

    sadex_after = np.append(sadex_after, sadex2[i][1])

mean_of_before = sum(sadex_before)/len(sadex_before)
mean_of_after = sum(sadex_after)/len(sadex_after)
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t
 
# function for calculating the t-test for two independent samples
def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# calculate standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p
independent_ttest(sadex_before, sadex_after, 0.05)

t_test_bna = stat.ttest_rel(sadex_before, sadex_after)
#%%
# 29
sadex_experiment = []
sadex_experiment = np.array(sadex_experiment)

sadex_control = []
sadex_control = np.array(sadex_control)

for i in range(len(sadex3)):
    if (sadex3[i][1] ==  1):
        sadex_experiment = np.append(sadex_experiment, sadex3[i][0])
    if (sadex3[i][1] ==  0):
        sadex_control = np.append(sadex_control, sadex3[i][0])
    


t_test = stat.ttest_ind(sadex_control, sadex_experiment)
#%%
# 30
sadex_before= []
sadex_before = np.array(sadex_before)

sadex_after = []
sadex_after = np.array(sadex_after)

for i in range(len(sadex4)):
    sadex_before = np.append(sadex_before, sadex4[i][0])

    sadex_after = np.append(sadex_after, sadex4[i][1])

mean_of_before = sum(sadex_before)/len(sadex_before)
mean_of_after = sum(sadex_after)/len(sadex_after)

t_test_before_and_after = stat.ttest_ind( sadex_before, sadex_after)

