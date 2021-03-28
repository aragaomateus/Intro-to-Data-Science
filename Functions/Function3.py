#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:34:33 2021

@author: aragaom
"""
import numpy as np

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

array = np.array([[1,2,3,4,5],[2,4,1,6,2]])
result1 = normalizedError(array,1)
result2 = normalizedError(array,2)
result3 = normalizedError(array,3)