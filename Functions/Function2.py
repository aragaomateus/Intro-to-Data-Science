#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:39:45 2021

@author: aragaom
"""
import numpy as np
import matplotlib.pyplot as plt
def slidWinDescStats(array, flag, window):
    answer = []
    if flag == 1:
        i = 0
        end = window
        while end < len(array)+1:
            tempArr = array[i:end]
            answer.append(np.mean(tempArr))
            i+=1
            end+=1
    if flag == 2:
        i = 0
        end = window        
        while end < len(array)+1:
            tempArr = array[i:end]
            answer.append(np.nanstd(tempArr,ddof = 1))
            i+=1
            end+=1
    if flag == 3:     
        i= 0
        end = window
        while end < len(array[0])+1:
            tempArr = array[0][i:end]
            tempArr2 = array[1][i:end]
            answer.append(np.corrcoef(tempArr,tempArr2))
            i+=1
            end+=1
        answer = [i[0][1] for i in answer]
    return answer
        

array = [1,3,5,7,9]
array2 = [[1,3,5,7,9],[1,2,4,3,5]]
arrayfile = np.genfromtxt('/Users/aragaom/Downloads/outputArrayExample300.csv',missing_values=None,delimiter=',')
arrayfile2 = np.genfromtxt('/Users/aragaom/Downloads/outputArrayExample300.csv',missing_values=None,delimiter=',')
m = np.vstack((arrayfile, arrayfile2.T))

output = slidWinDescStats(array2,3,3)
plt.plot(output)

