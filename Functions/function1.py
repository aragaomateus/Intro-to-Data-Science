#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 00:50:20 2021

@author: aragaom
"""

import numpy as np

arr = np.array([1,3,5,7,9])
arr1 = np.array([4.5,3.2,1.5,5.7,9.3,2.2,6.9])

def medAbsDev(arr):
    median = np.median(arr)
    sumDev = 0
    for i in arr:
        abso=np.absolute(i-median)
        sumDev = sumDev + abso
    return sumDev/len(arr)
        
print(medAbsDev(arr))
print(medAbsDev(arr1))

