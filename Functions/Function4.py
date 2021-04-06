#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:43:03 2021

@author: aragaom
"""
import numpy as np
def catCounter(data):
    categories = {}
    counter = 0 
    for variable in data:
        if variable not in categories.keys():
            categories[variable] = 1
        elif variable in categories.keys():
            categories[variable] = categories.get(variable) + 1
    categories = sorted(categories.items())
    return categories


data1 = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/F4/catCounterInput1.csv')
data2 = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/F4/catCounterInput2.csv')

array = [1,2,1,2,1]
array2 = [7,7,7,7.5,7.6]

val = catCounter(array)
val2 = catCounter(array2)
val3 = catCounter(data1)
val4 = catCounter(data2)