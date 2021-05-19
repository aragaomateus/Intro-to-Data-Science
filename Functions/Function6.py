#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:19:27 2021

@author: aragaom
"""

import numpy as np
data = np.genfromtxt('/Users/aragaom/Downloads/sampleInput1.csv',missing_values=None,delimiter=',')
data2 = np.genfromtxt('/Users/aragaom/Downloads/sampleInput2.csv',missing_values=None,delimiter=',')

def empiricalSampleBounds(data, bounds):
    data = sorted(data)
    bound_1 = (100 -bounds)/2 
    bound_2 = (100-(bound_1))
    bound_1 *= len(data)/100
    bound_2 *= len(data)/100
    lower_bound = data[int(bound_1-1)]
    upper_bound = data[int(bound_2-1)]
    return lower_bound,upper_bound
    
mu = empiricalSampleBounds(data, 95)
mu1 = empiricalSampleBounds(data, 99)
mu2 = empiricalSampleBounds(data, 50)
mu3 = empiricalSampleBounds(data2, 95)
mu4 = empiricalSampleBounds(data2, 99)
mu5 = empiricalSampleBounds(data2, 50)