#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:19:27 2021

@author: aragaom
"""

def empiricalSampleBounds(file_input, bounds):
    import numpy as np
    from math import erf, sqrt
    data = np.genfromtxt(file_input,missing_values=None,delimiter=',')
    data = sorted(data)
    bound_1 = (100 -bounds)/2 
    bound_2 = (100-(bound_1))
    bound_1 *= len(data)/100
    bound_2 *= len(data)/100
    lower_bound = data[int(bound_1-1)]
    upper_bound = data[int(bound_2-1)]
    return lower_bound,upper_bound
    
mu = empiricalSampleBounds('/Users/aragaom/Downloads/sampleInput1.csv', 95)
mu1 = empiricalSampleBounds('/Users/aragaom/Downloads/sampleInput1.csv', 99)
mu2 = empiricalSampleBounds('/Users/aragaom/Downloads/sampleInput1.csv', 50)