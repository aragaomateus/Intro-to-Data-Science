#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:05:51 2021

@author: aragaom
"""

def bayesCalculator(probA,probB,likilyBA,flag):
    result = 0
    notA = 1 - probA
    likilyBnotA = (probB - (probA  * probB)) / notA
    if flag is 1:
        # simple.
        result = (probA*likilyBA)/probB
    if flag is 2:
        #explicity.
        result = (probA*likilyBA)/((probA*likilyBA)+(likilyBnotA*notA))
    return result

# print (bayesCalculator(0.01,0.1,0.99,1))
# print (bayesCalculator(0.01,0.1,0.99,2))
# print (bayesCalculator(0.0001,0.01,1,1))
# print (bayesCalculator(0.06,0.1,0.95,2))
# print (bayesCalculator(0.3775,0.33,0.9,2))
# print (bayesCalculator(0.05,0.0625,1,1))
# print (bayesCalculator(0.05,0.0625,0.8,1))
# print (bayesCalculator(0.3,0.01,0.95,2))
# print (bayesCalculator(0.3,0.25,0.5,2))
print (bayesCalculator(0.0001,0.05,0.95,2))
print (bayesCalculator(0.1,0.9,0.99,1))
print (bayesCalculator(0.48,0.4,0.65,2))
print (bayesCalculator(0.03,0.05,0.6,2))
print (bayesCalculator(0.6,0.55,0.9,2))
print (bayesCalculator(0.999,0.01,0.99,2))
print (bayesCalculator(0.1,0.3,0.8,2))
print (bayesCalculator( 0.9 ,0.3, 0.7,2))
print (bayesCalculator( 0.25 ,0.2, 0.7,1))
print (bayesCalculator(  (1/1000000),(1/10000),.9,2))
print (bayesCalculator(  (1/1000000),(1/1000000),.99,2))
print (bayesCalculator( 0.3,0.1,.9,2))
print (bayesCalculator( 0.013,0.2,.8,2))

