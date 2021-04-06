#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:30:42 2021

@author: aragaom
"""

# data report 2
from scipy import stats
import numpy as np
import pandas as pd

sata = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Data Science code/movieRatingsDeidentified.csv',missing_values=None,usecols=np.arange(0,209),delimiter=',')

x=sata[:,0]
y=sata[:,1]
a=sata[:,62]
b=sata[:,125]
c=sata[:,109]
d=sata[:,110]
e=sata[1]
f=sata[3204]
g=sata[:,156]
h=sata[:,157]

# 1)
print(stats.spearmanr(x,y, nan_policy = 'omit'))
# 4)
print(stats.spearmanr(a,b, nan_policy = 'omit'))
# 5)
print(stats.spearmanr(c,d, nan_policy = 'omit'))

#8)
print(stats.spearmanr(e,f, nan_policy = 'omit'))

#10)
print(stats.spearmanr(g,h, nan_policy = 'omit'))

sata =pd.read_csv('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Data Science code/movieRatingsDeidentified.csv')
print(sata)
cols = sata.columns[sata.dtypes.eq(object)]
sata[cols] = sata[cols].apply(pd.to_numeric,errors = 'coerce')
# 2
print(sata['Donnie Darko (2001)'].corr(sata['Mulholland Dr. (2001)']))
# 3
print(sata['Kill Bill: Vol. 1 (2003)'].corr(sata['Kill Bill: Vol. 2 (2004)']))
# sata = sata.corr(method='pearson')
# 6
print(sata['10 things I hate about you (1999)'].corr(sata['12 Monkeys (1995)']))

#7
print(sata.iloc[0].corr(sata.iloc[3203]))

print(sata['Star Wars: Episode I - The Phantom Menace (1999)'].corr(sata['Star Wars: Episode II - Attack of the Clones (2002)']))



#%%

correlationn = sata.corr(method = "pearson" )
correlationS = sata.corr(method = "spearman" )
rotated = sata.T
# correlation2 = rotated.corr(method = "pearson" )
# correlationS2 = rotated.corr(method = "spearman" )## it takes so long.

#%%
#%% We could have also done it directly with logic:
from scipy import stats
import numpy as np
import pandas as pd

temp = np.copy(sata) #Make a copy of A
temp = temp[~np.isnan(temp).any(axis=1)] #Remove all rows where there are missing values
#Then do the same as in lines 41-42:
# r = np.corrcoef(temp[:,0],temp[:,1]) #Calculate the Pearson correlation between the surviving 4 rows of column 1 and column 2
# print('r =',r) #No more nans. Note the value is different than in line 14, as we deleted some rows
# #Same outcome as in 42.
# matrix_correlation = [np.corrcoef(sata[:,i],sata[:,j].T) for i in range(len(sata[1])-1)for j in range(len(sata[1])-1)]
correlation = np.corrcoef(temp, temp,rowvar=False )
# correlation = sata.corr(method = "pearson" )
# correlationS = sata.corr(method = "spearman" )
# rotated = sata.T
# correlation2 = rotated.corr(method = "pearson" )
# correlationS2 = rotated.corr(method = "spearman" )