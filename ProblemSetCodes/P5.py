#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:00:16 2021

@author: aragaom
"""
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot as meansPlot
klonopin = pd.read_csv("/Users/aragaom/Downloads/klonopin.txt", delim_whitespace=True)
# klonopin = np.genfromtxt('/Users/aragaom/Downloads/klonopin.txt')

#%%
# You study the effectiveness of Klonopin, a benzodiazepine on anxiety symptoms 
# of people who are already in therapy for anxiety. You measure anxiety with the 
# State-Trait Anxiety Inventory (STAI) after these participants receive a dose of 
# Klonopin. All participants are randomly assigned to a dosage level. The dosage 
# levels are 0, 0.125, 0.25, 0.5, 1 and 2 mg. This data is contained in the “Klonopin.txt”
# file. You analyze the results with an ANOVA.


# A. 1 factor, 6 levels

aov = pg.anova(data=klonopin, dv='anxiety', between='dosage', detailed=True)
print(aov)
# model = ols('' data=klonopin_data).fit() #Build the two-way ANOVA model. Value = y, X1,X2 = Main effects. X1:X2 = interaction effect
# anova_table = sm.stats.anova_lm(model, typ=3) #Create the ANOVA table. Residual = Within
# print(anova_table) #Show the ANOVA table
# print(model.summary())

anova = f_oneway(klonopin_0,klonopin_01,klonopin_02,klonopin_05,klonopin_1,klonopin_2)
# model = ols('0 ~ 1', data=klonopin).fit() #Build the two-way ANOVA model. Value = y, X1,X2 = Main effects. X1:X2 = interaction effect
# anova_table = sm.stats.anova_lm(model, typ=2) #Create the ANOVA table. Residual = Within
# print(anova_table) 
#%%
# What is true about degrees of freedom in this design?
# B. There are 5 between group and 144 within group degrees of freedom

#%%

# What is the error variance of this design, in terms of mean squared deviations?

# D. 818

#%%

# What do you conclude about the effectiveness of Klonopin for the treatment of anxiety, based on this study?
# E. Klonopin is likely highly effective for the treatment of anxiety because we are able to reject the null hypothesis at a level of p < 0.001.
#%%

# Looking at the means plot (have a look at the code in the last section of the relevant lab to see how to make 
# one) suggests that
model = ols('anxiety ~ dosage', data=klonopin).fit() #Build the two-way ANOVA model. Value = y, X1,X2 = Main effects. X1:X2 = interaction effect
anova_table = sm.stats.anova_lm(model, typ=2) #Create the ANOVA table. Residual = Within
print(anova_table) #Show the ANOVA table

# D. There seems to be a threshold effect - dosages below 0.5 mg don't seem to do much, but it seems to be increasingly effective above that.
#Show the corresponding means plot
fig = meansPlot(x=klonopin['anxiety'], trace=klonopin['dosage'], response=klonopin['dosage'])
#%%

# You want to study the effects of enriched environments on cortical development. 
# To do so, you raise rats in a carefully prepared environment. You randomly assign 
# rats to one of 3 social housing conditions (alone, with one companion, with 2 companions) 
# and one of 4 exercise conditions (no exercise, 30 minutes a day, 60 minutes a day and 90 minutes a day). 
# After 6 months of being housed like that, you measure the cortical thickness (in micrometers) of all 
# participating rats. This data is contained in “rats.txt”. Column 1 contains the dependent variable 
# - cortical thickness, Column 2 contains the social housing condition and column 3 contains the exercise condition. 

rats = pd.read_csv("/Users/aragaom/Downloads/rats.txt", delim_whitespace=True)
model = ols('cortical ~ housing + exercise + housing:exercise', data=rats).fit() #Build the two-way ANOVA model. Value = y, X1,X2 = Main effects. X1:X2 = interaction effect
anova_table = sm.stats.anova_lm(model, typ=2) #Create the ANOVA table. Residual = Within
aov = pg.anova(data=rats, dv='cortical', between=['housing','exercise'], detailed=True)
print(aov)
print(anova_table)
# How many conditions and factors does this design involve?
# C. 2 factors, 12 conditions
#%%
# What is true about degrees of freedom in this design?
# E. 3 between group df for exercise, 2 between group df for housing, 12 within group df.

#%%
# Given this design, there are this many rats in each "cell" (condition)
# D. 2
#%%
# What do you conclude about the effects of exercise and social housing on the 
# development of cortical thickness in rats?
# D. In all likelihood, both social housing and exercise have a positive effect on cortical thickness, but there doesn’t seem to be an interaction effect.
#%%
# y
# Click to see additional instructions
# What follows is a numeric fill in the blank question with 1 blanks. You are 
# interested in the stress levels experienced by undergraduate students. 
# The dependent measure is the level of cortisol (a stress hormone) in saliva 
# while giving a presentation. You do study with 4 factors, 2 levels each: 
# Sleep (more vs. less than 7 hours per night), number of classes taken 
# (more or less than 3 classes per semester), disposition (introvert vs. extravert) 
# and social setting (giving the presentation alone vs. in front of an entire 
# classroom). For this and the next 3 questions, use the socialstress.txt datafile. 
# The first column contains the cortisol level in ug/dl, the second column the amount 
# of sleep, the third column the number of classes taken, the fourth column the disposition 
# (introvert = 0, extravert = 1), the fifth column the social setting (alone = 0, in front of class = 1). 
# Rows represent individual participants, who have been randomly and independently drawn from the 
# undergraduate population. 
formula = ['cortisol ~ sleep + number_of_classes + disposition + setting + sleep * disposition + number_of_classes * disposition+sleep * setting+ number_of_classes * setting+disposition * setting + sleep * number_of_classes * disposition+sleep * number_of_classes * setting + sleep * disposition * setting+ number_of_classes * disposition * setting +  sleep * number_of_classes * disposition * setting +   Residual ']
rats = pd.read_csv("/Users/aragaom/Downloads/socialstress.txt", delim_whitespace=True)
model = ols('cortisol ~ sleep + number_of_classes + disposition + setting + sleep * disposition + number_of_classes * disposition+sleep * setting+ number_of_classes * setting+disposition * setting + sleep * number_of_classes * disposition+sleep * number_of_classes * setting + sleep * disposition * setting+ number_of_classes * disposition * setting +  sleep * number_of_classes * disposition * setting', data=rats).fit() #Build the two-way ANOVA model. Value = y, X1,X2 = Main effects. X1:X2 = interaction effect
# anova_table = sm.stats.anova_lm(model, typ=2) #Create the ANOVA table. Residual = Within
aov = pg.anova(data=rats, dv='cortisol', between=['sleep','number_of_classes','disposition','setting'], detailed=True)
print(aov)
print(model.summary())
sumario = model.summary()
# print(anova_table)
# Your model explains   Blank 2. Fill in the blank, read surrounding text. percent of the variance 
# 43.7
#%%
# There are   Blank 2. Fill in the blank, read surrounding text. significant interaction effects 
 #  1 
#%%
# There are   Blank 2. Fill in the blank, read surrounding text. significant main effects  
# 2
#%%
# You want to summarize your results for a press release. 
# The statement that would most appropriately capture your findings 
# (most specific *and* accurate) is
# A. When it comes to student stress levels, sleep duration and numbers of classes taken seem to matter, whereas it depends on your level of extraversion whether the social setting stresses students out or not.

#%%

# You write a blog and are curious if there are any trends whatsoever
#  (audience fatigue, getting more exposure, etc.). This data – over the year – 
#  is in the datafile “blogData.txt”. Every visitor is logged as a number, and 
#  the number represents the month of the year in which they visited the blog. 
#  You do a chi-square test to see if there are any trends or if people were equally
#  likely to visit each month.
from scipy import stats
blog_data = pd.read_csv("/Users/aragaom/Downloads/blogData.txt", header=None)

def categorical_Counter(data):
    import numpy as np
     # declaring the dictionary categories, where the variables are the keys
     # and the frequency of the variable are the values
    categories = {}
    for variable in data:
        if variable not in categories.keys():
            categories[variable] = 1
        elif variable in categories.keys():
            categories[variable] = categories.get(variable) + 1
    categories = np.array(sorted(categories.items()))
    return categories
data2 = np.genfromtxt('/Users/aragaom/Downloads/blogData.txt')

data = categorical_Counter(data2)
expected = sum(data[:,1]/365)
expected = [expected for  i in range(0,12)]
chisqrTest = chisquare(f_obs=data[:,1],f_exp=expected)
# chisqrTest = stats.chi2_contingency(observed=data )
# What is true about this scenario?
# B. There were over 100,000 visitors to the blog in general, and there are 11 degrees of freedom.
#%%
# What do you conclude about the likelihood of people visiting equally often 
# throughout the year? (Pick the strongest statement you can justifiably make, 
# given the data.)
# D. It is likely that the likelihood of people visiting the blog throughout the year is constant, as we fail to reject the null hypothesis at p = 0.05.

#%%
# You run the maternity ward of a hospital and want to know if you should
#  staff it equally during all days of the year or if there are days on which 
#  you should put more people on staff. There is all kinds of folk knowledge 
#  about moon cycles and the like, so you record the number of births over the 
#  course of a year, then do a chi-square test on the data. This data is stored
#  in “births.txt” – each birth was logged by the day of the year on which it 
#  happened.
birth = np.genfromtxt('/Users/aragaom/Downloads/births.txt')
birthCat = categorical_Counter(birth)
expected = sum(birthCat[:,1]/365)
expected = [expected for  i in range(0,365)]
chisqrTest_birth = stats.chi2_contingency(observed=birthCat )
chisqrTest_birth2 = chisquare(f_obs= birthCat[:,1],f_exp=expected )
# f_obs= observed,   # Array of observed counts
#                 f_exp= expected
# What is true about this scenario?
# E. There were over 35,000 births that year and the number of degrees of freedom is 364.
#%%
# What do you conclude about the likelihood of giving birth over the course 
# of the year?

# E. As we fail to reject the null hypothesis at p = 0.05, we maintain that it seems that births are equally likely for a given day, over the course of the year.
#%%
# You want to know whether people who win the lottery are happier than those
# who didn't. To answer this question, you ask 30 people who won the lottery
#  to rate how happy they are (on a scale from 1 to 10, with 10 representing 
# extremely happy and 1 extremely unhappy). You also ask 30 age and gender
#  matched - but otherwise randomly picked - individuals to rate how happy they are. 
#  This data is in the file "happiness.txt". The first column represents the happiness 
#  ratings, the second column whether they won the lottery (1) or not (0).
# What is the right test to run to test this hypothesis?
#E. Mann Whitney U test


#%%

# You do a t-test on this data. You conclude that
happiness = np.genfromtxt('/Users/aragaom/Downloads/happiness.txt')

experiment = []
experiment = np.array(experiment)

control = []
control = np.array(control)

for i in range(len(happiness)):
    if (happiness[i][1] ==  1):
        experiment = np.append(experiment, happiness[i][0])
    if (happiness[i][1] ==  0):
        control = np.append(control, happiness[i][0])
    
t_test = stats.ttest_ind(control, experiment)

# B. This is a moot point. A t-test is really an inappropriate test for this kind of data, so it is irrelevant what the outcome of the test is. We should not use it as a basis to make conclusions about the effect of winning the lottery on happiness.#%%
#%%
# You do a Mann-Whitney U test on this data. You conclude that
u_test = stats.mannwhitneyu(control, experiment)
# A. p < 0.05, so it is unlikely that the two sample means are different just by chance. Therefore, we can conclude that winning the lottery has an effect on happiness levels.