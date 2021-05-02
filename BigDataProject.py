#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:29:31 2021

@author: aragaom
"""

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import operator
sns.set_theme(style="whitegrid")
data = pd.read_csv('/Users/aragaom/Downloads/middleSchoolData.csv')
HSPHS_acceptances  = data['acceptances']

#%%
# The correlation between applications and acceptences. 
#1) What is the correlation between the number of applications and admissions to HSPHS?
applications_acceptances  = data[['applications','acceptances']]
# applications_acceptances.dropna()
r = np.corrcoef(applications_acceptances,rowvar=False)

# Plot the data:
plt.imshow(r) 
plt.colorbar()
pearson = applications_acceptances.corr('pearson')
print(applications_acceptances.corr('pearson'))
print(applications_acceptances.corr('spearman'))
# sns.scatterplot(data=applications_acceptances, x="acceptances", y="applications")

#%%
# 2)What is a better predictor of admission to HSPHS? Raw number of applications or
# application *rate*?
# should I run a simple regression for it ? 
# have a scatterplot for it

#%%

# 3)Which school has the best *per student* odds of sending someone to HSPHS?
# have a scatterplot for it

amount_of_students = data['school_size']
application_per_student = data['applications'] / amount_of_students

acceptances_per_student = data['acceptances'] / amount_of_students

acceptances_per_student = pd.DataFrame(acceptances_per_student)
index  = acceptances_per_student.columns.valuesindex
print(acceptances_per_student.max())
#%%

# 4) Is there a relationship between how students perceive their school (as reported in columns
# L-Q) and how the school performs on objective measures of achievement (as noted in
# columns V-X).
# have a scatterplot for it

students_perceive = data[['rigorous_instruction','collaborative_teachers','supportive_environment','effective_school_leadership','strong_family_community_ties','trust']]

objective_measures = data[['student_achievement','reading_scores_exceed','math_scores_exceed']]

#%%
# 5) Test a hypothesis of your choice as to which kind of school (e.g. small schools vs. large
# schools or charter schools vs. not (or any other classification, such as rich vs. poor school)) 
# performs differently than another kind either on some dependent measure, e.g. objective measures of 
# achievement or admission to HSPHS (pick one).
# -- should I do a ANOVA with it? 
# have a scatterplot for it


#%%

# 6) Is there any evidence that the availability of material resources (e.g. per student spending 
# or class size) impacts objective measures of achievement or admission to HSPHS?
# --correlation?
# have a scatterplot for it


#%%

# 7) What proportion of schools accounts for 90% of all students accepted to HSPHS?
# --(plot a normal distribution? and find it)
# *For question 7, a bar graph of schools, rank-ordered by decreasing number of 
# acceptances of students to HSPHS would be nice to see.
HSPHS_acceptances = HSPHS_acceptances.to_numpy()
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
    categories = np.array(sorted(categories.items(),key=operator.itemgetter(1),reverse=True))
    return categories

ranks = categorical_Counter(HSPHS_acceptances)

plt.bar(ranks[:,0],ranks[:,1],color ='maroon',
        width = 0.4)
plt.xlabel("Courses offered")
plt.ylabel("No. of students enrolled")
plt.title("Students enrolled in different courses")
plt.show()

#%%

# 8) Build a model of your choice – clustering, classification or prediction – that includes all
#factors – as to what school characteristics are most important in terms of 
# a) sending students to HSPHS,
# b) achieving high scores on objective measures of achievement?
ranks = categorical_Counter(HSPHS_acceptances)
fig = plt.figure(figsize = (10, 10))
plt.bar(ranks[:,1],ranks[:,0],width = 1)
plt.xlabel("Courses offered")
plt.ylabel("No. of students enrolled")
plt.title("Students enrolled in different courses")
plt.show()
#%%
# 9) Write an overall summary of your findings – what school characteristics seem to be most
# relevant in determining acceptance of their students to HSPHS?

#%%
# 10) Imagine that you are working for the New York City Department of Education as a data
# scientist (like one of my former students). What actionable recommendations would you make 
# on how to improve schools so that they 
# a) send more students to HSPHS and 
# b) improve objective measures or achievement.
