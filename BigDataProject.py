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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from scipy import stats as stat
import statsmodels.api as sm


sns.set_theme(style="whitegrid")
data = pd.read_csv('/Users/aragaom/Downloads/middleSchoolData.csv')
HSPHS_acceptances  = data['acceptances']

#%%

# The correlation between applications and acceptences. 
#1) What is the correlation between the number of applications and admissions to HSPHS?
applications_acceptances  = data[['applications','acceptances']]
# applications_acceptances.dropna()
r = np.corrcoef(applications_acceptances,rowvar=False)
x = data['applications']
y = data['acceptances']

# Plot the data:
plt.imshow(r) 
plt.colorbar()
pearson = applications_acceptances.corr('pearson')
print(applications_acceptances.corr('pearson'))
sns.regplot(x, y, ci=None).set_title(' Figure Q1')

#%%

# 2)What is a better predictor of admission to HSPHS? Raw number of applications or
# application *rate*?
# correlation bet rate and the raw number _ bruce
# should I run a simple regression for it ? 
# have a scatterplot for it
# for this we need to run 2 linear regressions to one with acceptance and 
# raw application umbers and the other rate of apps and the acceptance, check the 
# check the r^2 to see if which one of the models is a better predictor.
# 
def simple_linear_regress_func(data):
    # Load numpy:
    import numpy as np
    # Initialize container:
    temp_cont = np.empty([len(data),5]) # initialize empty container, n x 5
    temp_cont[:] = np.NaN # convert to NaN
    # Store data in container:
    for i in range(len(data)):
        temp_cont[i,0] = data[i,0] # x 
        temp_cont[i,1] = data[i,1] # y 
        temp_cont[i,2] = data[i,0]*data[i,1] # x*y
        temp_cont[i,3] = data[i,0]**2 #x^2
        temp_cont[i,4] = data[i,1]**2 # y^2
    # Compute numerator and denominator for m:
    m_numer = len(data)*sum(temp_cont[:,2]) - sum(temp_cont[:,0])*sum(temp_cont[:,1])
    m_denom = len(data)*sum(temp_cont[:,3]) - (sum(temp_cont[:,0]))**2
    m = m_numer/m_denom
    # Compute numerator and denominator for b:
    b_numer = sum(temp_cont[:,1]) - m * sum(temp_cont[:,0])
    b_denom = len(data)
    b = b_numer/b_denom
    # Compute r^2:
    temp = np.corrcoef(data[:,0],data[:,1]) # pearson r
    r_sqr = temp[0,1]**2 # L-7 weenie it (square)
    # Output m, b & r^2:
    output = np.array([m, b, r_sqr])
    return output 


rates = data['applications'] / data['school_size']
rates = rates.to_frame(name='rates' )
raw_application = data['applications']
raw_application=raw_application.to_frame(name='raw')
rates_app  = pd.concat([rates,data['acceptances']], axis=1).dropna().to_numpy()
raw_app  = pd.concat([raw_application,data['acceptances']], axis=1).dropna().to_numpy()

output = simple_linear_regress_func(rates_app) # run custom SLR function
rSqr1 = output[2]
x1 = rates 
y1 = data['acceptances']
sns.regplot(y1, x1, ci=None).set_title(' Figure Q2.1')



output = simple_linear_regress_func(raw_app) # run custom SLR function
rSqr2 = output[2]
x2 = data['applications']
y2 = data['acceptances']
sns.regplot(x2, y2, ci=None).set_title(' Figure Q2.2')


# raw number of applications is better 
# rates_raw_correlation = rates_raw.corr('pearson')


#%%

# 3)Which school has the best *per student* odds of sending someone to HSPHS?
# have a scatterplot for it

amount_of_students = data['school_size']
application_per_student = data['applications'] / amount_of_students

acceptances_per_student = data['acceptances'] / amount_of_students

acceptances_per_student = pd.DataFrame(acceptances_per_student)
acceptances_per_student = acceptances_per_student.to_numpy()
# index  = acceptances_per_student.columns.valuesindex

high_odds = data['school_name'].iloc[maxOdd]
odds = (data['acceptances']/(data['applications']-data['acceptances']))

maxOdd = np.argmax(odds)
for i in acceptances_per_student:
    if i == maxOdd:
        print (i)
sns.regplot(amount_of_students,data['acceptances'] , ci=None).set_title(' Figure Q2.2')
sns.regplot(data['applications'],odds , ci=None).set_title(' Figure Q3')

# THE CHRISTA MCAULIFFE SCHOOL\I.S. 187 index 304
# print(acceptances_per_student.max())

#%%

# 4) Is there a relationship between how students perceive their school (as reported in columns
# L-Q) and how the school performs on objective measures of achievement (as noted in
# columns V-X).
# have a scatterplot for it
# hypothesis testinf
# pca to get the independe info and then corarltion

students_perceive = data[['rigorous_instruction','collaborative_teachers','supportive_environment','effective_school_leadership','strong_family_community_ties','trust']].dropna()

zscoredData = stats.zscore(students_perceive)

# Run the PCA:
pca = PCA()
pca.fit(zscoredData)

# eig_vals: Single vector of eigenvalues in decreasing order of magnitude
eigVals = pca.explained_variance_

# loadings: Weights per factor in terms of the original data. Where do the
# principal components point, in terms of the 17 questions?
loadings = pca.components_

# rotated-data: Simply the transformed data - we had 40 courses (rows) in
# terms of 17 variables (columns), now we have 40 courses in terms of 17
# factors ordered by decreasing eigenvalue
rotatedData = pca.fit_transform(zscoredData)
nDraws = 10000 # How many repetitions per resampling?
numRows = 540 # How many rows to recreate the dimensionality of the original data?
numColumns = 6 # How many columns to recreate the dimensionality of the original data?
eigSata = np.empty([nDraws,numColumns]) # Initialize array to keep eigenvalues of sata
eigSata[:] = np.NaN # Convert to NaN

for i in range(nDraws):
    # Draw the sata from a normal distribution:
    sata = np.random.normal(0,1,[numRows,numColumns]) 
    # Run the PCA on the sata:
    pca = PCA()
    pca.fit(sata)
    # Keep the eigenvalues:
    temp = pca.explained_variance_
    eigSata[i] = temp
    
plt.plot(np.linspace(1,numColumns,numColumns),eigVals,color='blue') # plot eig_vals from section 4
plt.plot(np.linspace(1,numColumns,numColumns),np.transpose(eigSata),color='black') # plot eig_sata
plt.plot([1,numColumns],[1,1],color='red') # Kaiser criterion line
plt.xlabel('Principal component (SATA)')
plt.ylabel('Eigenvalue of SATA')
plt.legend(['data','sata'])
# For the purposes of this, you can think of eigenvalues in terms of 
# (co)variance explained
# covarExplained = eigVals/sum(eigVals)*100
# numClasses = 6
# #plt.bar(np.linspace(1,num_classes,num_classes),eig_vals)
# plt.plot(eigVals)
# plt.xlabel('Principal component')
# plt.ylabel('Eigenvalue')
# plt.plot([0,numClasses],[1,1],color='red',linewidth=1) # Kaiser criterion line

plt.bar(np.linspace(1,6,6),loadings[:,0])
plt.xlabel('Question')
plt.ylabel('Loading')



objective_measures = data[['student_achievement','reading_scores_exceed','math_scores_exceed']].dropna()
zscoredData2 = stats.zscore(objective_measures)

# Run the PCA:
pca2 = PCA()
pca2.fit(zscoredData2)

# eig_vals: Single vector of eigenvalues in decreasing order of magnitude
eigVals2 = pca2.explained_variance_

# loadings: Weights per factor in terms of the original data. Where do the
# principal components point, in terms of the 17 questions?
loadings2 = pca2.components_

# rotated-data: Simply the transformed data - we had 40 courses (rows) in
# terms of 17 variables (columns), now we have 40 courses in terms of 17
# factors ordered by decreasing eigenvalue
rotatedData = pca2.fit_transform(zscoredData2)

rotatedData = pca.fit_transform(zscoredData2)
nDraws = 10000 # How many repetitions per resampling?
numRows = 547 # How many rows to recreate the dimensionality of the original data?
numColumns = 3 # How many columns to recreate the dimensionality of the original data?
eigSata = np.empty([nDraws,numColumns]) # Initialize array to keep eigenvalues of sata
eigSata[:] = np.NaN # Convert to NaN

for i in range(nDraws):
    # Draw the sata from a normal distribution:
    sata = np.random.normal(0,1,[numRows,numColumns]) 
    # Run the PCA on the sata:
    pca = PCA()
    pca.fit(sata)
    # Keep the eigenvalues:
    temp = pca.explained_variance_
    eigSata[i] = temp
    
plt.plot(np.linspace(1,numColumns,numColumns),eigVals2,color='blue') # plot eig_vals from section 4
plt.plot(np.linspace(1,numColumns,numColumns),np.transpose(eigSata),color='black') # plot eig_sata
plt.plot([1,numColumns],[1,1],color='red') # Kaiser criterion line
plt.xlabel('Principal component (SATA)')
plt.ylabel('Eigenvalue of SATA')
plt.legend(['data','sata'])

plt.bar(np.linspace(1,3,3),loadings2[:,0])
plt.xlabel('Question')
plt.ylabel('Loading')
plt.bar(np.linspace(1,3,3),loadings2[:,1])
plt.xlabel('Question')
plt.ylabel('Loading')

x = data['student_achievement']
y = data['supportive_environment']

# m, b = np.polyfit(x, y, 1)

# line = give_me_a_straight_line(x,y)

# plt.scatter(x, y, alpha=0.5)
# plt.title('Scatter plot pythonspot.com')
# plt.plot(x,m*x+b)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
sns.regplot(x, y, ci=None).set_title(' Figure Q4')


correlation4 = data[['student_achievement','supportive_environment']]

correlation4 = correlation4.corr('pearson')
 # i have found no correlation. 
# For the purposes of this, you can think of eigenvalues in terms of 
# (co)variance explained
# covarExplained2 = eigVals2/sum(eigVals2)*100
# numClasses = 3
# #plt.bar(np.linspace(1,num_classes,num_classes),eig_vals)
# plt.plot(eigVals2)
# plt.xlabel('Principal component2')
# plt.ylabel('Eigenvalue2')
# plt.plot([0,numClasses],[1,1],color='red',linewidth=1) # Kaiser criterion line

#%%
# 5) Test a hypothesis of your choice as to which kind of school (e.g. small schools vs. large
# schools or charter schools vs. not (or any other classification, such as rich vs. poor school)) 
# performs differently than another kind either on some dependent measure, e.g. objective measures of 
# achievement or admission to HSPHS (pick one).
# -- should I do a ANOVA with it? 
# have a scatterplot for it

# initial hypostheisi that having rigorous instructors does not effect on student achievements on 
# on standarized test. 
# we reject null hypothesis
rigorous_instruction_data = data.rigorous_instruction.to_numpy()

achievement_data = data.student_achievement.to_numpy()

new_set = []
new_set = np.array(new_set)

new_set2 = []
new_set2 = np.array(new_set2)
for i in range(len(rigorous_instruction_data)):
    if not np.isnan(rigorous_instruction_data[i]) and not np.isnan(achievement_data[i]):
        new_set = np.append(new_set,rigorous_instruction_data[i])
        new_set2 = np.append(new_set2,achievement_data[i])

clear_rigorous_instruction_data = pd.DataFrame(data = new_set,columns=['rigorous_instruction'])
clear_achievement_data = pd.DataFrame(data = new_set2, columns=['achievement'])
combined_data = pd.concat([clear_rigorous_instruction_data,clear_achievement_data],axis = 1)
combined_data = combined_data.to_numpy()

less_rigorous = []
less_rigorous = np.array(less_rigorous)

very_rigorous = []
very_rigorous = np.array(very_rigorous)

for i in range(len(combined_data)):
    if (combined_data[i][0] < 3.65):
        less_rigorous = np.append(less_rigorous, combined_data[i][1])
    if (combined_data[i][0] > 3.65):
        very_rigorous = np.append(very_rigorous, combined_data[i][1])
        
        
t_test = stat.ttest_ind(very_rigorous,less_rigorous)

mean_very_rigorous = sum(very_rigorous)/len(very_rigorous)
mean_less_rigorous = sum(less_rigorous)/len(less_rigorous)

sns.regplot(data.rigorous_instruction, data.student_achievement, ci=None).set_title(' Figure Q5')

        
# mask = ~np.isnan(rigorous_instruction_data) & ~np.isnan(achievement_data) 
# rigorous_instruction_data = rigorous_instruction_data[mask]
# achievement_data = achievement_data[mask]


#%%

# 6) Is there any evidence that the availability of material resources (e.g. per student spending 
# or class size) impacts objective measures of achievement or admission to HSPHS?
# --correlation?
# have a scatterplot for it
acceptance_spending = data[['acceptances','per_pupil_spending']]
correlation_accept_spending = acceptance_spending.corr('pearson')
sns.regplot(data.acceptances, data.per_pupil_spending, ci=None).set_title(' Figure Q6.1')


acceptance_classSize = data[['acceptances','avg_class_size']]
correlation_accept_classSize= acceptance_classSize.corr('pearson')
sns.regplot(data.acceptances, data.avg_class_size, ci=None).set_title(' Figure Q6.2')

achievement_spending = data[['student_achievement','per_pupil_spending']]
correlation_achievement_spending = acceptance_spending.corr('pearson')
sns.regplot(data.student_achievement, data.per_pupil_spending, ci=None).set_title(' Figure Q6.3')

achievement_classSize = data[['student_achievement','avg_class_size']]
correlation_achievement_classSize= acceptance_classSize.corr('pearson')
sns.regplot(data.student_achievement, data.avg_class_size, ci=None).set_title(' Figure Q6.4')

math_spending = data[['math_scores_exceed','per_pupil_spending']].dropna()
correlation5= math_spending.corr('pearson')
sns.regplot(data.math_scores_exceed, data.per_pupil_spending, ci=None).set_title(' Figure Q6.5')

math_spending = data[['math_scores_exceed','avg_class_size']].dropna()
correlation6= math_spending.corr('pearson')
sns.regplot(data.math_scores_exceed, data.avg_class_size, ci=None).set_title(' Figure Q6.6')

math_spending = data[['reading_scores_exceed','per_pupil_spending']].dropna()
correlation7= math_spending.corr('pearson')
sns.regplot(data.reading_scores_exceed, data.per_pupil_spending, ci=None).set_title(' Figure Q6.7')

math_spending = data[['reading_scores_exceed','avg_class_size']].dropna()
correlation8= math_spending.corr('pearson')
sns.regplot(data.reading_scores_exceed, data.avg_class_size, ci=None).set_title(' Figure Q6.8')

#%%

# 7) What proportion of schools accounts for 90% of all students accepted to HSPHS?
# --(plot a normal distribution? and find it)
# *For question 7, a bar graph of schools, rank-ordered by decreasing number of 
# acceptances of students to HSPHS would be nice to see.

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
    categories = np.array(sorted(categories.items(),key=operator.itemgetter(1),reverse=False))
    return categories

ranks = categorical_Counter(HSPHS_acceptances)
sorted(ranks)
# ranks = pd.DataFrame(data = ranks,columns = ['ranks','frequency'] )
# data2  = pd.concat([rankss,data['acceptances']], axis=1).dropna().to_numpy()

# fig = plt.figure(figsize = (10, 10))
# plt.bar(ranks[:,1],ranks[:,0],width = 1)
# plt.xlabel("Courses offered")
# plt.ylabel("No. of students enrolled")
# plt.title("Students enrolled in different courses")
# plt.show()
soma = sum(HSPHS_acceptances)
sort = HSPHS_acceptances.sort_values(ascending=False)
ranks = pd.DataFrame(data = sort,columns = ['acceptances'])


add = 0
count = 0
order = np.arange(1,len(sort)+1,1).transpose()
for i in range(len(sort)):
    if add/soma <= .90:
        add += sort[i]
        count += 1

proportion = count/len(sort)
# ranks = pd.DataFrame(data = sort , columns = ['acceptances'])
# ranks.plot(y = 'acceptances',  kind = 'bar')
y = np.arange(len(sort))
array =  np.arange(len(HSPHS_acceptances))
array[::-1].sort()

plt.plot( y, sort, color = 'red' )
plt.bar(np.linspace(1,len(sort),len(sort)),sort, color = 'red')
ax = sns.barplot( x = array,y = HSPHS_acceptances, data=HSPHS_acceptances,order =HSPHS_acceptances.sort_values())
ax.set(xlabel='common xlabel', ylabel='common ylabel')
sns.histplot(x=sort, stat="frequency", bins=500,binwidth=3)

plt.xticks()
plt.show()
# order = pd.DataFrame(data = order , columns = ['order'])
# data2  = pd.concat([ranks,order], axis=1)


# ax = sns.barplot( x = 'order',y = 'acceptances', data=data2)
# ax = sns.distplot(sort, norm_hist=True)                                    
# yy = stats.norm.pdf(HSPHS_acceptances)                                                         
# ax.plot(HSPHS_acceptances, yy, 'r', lw=2)

#%%
sort = HSPHS_acceptances.sort_values(ascending=False)
ranks = pd.DataFrame(data = sort,columns = ['acceptances'])
plt.bar(np.linspace(1,len(sort),len(sort)),sort, color = 'red')


#%%


# 8) Build a model of your choice – clustering, classification or prediction – that includes all
#factors – as to what school characteristics are most important in terms of 
# a) sending students to HSPHS,
# b) achieving high scores on objective measures of achievement?
# run a pca first 
acceptance_model_predictors = data[['per_pupil_spending','avg_class_size', 'rigorous_instruction','collaborative_teachers' , 'supportive_environment' ,'strong_family_community_ties', 'trust' , 'poverty_percent','ESL_percent','school_size','student_achievement','reading_scores_exceed','math_scores_exceed']].dropna()
zscoredData = stats.zscore(acceptance_model_predictors)
pca = PCA()
pca.fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.fit_transform(zscoredData)
nDraws = 10000 # How many repetitions per resampling?
numRows = 449 # How many rows to recreate the dimensionality of the original data?
numColumns = 13 # How many columns to recreate the dimensionality of the original data?
eigSata = np.empty([nDraws,numColumns]) # Initialize array to keep eigenvalues of sata
eigSata[:] = np.NaN # Convert to NaN
for i in range(nDraws):
    # Draw the sata from a normal distribution:
    sata = np.random.normal(0,1,[numRows,numColumns]) 
    # Run the PCA on the sata:
    pca = PCA()
    pca.fit(sata)
    # Keep the eigenvalues:
    temp = pca.explained_variance_
    eigSata[i] = temp
plt.plot(np.linspace(1,numColumns,numColumns),eigVals,color='blue') # plot eig_vals from section 4
plt.plot(np.linspace(1,numColumns,numColumns),np.transpose(eigSata),color='black') # plot eig_sata
plt.plot([1,numColumns],[1,1],color='red') # Kaiser criterion line
plt.xlabel('Principal component (SATA)')
plt.ylabel('Eigenvalue of SATA')
plt.legend(['data','sata'])

plt.bar(np.linspace(1,13,13),loadings[:,0])
plt.xlabel('Question')
plt.ylabel('Loading')

plt.bar(np.linspace(1,13,13),loadings[:,1])
plt.xlabel('Question')
plt.ylabel('Loading')

plt.bar(np.linspace(1,13,13),loadings[:,2])
plt.xlabel('Question')
plt.ylabel('Loading')
# 'per_pupil_spending','collaborative_teachers'  per_pupil_spending+ collaborative_teachers +
acceptance_model = data[['acceptances','rigorous_instruction','per_pupil_spending','strong_family_community_ties','reading_scores_exceed','student_achievement']].dropna()
model_lin = sm.OLS.from_formula("acceptances ~ rigorous_instruction +per_pupil_spending+strong_family_community_ties+  reading_scores_exceed+ student_achievement", data=acceptance_model)
result_lin = model_lin.fit()
print(result_lin.summary())


scores_model = data[['student_achievement','per_pupil_spending','avg_class_size', 'rigorous_instruction','collaborative_teachers' , 'supportive_environment','school_size','reading_scores_exceed','math_scores_exceed']].dropna()
model_lin = sm.OLS.from_formula("student_achievement ~ per_pupil_spending + avg_class_size + rigorous_instruction + collaborative_teachers + supportive_environment+school_size + reading_scores_exceed + math_scores_exceed", data=scores_model)
result_lin = model_lin.fit()
print(result_lin.summary())




#%%
# 9) Write an overall summary of your findings – what school characteristics seem to be most
# relevant in determining acceptance of their students to HSPHS?

#%%
# 10) Imagine that you are working for the New York City Department of Education as a data
# scientist (like one of my former students). What actionable recommendations would you make 
# on how to improve schools so that they 
# a) send more students to HSPHS and 
# b) improve objective measures or achievement.
