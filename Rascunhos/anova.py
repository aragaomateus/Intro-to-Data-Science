import numpy as np
from scipy.stats import f_oneway
data = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Data Science code/movieRatingsDeidentified.csv',missing_values=None,usecols=np.arange(0,209),delimiter=',')

titanic = data[:,197]
starWars1 = data[:,156]
starWars2 = data[:,157]

mask = ~np.isnan(starWars1) & ~np.isnan(starWars2)


output = f_oneway(starWars1[mask],starWars2[mask],axis=0)
import matplotlib.pyplot as plt
from scipy.stats import f
fig, ax = plt.subplots(1, 1)
dfn, dfd = 2,15
x = np.linspace(f.ppf(0.01, dfn, dfd),f.ppf(0.99, dfn, dfd), 100)
ax.plot(x, f.pdf(x, dfn, dfd),'r-', lw=5, alpha=0.6, label='f pdf')
plt.axvline(x=3.68, label='Critical value for alpha=0.05', color='g')
plt.axvline(x=output[0], label='F-score')
plt.legend()
#%%
import statsmodels.api as sm
from statsmodels.formula.api import ols
data = np.genfromtxt('/Users/aragaom/Desktop/NYU Classes/Intro to Data Science Class/Data Science code/movieRatingsDeidentified.csv',missing_values=None,usecols=np.arange(0,209),delimiter=',')

titanic = data[:,197]
starWars1 = data[:,156]
starWars2 = data[:,157]

mask = ~np.isnan(starWars1) & ~np.isnan(starWars2)
mod = ols(data=[starWars1[mask],starWars2[mask]]).fit()
                
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)